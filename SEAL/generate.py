import os
from tqdm import tqdm
import re
from pathlib import Path
import random
import json

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # Adjust based on your available GPUs'

def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

def main():
    random.seed(42)

    # Prepare dataset
    test_data = []
    data = load_dataset("gsm8k", "main", split="test")
    data = data.shuffle(seed=42).select(range(1000))  # Use a subset for testing
    for example in tqdm(data, desc="Processing examples"):
        answer = example["answer"].split("####")[1].strip()
        answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
        test_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[0].strip(),
            "gt": answer,
        })
    
    # Setup Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prefix = "Answer the following questions with step by step reasoning including transition and reflection thoughts if needed, then state your final answer as 'Answer: <NUMBER>' on a new line.\n\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + example["question"].strip()

        # Apply chat template for Llama-3.1
        if "Llama" in MODEL_ID:
            messages = [{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        elif "qwen" in MODEL_ID.lower():
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # Generate answers
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

    model = LLM(MODEL_ID, dtype=torch_dtype, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), max_model_len=512+2000, download_dir="/opt/huggingface_cache", gpu_memory_utilization=0.60)
    params = SamplingParams(temperature=0.0, max_tokens=1000, top_p=1.0, top_k=-1, stop=None)
    
    print("Model loaded successfully.")
    outputs = model.generate(prompts, params)
    decoded_outputs = [trim_output(output.outputs[0].text) for output in outputs]

    print("Generation completed.")

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in tqdm(zip(test_data, decoded_outputs, prompts), desc="Creating predictions", total=len(test_data))]

    # Save predictions to output directory
    model_name_for_dir = MODEL_ID.split("/")[-1]
    output_dir = os.path.join(f"{model_name_for_dir}_SEAL")
    os.makedirs(output_dir, exist_ok=True)

    # Record predictions to the desired output directory
    with open(os.path.join(output_dir, "baseline_predictions.json"), "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

if __name__ == "__main__":
    print("Starting SEAL generation script...")
    print(f"Using model: {MODEL_ID}")
    torch.cuda.empty_cache()
    main()