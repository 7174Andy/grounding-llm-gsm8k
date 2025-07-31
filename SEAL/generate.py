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

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"  # Adjust based on your available GPUs'

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
    print("Loading Data...")
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
    
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + "Question: " + example["question"].strip()+"\nAnswer: "

        # Apply chat template for Llama-3.1
        messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # Generate answers
    print("Generating answers...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir="/opt/huggingface_cache",
            trust_remote_code=True,
        )
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)

    outputs = [[trim_output(o) for o in output] for output in result]

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in zip(test_data, outputs, prompts)]

    with open(os.path.join("SEAL", f"{MODEL_ID}_SEAL", "predictions.json"), "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

if __name__ == "__main__":
    print("Starting SEAL generation script...")
    print(f"Using model: {MODEL_ID}")
    torch.cuda.empty_cache()
    main()