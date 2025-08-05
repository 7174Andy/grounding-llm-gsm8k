import os
from tqdm import tqdm
import re
from pathlib import Path
import random
import json
import time

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

from modeling_qwen2 import Qwen2ForCausalLM
from modeling_gemma import GemmaForCausalLM
from tqdm import trange

MODEL_ID = "google/gemma-2-9b-it"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,6"


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans

def main():
    # Load the dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(132))

    test_data = []
    for example in tqdm(dataset, desc="Processing examples"):
        answer = example["answer"].split("####")[1].strip()
        answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
        test_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[0].strip(),
            "gt": answer,
        })

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare prompts
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in tqdm(enumerate(test_data), total=len(test_data), desc="Preparing prompts"):
        prompt = prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if "Llama" in MODEL_ID:
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
        else:
            messages = [{"role": "user", "content": example["question"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    
    # Load the steering vector
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if "Qwen" in MODEL_ID:
        model = Qwen2ForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/opt/huggingface_cache",
        )
    elif "gemma" in MODEL_ID:
        model = GemmaForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir="/opt/huggingface_cache",
            trust_remote_code=True,
        )
    else: 
        raise ValueError(f"Unsupported model ID: {MODEL_ID}")
    layer = 20 # adjust for layer
    alpha = 1.0 # adjust for alpha
    steer_vec = torch.load(
        f"hidden_analysis_{MODEL_ID.split('/')[-1]}/vector_transition_reflection_steervec/layer_{layer}_transition_reflection_steervec.pt",
        weights_only=True
    ).to(torch_dtype)

    print(f"Steering vector loaded for layer {layer} with alpha {alpha}")
    model.set_steering_flag(steering_flag=True, steering_layer=layer, steer_vec=steer_vec, steer_coef=alpha, tokenizer=tokenizer)
    model.start_new_round()
    print("Model loaded successfully.")

    start_time = time.time()
    # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # generated_tokens = model.generate(**inputs, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_tokens = []
    input_lengths = []
    prompt = prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text for first prompt: {generated_text}")

    # for i in trange(0, len(prompts), 8):
    #     model.start_new_round()
    #     batch_prompts = prompts[i:i+8]
    #     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    #     batch_input_lens = [len(x) for x in inputs['input_ids']]
    #     input_lengths.extend(batch_input_lens)
    #     with torch.no_grad():
    #         generated_batch = model.generate(**inputs, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    #     generated_tokens.extend(generated_batch)

    end_time = time.time()
    print("Generation completed.")
    print(f"Generation time: {end_time - start_time:.4f} seconds")

    # Process outputs
    decoded_outputs = []
    for i, output in enumerate(generated_tokens):
        gen_only = output[input_lengths[i]:]
        output_str = tokenizer.decode(gen_only, skip_special_tokens=True)
        decoded_outputs.append(output_str)

    outputs = [trim_output(output) for output in decoded_outputs]

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "solution": example["gt"],
        "answer": example["answer"],
        "generated_answer": extract_last_number(extract_box(output)),
        "model_generation": output,
    } for example, output, prompt in tqdm(zip(test_data, outputs, prompts), desc="Creating predictions", total=len(test_data))]

    # Make output directory
    model_name_for_dir = MODEL_ID.split("/")[-1]
    output_dir = os.path.join(f"{model_name_for_dir}_SEAL")
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    output_file = os.path.join(output_dir, "predictions.json")
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {output_file}")
    print("Processing complete.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    

