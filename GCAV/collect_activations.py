import os
from tqdm import tqdm
import re
from pathlib import Path
import random
import json
import argparse

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from vllm import LLM, SamplingParams

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # Adjust based on your available GPUs'

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

def pool_last_prompt_token(hidden_states, attention_mask, layer_idx: int):
    """
    hidden_states: tuple(len = num_layers+1) of [B,T,H] (0 is embeddings)
    attention_mask: [B,T] with 1=token, 0=pad
    layer_idx: 1..num_layers
    returns [B,H] for the final non-pad input token
    """
    hs = hidden_states[layer_idx]           # [B,T,H]
    lengths = attention_mask.sum(dim=1)    # [B]
    idx = (lengths - 1).clamp(min=0)       # [B]
    return hs[torch.arange(hs.size(0), device=hs.device), idx]  # [B,H]

def main():
    parser = argparse.ArgumentParser(description="GCAV Generation Script")
    parser.add_argument("--model_id", type=str, default="", help="Model ID to use for generation")
    args = parser.parse_args()

    random.seed(42)

    # Prepare dataset
    test_data = []
    data = load_dataset("gsm8k", "main", split="test")
    data = data.shuffle(seed=42).select(range(10))  # Use a subset for testing
    for example in tqdm(data, desc="Processing examples"):
        answer = example["answer"].split("####")[1].strip()
        answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
        test_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[0].strip(),
            "gt": answer,
        })
    
    # Setup Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + "Question: " + example["question"].strip()+"\nAnswer: "

        # Apply chat template for Llama-3.1
        if "Llama" in args.model_id:
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
        else:
            messages = [{"role": "user", "content": example["question"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # Generate answers
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

    model = LLM(args.model_id, dtype=torch_dtype, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), max_model_len=512+2000, download_dir="/opt/huggingface_cache", gpu_memory_utilization=0.75)
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
        "label": 1 if example["gt"] == extract_last_number(extract_box(output)) else 0
    } for example, output, prompt in tqdm(zip(test_data, decoded_outputs, prompts), desc="Creating predictions", total=len(test_data))]

    # Save predictions to output directory
    model_name_for_dir = args.model_id.split("/")[-1]
    output_dir = os.path.join(f"{model_name_for_dir}_dataset")
    os.makedirs(output_dir, exist_ok=True)

    # Record predictions to the desired output directory
    with open(os.path.join(output_dir, "baseline_predictions.json"), "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
    
    enc_all = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    del model

    hf_device = "cpu"  # safest default alongside vLLM
    hf_dtype = torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=hf_dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/opt/huggingface_cache"
    ).eval()

    with torch.inference_mode():
        dummy = tokenizer("hi", return_tensors="pt").to(hf_device)
        o = hf_model(**dummy, output_hidden_states=True)
    num_layers = len(o.hidden_states) - 1

    N = len(prompts)
    # Hidden size from layer 1
    H = o.hidden_states[1].shape[-1]
    # Preallocate per-layer feature matrices
    layer_arrays = [np.zeros((N, H), dtype=np.float32) for _ in range(num_layers)]

    bs = 8
    with torch.inference_mode():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            in_batch = {
                "input_ids": enc_all["input_ids"][start:end].to(hf_device),
                "attention_mask": enc_all["attention_mask"][start:end].to(hf_device),
            }
            out = hf_model(**in_batch, output_hidden_states=True)
            for li in range(1, num_layers + 1):
                pooled = pool_last_prompt_token(out.hidden_states, in_batch["attention_mask"], li)  # [B,H]
                layer_arrays[li - 1][start:end, :] = pooled.cpu().numpy()

    # ADDED: save per-layer arrays
    for li in range(1, num_layers + 1):
        np.save(os.path.join(output_dir, f"layer_{li:02d}.npy"), layer_arrays[li - 1])
    # ADDED: tiny meta file
    with open(os.path.join(output_dir, "activations_meta.json"), "w") as f:
        json.dump({
            "pooling": "last_prompt_token",
            "num_examples": N,
            "num_layers": num_layers,
            "hidden_size": H,
        }, f, indent=2)

    print(f"Saved per-layer activations to: {output_dir}")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()