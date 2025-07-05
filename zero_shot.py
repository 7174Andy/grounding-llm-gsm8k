import os
from tqdm import tqdm
import re
import argparse
from pathlib import Path

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# Configuration
MODEL_ID = "google/gemma-2-9b-it"
BATCH_SIZE = 8
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation script for GSM8K dataset.")

    parser.add_argument('--metric', type=str, choices=['accuracy', 'bleu'], required=False, 
                        help="Evaluation metric to use. Default is 'accuracy'.")
    
    parser.add_argument('--sample', type=bool, default=False,
                        help="If True, generates samples from the model. Default is False.")

    return parser

def setup_model():
    """
    Load the model and tokenizer with appropriate settings.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

    print(f"Loading model {MODEL_ID} with dtype {torch_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir="/opt/huggingface_cache"
        )
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def extract_final_number(text):
    """
    Extract the last numeric answer from a string (can include $, commas, etc.)
    """
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if matches:
        try:
            return int(float(matches[-1]))  # Convert to int if possible
        except ValueError:
            return None
    return None

def extract_answer_hf(completion):
    """ Extract the numeric answer from the model's completion."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return None

# Function to generate answers using the model in a batch
def generate_answer(model, tokenizer, prompts):
    """ Generate answers for a batch of prompts using the model. """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def generate_samples(model, tokenizer, device, gsm8k_test, len=10):
    """
    Generate samples from the model for the GSM8K dataset.
    """

    # Prepare prompts and gold answers
    with open("gsm8k_zero_shot_samples.txt", "w") as f:
        for idx in tqdm(range(len)):
            question = gsm8k_test[idx]['question']
            inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            f.write("----------------------\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n\n")
            f.write("----------------------\n")

def evaluate_accuracy(model, tokenizer, device, gsm8k_test):
    """
    Evaluate the accuracy of the model on the GSM8K dataset.
    """
    accuracy_metric = evaluate.load("accuracy")

    predictions = []
    references = []

    total = len(gsm8k_test)
    prompts = []
    gold_answers = []

    # Prepare prompts and gold answers
    for ex in gsm8k_test:
        question = ex['question']
        answer = extract_answer_hf(ex['answer'])
        prompts.append(f"Question: {question}Answer the question step by step. At the end, give your final numeric answer in the format '#### answer'.\nAnswer:")
        gold_answers.append(answer)

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Evaluating Accuracy"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_answers = gold_answers[i:i + BATCH_SIZE]

        # Generate answers for the batch
        responses = generate_answer(model, tokenizer, batch_prompts)

        for response, reference in zip(responses, batch_answers):
            pred = extract_final_number(response)
            predictions.append(pred if pred is not None else "")
            references.append(reference)

    # Filter out None values
    filtered = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
    if not filtered:
        print("No valid prediction-reference pairs found.")
        exit()

    predictions, references = zip(*filtered)

    # Compute accuracy
    results = accuracy_metric.compute(predictions=predictions, references=references)
    return results['accuracy']

def evaluate_bleu(model, tokenizer, device, gsm8k_test):
    """
    Evaluate the BLEU score of the model on the GSM8K dataset.
    """
    bleu_metric = evaluate.load("bleu")

    predictions = []
    references = []

    total = len(gsm8k_test)
    prompts = []
    gold_answers = []

    # Prepare prompts and gold answers
    for ex in gsm8k_test:
        question = ex['question']
        answer = extract_answer_hf(ex['answer'])
        prompts.append(f"Question: {question}\nAnswer:")
        gold_answers.append(answer)

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Evaluating BLEU"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_answers = gold_answers[i:i + BATCH_SIZE]

        # Generate answers for the batch
        responses = generate_answer(model, tokenizer, batch_prompts)

        for response, reference in zip(responses, batch_answers):
            pred = response
            if pred is not None:
                predictions.append(str(pred))
                references.append(str(reference))

    # Compute BLEU score
    results = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return results['bleu']

def evaluate_rouge(gsm8k_test):
    """
    Evaluate the Rouge score of the model on the GSM8K dataset.
    """
    rouge_metric = evaluate.load("rouge")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.metric not in ['accuracy', 'bleu'] and not args.sample:
        print(f"Invalid metric: {args.metric}. Choose either 'accuracy' or 'bleu'.")
        exit()

    # Load the model and tokenizer
    print("Setting up the model...")
    tokenizer, model, device = setup_model()
    print("Model setup complete.")
    gsm8k_test = load_dataset("gsm8k", "main", split="test")

    # Generate samples if requested
    if args.sample:
        print("Generating samples...")
        generate_samples(model, tokenizer, device, gsm8k_test)
        print("Sample generation completed.")
        exit()

    # Evaluate the model's accuracy
    # Change to whatever evaluation metric you want to use
    print(f"Starting evaluation using metric: {args.metric}")
    if args.metric == 'accuracy':
        results = evaluate_accuracy(model, tokenizer, device, gsm8k_test)
        print("Accuracy evaluation completed.")
        print(f"Accuracy: {results:.4f}")
    else:
        results = evaluate_bleu(model, tokenizer, device, gsm8k_test)
        print("BLEU evaluation completed.")
        print(f"BLEU score: {results:.4f}")