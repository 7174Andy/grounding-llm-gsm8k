import os
from tqdm import tqdm
import re

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

gsm8k_test = load_dataset("gsm8k", "main", split="test")
MODEL_ID = "google/gemma-2-9b-it"
BATCH_SIZE = 8

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    device_map="auto",
    cache_dir="/opt/huggingface_cache"
)

# Helper to extract the answer number from the model's text
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
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return None

def generate_answer(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

print("Starting evaluation...")
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
    responses = generate_answer(batch_prompts)

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
print(f"Accuracy: {results['accuracy']:.4f}")