import json
import os
import re

from decimal import Decimal, InvalidOperation


MODEL_ID = "google/gemma-2-9b-it"

NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")

def extract_final_number(text: str):
    """Return the most likely final numeric answer as a string."""
    # remove thousands separators like 1,234 -> 1234
    s = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Prefer common GSM8K-style markers if present
    m = re.search(r"####\s*([-+]?\d*\.\d+|\d+)", s)
    if m:
        return m.group(1)

    m = re.search(r"(final answer|answer)\s*[:\-]?\s*([-+]?\d*\.\d+|\d+)", s, flags=re.I)
    if m:
        return m.group(2)

    # Fallback: last number in the string
    nums = NUM_RE.findall(s)
    return nums[-1] if nums else None

def to_decimal(num_str: str):
    if num_str is None:
        return None
    try:
        return Decimal(num_str)
    except InvalidOperation:
        return None

def load_predictions(file_path):
    """
    Load predictions from a JSON file.
    """
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    return predictions

def evaluate_predictions(predictions):
    """
    Evaluate the model's predictions against the ground truth.
    """
    correct = 0
    total = 0
    for pred in predictions:
        if to_decimal(extract_final_number(pred["generated_answer"])) == to_decimal(extract_final_number(pred["gt"])):
            correct += 1
        else:
            print(f"Incorrect Prediction:\nQuestion: {pred['question']}\nPredicted: {pred['generated_answer']}\nGround Truth: {pred['gt']}\n")
        total += 1
    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }

def main():
    pred_path = os.path.join("steered_gsm8k_results_qwen_layer10.json")
    print(f"Loading predictions from {pred_path}")
    predictions = load_predictions(pred_path)
    eval_results = evaluate_predictions(predictions)
    print("Accuracy Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()