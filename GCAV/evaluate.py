import json
import os
import re

MODEL_ID = "google/gemma-2-9b-it"

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


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
        if extract_last_number(pred["generated_answer"]) == extract_last_number(pred["gt"]):
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
    pred_path = os.path.join("steered_gsm8k_results.json")
    print(f"Loading predictions from {pred_path}")
    predictions = load_predictions(pred_path)
    eval_results = evaluate_predictions(predictions)
    print("Accuracy Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()