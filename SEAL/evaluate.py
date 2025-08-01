import os
import json
from tqdm import tqdm

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
        if pred["generated_answer"] == pred["solution"]:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }

def main():
    pred_path = os.path.join("Llama-3.1-8B-Instruct_SEAL", "predictions.json")
    predictions = load_predictions(pred_path)
    results = evaluate_predictions(predictions)
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
