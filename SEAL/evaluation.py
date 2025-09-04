import os
import json
from tqdm import tqdm
from bert_score import score as bert_score
import evaluate

MODEL_ID = "Qwen/Qwen2-7B-Instruct"

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

def evaluate_bert(predictions):
    """
    Evaluate BERTScore between generated answers and solutions.
    """
    candidates = [pred["model_generation"] for pred in predictions]
    references = [pred["answer"] for pred in predictions]

    P, R, F1 = bert_score(candidates, references, lang="en", rescale_with_baseline=True)
    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item()
    }

def evaluate_rouge(predictions):
    """
    Evaluate ROUGE-L between generated answers and solutions.
    """
    rouge = evaluate.load("rouge")
    candidates = [pred["model_generation"] for pred in predictions]
    references = [pred["answer"] for pred in predictions]
    results = rouge.compute(predictions=candidates, references=references, use_stemmer=True)
    return {
        "rougeL": results["rougeL"]
    }

def evaluate_bleu(predictions):
    """
    Evaluate BLEU score between generated answers and solutions.
    """
    bleu = evaluate.load("bleu")
    candidates = [pred["model_generation"] for pred in predictions]
    references = [[pred["answer"]] for pred in predictions]  # BLEU expects list of lists
    results = bleu.compute(predictions=candidates, references=references)
    return {
        "bleu": results["bleu"]
    }

def main():
    model_id_string = MODEL_ID.split("/")[-1]
    pred_path = os.path.join(f"{model_id_string}_SEAL", "predictions.json")
    predictions = load_predictions(pred_path)
    results = evaluate_predictions(predictions)
    print("Accuracy Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    bert_results = evaluate_bert(predictions)
    print("BERTScore Results:")
    for key, value in bert_results.items():
        print(f"  {key}: {value}")
    rouge_results = evaluate_rouge(predictions)
    print("ROUGE Results:")
    for key, value in rouge_results.items():
        print(f"  {key}: {value}")
    bleu_results = evaluate_bleu(predictions)
    print("BLEU Results:")
    for key, value in bleu_results.items():
        print(f"  {key}: {value}")
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
