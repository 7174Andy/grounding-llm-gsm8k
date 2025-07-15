import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import evaluate

from rfm_train import setup_model

RANDOM_SEED = 42
MODEL_ID = "google/gemma-2-9b-it"
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

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

def extract_reference_answer(completion):
    """ Extract the numeric answer from the model's completion."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return None

def evaluate_accuracy(gsm8k_test, controller):
    """
    Evaluate the accuracy of the model on the GSM8K dataset.
    """

    predictions = []
    references = []

    total = len(gsm8k_test)
    prompts = []
    gold_answers = []

    # Prepare prompts and gold answers
    for ex in gsm8k_test:
        question = ex['question']
        answer = extract_reference_answer(ex['answer'])
        prompts.append(f"Question: {question}  At the end, give your final numeric answer in the format '#### answer'.\nAnswer:")
        gold_answers.append(answer)

    for i in tqdm(range(0, total), desc="Evaluating Accuracy for RFM..."):
        prompt = controller.format_prompt(prompts[i])
        batch_answers = gold_answers[i]

        # Generate answers for the batch 
        response = controller.generate(prompt, max_new_tokens=256, control_coef=0.2, layers_to_control=list(range(-1, -31, -1)))
        print(response)

        pred = extract_final_number(response)
        print(f"Prediction for example {i + 1}: {pred}")
        if pred is not None:
            predictions.append(pred)
            references.append(batch_answers)
        else:
            print(f"Warning: No valid prediction for example {i + 1}")

    # Filter out None values
    filtered = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
    if not filtered:
        print("No valid prediction-reference pairs found.")
        exit()

    predictions, references = zip(*filtered)

    # Compute accuracy
    correct = sum(p == r for p, r in zip(predictions, references))
    accuracy = correct / len(predictions)
    return accuracy

def evaluate_all(gsm8k_test, controller):
    """
    Evaluate the model on the GSM8K dataset using the controller.
    """
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bert_metric = evaluate.load("bertscore")

    predictions = []
    references = []

    total = len(gsm8k_test)
    prompts = []
    gold_answers = []

    # Prepare prompts and gold answers
    for ex in gsm8k_test:
        question = ex['question']
        answer = ex['answer']
        prompts.append(f"Question: {question} Answer the question step by step.\nAnswer:")
        gold_answers.append(answer)
    
    for i in tqdm(range(0, total), desc="Evaluating RFM..."):
        prompt = controller.format_prompt(prompts[i])
        batch_answers = gold_answers[i]

        # Generate answers for the batch 
        response = controller.generate(prompt, max_new_tokens=256, control_coef=0.2, layers_to_control=list(range(-1, -31, -1)))
        print(response)

        pred = response.strip()
        print(f"Prediction for example {i + 1}: {pred}")
        if pred:
            predictions.append(pred)
            references.append(batch_answers)
        else:
            print(f"Warning: No valid prediction for example {i + 1}")
        
    # Filter out None values
    filtered = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
    if not filtered:
        print("No valid prediction-reference pairs found.")
        exit()

    predictions, references = zip(*filtered)

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    print(f"RFM BLEU Score: {bleu_score['bleu']:.4f}")

    # Compute ROUGE score
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    print(f"RFM ROUGE Score: {rouge_score['rouge1']:.4f}")

    # Compute BERTScore
    bert_score = bert_metric.compute(predictions=predictions, references=references, lang="en")
    print(f"RFM BERTScore: {np.mean(bert_score['f1']):.4f}")

if __name__ == "__main__":
    # Load the GSM8K dataset
    dataset = load_dataset('gsm8k', 'main', split='test')
    gsm8k_test_sample = dataset.shuffle(seed=RANDOM_SEED).select(range(132))

    # Set up the model and tokenizer
    tokenizer, model, device = setup_model()

    # Load the controller
    controller = NeuralController(
        model=model,
        tokenizer=tokenizer,
        n_components=3,
        control_method='rfm',
    )

    # Load the controller state
    filename = './rfm_gsm8k_meta-llama/Llama-3.1-8B-Instruct.pkl'
    controller.load(concept="gsm8k", model_name=MODEL_ID)

    print(f"Controller loaded from {filename}")

    # Prepare the dataset
    dataset_inputs = [ex['question'] for ex in dataset]
    layers_to_control = list(range(-1, -31, -1))
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(gsm8k_test_sample, controller)
    print(f"RFM Accuracy on GSM8K: {accuracy:.4f}")

    # Evaluate all metrics
    evaluate_all(gsm8k_test_sample, controller)
