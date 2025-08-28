import re
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import evaluate
from pathlib import Path
import random

from rfm_train import setup_model

# Word Count dataset is from https://github.com/brianpeiris/llm-basic-letter-counting-benchmark/blob/main/words.json

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

RANDOM_SEED = 50
MODEL_ID = "google/gemma-2-9b-it"
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def process_json_file(file_path):
    """
    Process a JSON file to extract and return its content.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The content of the JSON file.
    """
    import json
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

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

def extract_reference_answer(completion):
    """ Extract the numeric answer from the model's completion."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return None

def evaluate_accuracy(gsm8k_test, controller, processor=None):
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
        prompts.append(f"Question: {question} and put the answer in a box.\nAnswer:")
        gold_answers.append(answer)

    for i in tqdm(range(0, total), desc="Evaluating Accuracy for RFM..."):
        prompt = controller.format_prompt(prompts[i])
        batch_answers = gold_answers[i]

        # Generate answers for the batch
        control_layers = list(range(-1, -27, -1)) # All layers
        if processor is not None:
            response = controller.generate(prompt, max_new_tokens=256, control_coef=0.2, layers_to_control=control_layers, do_sample=False, logits_processor=[processor])
        else:
            response = controller.generate(prompt, max_new_tokens=256, control_coef=0.2, layers_to_control=control_layers, do_sample=False)
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

def evaluate_all(gsm8k_test, controller, processor=None):
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
        control_layers = list(range(-1, -31, -1))
        if processor is not None:
            response = controller.generate(prompt, max_new_tokens=450, control_coef=0.2, layers_to_control=control_layers, do_sample=False, logits_processor=[processor])
        else:
            response = controller.generate(prompt, max_new_tokens=450, control_coef=0.2, layers_to_control=control_layers, do_sample=False)
        print(response)

        pred = response.strip()
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

def eval_count_words(data_dir, controller, processor=None):
    """
    Evaluate the word count of the model's responses on the GSM8K dataset.
    """
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    json_file = directory / "words.json"

    dataset = process_json_file(json_file)

    print(f"Loaded {len(dataset)} examples from {json_file}")

    # use the first 887 examples
    dataset = random.sample(dataset, 887)

    # Generate prompts
    prompts = [f"How many times does the letter '{ex['letter']}' appear in '{ex['word']}'? " for ex in dataset]
    assert len(prompts) == len(dataset), "Mismatch between prompts and dataset length."
    prefix = "Put your final answer within \\boxed{}.\n"

    # Generate answers using the controller
    predictions = []
    control_layers = list(range(-1, -31, -1))
    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating word count...")):
        prompt = controller.format_prompt(prompt + prefix, steer=True)
        if processor is not None:
            response = controller.generate(prompt, max_new_tokens=256, do_sample=False, control_coef=0.2, layers_to_control=control_layers, logits_processor=[processor])
        else:
            response = controller.generate(prompt, max_new_tokens=256, do_sample=False, control_coef=0.2, layers_to_control=control_layers)
        print(f"Response for example {i + 1}: {response}")
        response = response.strip()
        if response:
            try:
                answer = extract_box(response)
                if answer == "":
                    answer = extract_final_number(response)
                print(f"Extracted answer for example {i + 1}: {answer}")
                count = int(answer) if answer is not None else None
                predictions.append(count)
            except ValueError:
                print(f"Warning: Invalid response for example {i + 1}: {response}")
                predictions.append(None)

    # Evaluate the word count accuracy
    correct_count = 0
    for i, ex in enumerate(dataset):
        if predictions[i] is not None and predictions[i] == ex['count']:
            correct_count += 1
        else:
            print(f"Example {i + 1} failed: Predicted {predictions[i]}, Expected {ex['count']}")

    accuracy = correct_count / len(dataset) if dataset else 0
    print(f"Word Count Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Load the GSM8K dataset
    dataset = load_dataset('gsm8k', 'main', split='test')
    gsm8k_test_sample = dataset.shuffle(seed=RANDOM_SEED).select(range(132))

    # Set up the model and tokenizer
    tokenizer, model, device = setup_model()

    eos_token_id = tokenizer.eos_token_id

    class SuppressEosLogitsProcessor(torch.nn.Module):
        def __call__(self, input_ids, scores):
            scores[:, eos_token_id] = -1e9
            return scores

    # Load the controller
    controller = NeuralController(
        model=model,
        tokenizer=tokenizer,
        batch_size=4,
        control_method='rfm',
        rfm_iters=10,
        n_components=4,
    )

    # Load the controller state
    controller.load(concept="gsm8k", model_name=MODEL_ID)

    # Prepare the dataset
    dataset_inputs = [ex['question'] for ex in dataset]
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(gsm8k_test_sample, controller, SuppressEosLogitsProcessor())
    print(f"RFM Accuracy on GSM8K: {accuracy:.4f}")
