import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
from datasets import load_dataset
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

RANDOM_SEED = 44
MODEL_ID = "google/gemma-2-9b-it"

def prepare_dataset(gsm8k_dataset, controller):
    sampled_dataset = gsm8k_dataset.shuffle(seed=RANDOM_SEED).select(range(3000))

    # Binary classification: 0 = original question, 1 = CoT-augmented
    training_dataset = [
        {'prompt': controller.format_prompt(ex['question']), 'label': 0} for ex in sampled_dataset
    ]
    training_dataset += [
        {'prompt': controller.format_prompt(ex['question']) + "Please think through your reasoning step by step before answering.", 'label': 1}
        for ex in sampled_dataset
    ]

    train_inputs = [ex['prompt'] for ex in training_dataset]
    train_labels = np.array([ex['label'] for ex in training_dataset], dtype=np.int64)

    return train_inputs, train_labels


def setup_model():
    """
    Load the model and tokenizer with appropriate settings.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

    print(f"Loading model {MODEL_ID} with dtype {torch_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir="/opt/huggingface_cache",
            trust_remote_code=True,
        )
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

if __name__ == "__main__":
    dataset = load_dataset('gsm8k', 'main', split='train')

    # Set up the model and tokenizer
    tokenizer, model, device = setup_model()

    # Initialize the neural controller
    controller = NeuralController(
        model=model,
        tokenizer=tokenizer,
        n_components=3,
        control_method='rfm',
    )

    # Prepare the dataset
    dataset_inputs, dataset_labels = prepare_dataset(dataset, controller)
    print(f"train_labels shape: {len(dataset_inputs)}, unique values: {np.unique(dataset_labels)}")

    # Compute concept vectors
    controller.compute_directions(dataset_inputs, dataset_labels)

    # Save controller state
    filename = './rfm_gsm8k_meta-llama/Llama-3.1-8B-Instruct.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    controller.save(concept="gsm8k", model_name=MODEL_ID)

    print(f"Controller state saved to {filename}")
    print("Training complete.")
