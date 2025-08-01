import json
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

def generate_index(text, tokenizer, split_id, think_only=True):
    check_words=["verify","make sure","hold on","think again","'s correct","'s incorrect","let me check","seems right"]
    check_prefix = ["wait"]
    switch_words=["think differently","another way","another approach","another method","another solution","another strategy","another technique"]
    switch_prefix=["alternatively"]
    
    if think_only:
        # Use string search instead of token IDs
        start_pos = text.lower().find("<think>")
        end_pos = text.lower().find("</think>")
        if start_pos == -1:
            return [], [], []
        if end_pos == -1:
            end_pos = len(text)
        segment = text[start_pos + len("<think>"):end_pos]
        think_tokens = tokenizer.encode(segment)
        start = 0
    else:
        think_tokens = tokenizer.encode(text)
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index, check_index, switch_index = [], [], []
    for i in range(len(index)-1):
        step_index.append(index[i]+start)
        step = think_tokens[index[i]+1:index[i+1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        if any(step.lower().startswith(p) for p in check_prefix) or any(w in step.lower() for w in check_words):
            check_index.append(i)
        elif any(step.lower().startswith(p) for p in switch_prefix) or any(w in step.lower() for w in switch_words):
            switch_index.append(i)
    return step_index, check_index, switch_index

def generate(data, save_dir):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir="/opt/huggingface_cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"]+d["model_generation"] for d in data]

    layer_num = model.config.num_hidden_layers+1
    hidden_dict=[{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts), desc="Generating hidden states"):
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model(**tokenized_batch, output_hidden_states=True)
            hidden_states = output.hidden_states
            hidden_states = [h.detach().cpu() for h in hidden_states]
        layer_num = len(hidden_states)
        step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=False)
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)
        for i in range(layer_num):
            h = hidden_states[i][0]
            step_h = h[step_index]
            hidden_dict[i][k] = {"step":step_h, "check_index": check_index, "switch_index": switch_index}
        del hidden_states
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    json.dump(prompts, open(f"{save_dir}/prompts.json", "w"))

def extract_data(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip blank lines
                data.append(json.loads(line))
    return data
def main():
    input_file = "./Llama-3.1-8B-Instruct_SEAL/baseline_predictions.json"
    save_dir = "./hidden_analysis"
    
    print("Extracting data from input file...")
    data = extract_data(input_file)
    
    print("Generating hidden states...")
    generate(data, save_dir)
    
    print(f"Hidden states saved to {save_dir}/hidden.pt and prompts saved to {save_dir}/prompts.json")

if __name__ == "__main__":
    main()