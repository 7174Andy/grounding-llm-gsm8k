# steer_adaptive.py
import argparse
import re, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

def sigmoid(x): return 1 / (1 + torch.exp(-x))
def logit(p):   return torch.log(p/(1-p))

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

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans

def load_probe(path: str, return_v: bool = False):
    x = np.load(path, allow_pickle=True)

    # Case A: zipped dict (.npz)
    if isinstance(x, np.lib.npyio.NpzFile):
        w = np.asarray(x["w"])
        b = float(np.asarray(x["b"]))
        if return_v and "v" in x:
            return w, b, np.asarray(x["v"])
        return w, b

    # Case B: 0-D object array containing a dict (your current saver)
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        d = x.item()  # unwrap dict
        w = np.asarray(d["w"])
        b = float(d["b"])
        if return_v and "v" in d:
            return w, b, np.asarray(d["v"])
        return w, b

    # Case C: structured array with named fields
    if hasattr(x, "dtype") and getattr(x.dtype, "names", None) and {"w", "b"}.issubset(x.dtype.names):
        w = np.asarray(x["w"])
        b = float(np.asarray(x["b"]))
        if return_v and "v" in x.dtype.names:
            return w, b, np.asarray(x["v"])
        return w, b

    # Case D: flat vector [w..., b]
    if x.ndim == 1 and x.size >= 2:
        return x[:-1], float(x[-1]) if not return_v else (x[:-1], float(x[-1]), None)

    raise ValueError(f"Unrecognized probe format for {path!r}")

class AdaptiveCAVSteerer:
    """
    Implements e' = e + ε v  with  ε = (s0 - b - w·e) / ||w||
    using the indicator condition from the paper.
    By default, steers ONLY generated tokens (not the initial prompt pass).
    """
    def __init__(self, model, layer_to_wb, mode="amplify", p0=0.7, eps_cap=1.0, steer_prompt=False):
        """
        layer_to_wb: { L (1-based): {"w": np.ndarray[H], "b": float} }  in ORIGINAL space
        mode: "amplify" (ensure P_d >= p0) or "suppress" (ensure P_d <= p0)
        p0: target probability threshold
        eps_cap: clamp |ε| for stability
        """
        self.model = model
        self.layers = self._find_layers(model)
        self.mode = mode
        self.p0 = float(p0)
        self.eps_cap = float(eps_cap)
        self.steer_prompt = steer_prompt
        # keep on CPU; move per-call
        self.params = {L: {"w": torch.from_numpy(d["w"].astype(np.float32)),
                           "b": torch.tensor(float(d["b"]), dtype=torch.float32)}
                       for L, d in layer_to_wb.items()}
        self.hooks = []

    def _find_layers(self, model):
        for attr in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
            try:
                seq = eval(f"model.{attr}")
                if hasattr(seq, "__len__"): return list(seq)
            except Exception: pass
        raise RuntimeError("Could not locate decoder layers on this model.")

    def _make_hook(self, L):
        def hook(_module, _inputs, outputs):
            # outputs may be Tensor or tuple; first item is hidden states [B,T,H]
            if isinstance(outputs, tuple):
                hs, *rest = outputs
            else:
                hs, rest = outputs, None

            B, T, H = hs.shape
            # steer only generated tokens unless requested
            if (not self.steer_prompt) and T > 1:
                return outputs

            p = self.params[L]
            w = p["w"].to(hs.device, dtype=hs.dtype)   # [H]
            b = p["b"].to(hs.device, dtype=hs.dtype)   # scalar
            w_norm = torch.linalg.norm(w) + 1e-12
            v = w / w_norm                              # [H]

            # Current concept probability: σ(w·e + b)
            logits = torch.einsum("bth,h->bt", hs, w) + b  # [B,T]
            probs  = sigmoid(logits)

            s0 = logit(torch.tensor(self.p0, device=hs.device, dtype=hs.dtype))
            need = (probs < self.p0) if self.mode == "amplify" else (probs > self.p0)

            # ε = (s0 - b - w·e) / ||w||  (indicator-zeroed, clamped)
            eps = (s0 - b - torch.einsum("bth,h->bt", hs, w)) / w_norm
            eps = torch.where(need, eps, torch.zeros_like(eps))
            eps = torch.clamp(eps, min=-self.eps_cap, max=self.eps_cap)

            hs = hs + eps.unsqueeze(-1) * v.view(1,1,-1)
            return (hs, *rest) if rest is not None else hs
        return hook

    def enable(self, layers_1_based):
        for L in layers_1_based:
            idx = L - 1
            if not (0 <= idx < len(self.layers)):
                raise ValueError(f"Layer {L} out of range 1..{len(self.layers)}")
            self.hooks.append(self.layers[idx].register_forward_hook(self._make_hook(L)))

    def disable(self):
        for h in self.hooks:
            try: h.remove()
            except: pass
        self.hooks = []

def run_demo(model_id, probes_paths: dict):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        trust_remote_code=True, 
        cache_dir="/opt/huggingface_cache",
    ).eval()

    print("hidden_size:", model.config.hidden_size)  # expect 3584 for gemma-2-9b-it


    # Load (w,b) per layer
    layer_to_wb = {
        L: (lambda wb=load_probe(path): {"w": wb[0], "b": wb[1]})()
        for L, path in probes_paths.items()
    }

    print(f"Shape check of loaded CAVs:")
    for L, p in layer_to_wb.items():
        print(f"[Layer {L}] w.shape = {p['w'].shape}, b.shape = {p['b']}")

    # Choose amplify/suppress + target prob p0
    steerer = AdaptiveCAVSteerer(model, layer_to_wb, mode="amplify", p0=0.7, eps_cap=0.8, steer_prompt=False)
    steerer.enable(sorted(layer_to_wb.keys()))

    # Prepare test dataset
    test_data = []
    dataset = load_dataset("gsm8k", "main", split="test")
    data = dataset.shuffle(seed=42).select(range(132))
    for example in tqdm(data, desc="Processing examples"):
        answer = example["answer"].split("####")[1].strip()
        answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
        test_data.append({
            "question": example["question"],
            "solution": example["answer"].split("####")[0].strip(),
            "gt": answer,
        })

    prompts = []
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for example in tqdm(test_data, desc="Preparing Prompts..."):
        prompt = prefix + "Question: " + example["question"].strip()+"\n"

        # Apply chat template for Llama-3.1
        if "Llama" in model_id:
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
        else:
            messages = [{"role": "user", "content": prompt}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    for i, prompt in tqdm(enumerate(prompts), desc="Generating answers...", total=len(prompts)):
        enc = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(next(model.parameters()).device)
        out = model.generate(**enc, max_new_tokens=512, do_sample=False, use_cache=True)
        text = tok.decode(out[0], skip_special_tokens=True)

        test_data[i]["pred"] = text
        test_data[i]["generated_answer"] = extract_last_number(text)
        print(f"Example {i}: {test_data[i]['question']} {test_data[i]['gt']}")
        print(f"Prediction:\n{text}\n")

    # Save results
    with open("steered_gsm8k_results.json", "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    steerer.disable()
    print("Demo complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Model ID to use for generation")
    args = parser.parse_args()
    model_name_for_dir = args.model_id.split("/")[-1]
    run_demo(args.model_id, {10: f"{model_name_for_dir}_dataset/CAV/cav_layer_15.npy"})
