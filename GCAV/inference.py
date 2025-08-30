# steer_adaptive.py
import re, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def sigmoid(x): return 1 / (1 + torch.exp(-x))
def logit(p):   return torch.log(p/(1-p))

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

def run_demo(model_id, probes_npz: dict):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        trust_remote_code=True
    ).eval()

    # Load (w,b) per layer
    layer_to_wb = {L: {"w": np.load(path)["w"], "b": float(np.load(path)["b"])} for L, path in probes_npz.items()}

    # Choose amplify/suppress + target prob p0
    steerer = AdaptiveCAVSteerer(model, layer_to_wb, mode="amplify", p0=0.7, eps_cap=0.8, steer_prompt=False)
    steerer.enable(sorted(layer_to_wb.keys()))

    # Prepare test dataset
    test_data = []
    dataset = load_dataset("gsm8k", "main", split="test")
    data = data.shuffle(seed=42).select(range(132))
    for example in tqdm(data, desc="Processing examples"):
        answer = example["answer"].split("####")[1].strip()
        answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
        test_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[0].strip(),
            "gt": answer,
        })

    prompts = []
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in tqdm(enumerate(test_data), desc="Preparing Prompts..."):
        prompt = prefix + "Question: " + example["question"].strip()+"\nAnswer: "

        # Apply chat template for Llama-3.1
        if "Llama" in model_id:
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
        else:
            messages = [{"role": "user", "content":prompt}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=512, do_sample=False, use_cache=True)
    text = tok.decode(out[0], skip_special_tokens=True)

    steerer.disable()
    print(text)

if __name__ == "__main__":
    # Example: steer at layer 10 using probes/layer_10.npz
    run_demo("google/gemma-2-9b-it", {10: "probes/layer_10.npz"})
