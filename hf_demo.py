"""
Optional: the same experiment on a real pretrained model (GPT-2, 124M) with
Hugging Face transformers, so you can see real text come out.

    pip install torch transformers
    python hf_demo.py

Downloads GPT-2 (~500 MB) on first run. Not needed for the main benchmarks,
which run on NumPy alone. (This script was not executed in the environment
that produced the results in results/, which had no access to model weights.)
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvlab.support_bot import SYSTEM_PROMPT

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = SYSTEM_PROMPT + "\nCustomer: I lost my card at a restaurant last night.\nPeach:"
ids = tok(prompt, return_tensors="pt").input_ids
print(f"prompt: {ids.shape[1]} GPT-2 tokens")

for use_cache in (False, True):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, do_sample=False,
                             use_cache=use_cache, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\nuse_cache={use_cache}: {dt:.2f}s")
    print(text.strip()[:300])
