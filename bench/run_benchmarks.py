"""
Runs every experiment and writes results/results.json.

    python bench/run_benchmarks.py            # full run (~5 min on a laptop CPU)
    python bench/run_benchmarks.py --quick    # smaller sizes, ~1 min
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvlab import KVCache, TinyGPT, generate_cached, generate_no_cache  # noqa: E402
from kvlab.memory import MODELS, bytes_per_token, cache_gib  # noqa: E402
from kvlab.support_bot import run_scenario  # noqa: E402


def log(msg):
    print(msg, flush=True)


# 1. Correctness ---------------------------------------------------------------
def exp_correctness(model):
    """The cache must be an optimisation only: same logits, same tokens."""
    rng = np.random.default_rng(7)
    rows = []
    for plen in (16, 256, 1024):
        prompt = rng.integers(0, 256, plen).tolist()
        a = generate_no_cache(model, prompt, 32)
        b, _ = generate_cached(model, prompt, 32)
        rows.append(dict(prompt_tokens=plen, generated=32,
                         identical_tokens=a.tokens == b.tokens,
                         max_abs_logit_diff=float(np.abs(a.last_logits - b.last_logits).max())))

    # Stronger check: feed a sequence one token at a time through the cache and
    # compare the logits at EVERY position with one uncached pass.
    seq = rng.integers(0, 256, 400).tolist()
    full = model.forward(seq)
    cache = KVCache(model.cfg)
    inc = np.stack([model.forward([t], cache)[-1] for t in seq])
    stepwise = dict(positions=len(seq), max_abs_logit_diff=float(np.abs(full - inc).max()),
                    argmax_agreement=float((full.argmax(-1) == inc.argmax(-1)).mean()))
    return dict(generation=rows, stepwise=stepwise)


# 2. Per-token latency ---------------------------------------------------------
def exp_latency(model, prompt_len, n_new):
    prompt = np.random.default_rng(3).integers(0, 256, prompt_len).tolist()
    log(f"  no cache: {n_new} tokens after a {prompt_len}-token prompt ...")
    a = generate_no_cache(model, prompt, n_new)
    log("  kv cache ...")
    b, cache = generate_cached(model, prompt, n_new)
    return dict(
        prompt_tokens=prompt_len, generated=n_new,
        no_cache=dict(step_ms=[t * 1e3 for t in a.step_times_s], total_s=a.total_s,
                      tokens_processed=a.tokens_processed),
        kv_cache=dict(step_ms=[t * 1e3 for t in b.step_times_s], total_s=b.total_s,
                      tokens_processed=b.tokens_processed),
        identical_tokens=a.tokens == b.tokens,
        cache_mib_at_end=cache.nbytes_used() / 2 ** 20,
    )


# 3. The support-bot scenario ----------------------------------------------------
def exp_chat(model, reply_tokens, repeats):
    out = {}
    for strategy in ("none", "kv", "prefix"):
        reps = 1 if strategy == "none" else repeats   # "none" is slow and very stable
        log(f"  strategy={strategy} x{reps}")
        runs = [run_scenario(model, strategy, reply_tokens) for _ in range(reps)]
        turns = []
        for i in range(len(runs[0])):
            base = asdict(runs[0][i])
            base["ttft_s"] = statistics.median(r[i].ttft_s for r in runs)
            base["total_s"] = statistics.median(r[i].total_s for r in runs)
            turns.append(base)
        out[strategy] = turns
    return out


# 4. Memory for real models -----------------------------------------------------
def exp_memory():
    rows = []
    for s in MODELS:
        rows.append(dict(model=s.name, attention=s.attention, layers=s.layers,
                         kv_heads=s.kv_heads, head_dim=s.head_dim,
                         kib_per_token_fp16=bytes_per_token(s) / 1024,
                         gib_8k_context=cache_gib(s, 8192),
                         gib_32_users_x_8k=cache_gib(s, 8192, batch=32)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    model = TinyGPT()
    np.dot(np.ones((256, 256), np.float32), np.ones((256, 256), np.float32))  # warm BLAS
    model.forward([1] * 64)

    t0 = time.perf_counter()
    res = dict(
        model=dict(**model.cfg.__dict__, params=model.n_params()),
        machine=dict(platform=platform.platform(), python=platform.python_version(),
                     numpy=np.__version__, cpus=os.cpu_count()),
    )
    log("[1/4] correctness");  res["correctness"] = exp_correctness(model)
    log("[2/4] latency")
    res["latency"] = exp_latency(model, 256, 128 if args.quick else 512)
    log("[3/4] support-bot scenario")
    res["chat_reply_tokens"] = 12 if args.quick else 32
    res["chat"] = exp_chat(model, res["chat_reply_tokens"], 1 if args.quick else 3)
    log("[4/4] memory math");  res["memory"] = exp_memory()
    res["runtime_s"] = time.perf_counter() - t0

    out = ROOT / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    log(f"wrote {out.relative_to(ROOT)} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
