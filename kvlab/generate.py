"""Two ways to generate text with the same model: with and without a KV cache."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import KVCache, TinyGPT


@dataclass
class GenResult:
    tokens: list[int]                      # generated token ids only
    ttft_s: float                          # time to first token
    step_times_s: list[float] = field(default_factory=list)  # one per generated token
    tokens_processed: int = 0              # how many token-positions went through the network
    last_logits: np.ndarray | None = None

    @property
    def total_s(self) -> float:
        return sum(self.step_times_s)


def generate_no_cache(model: TinyGPT, prompt: list[int], n_new: int) -> GenResult:
    """The naive loop: every new token re-runs the ENTIRE sequence."""
    seq = list(prompt)
    out, times, processed = [], [], 0
    logits = None
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits = model.forward(seq)[-1]    # recompute K/V for all len(seq) positions
        tok = int(np.argmax(logits))
        times.append(time.perf_counter() - t0)
        processed += len(seq)
        seq.append(tok)
        out.append(tok)
    return GenResult(out, times[0], times, processed, logits)


def generate_cached(model: TinyGPT, prompt: list[int], n_new: int,
                    cache: KVCache | None = None) -> tuple[GenResult, KVCache]:
    """Prefill once, then feed ONE token per step and read the rest from the cache.

    If `cache` is passed in already holding some prefix of `prompt`
    (see prefix_cache.py), only the part of the prompt after that prefix is
    prefilled.
    """
    cache = cache if cache is not None else KVCache(model.cfg)
    already = cache.length
    todo = prompt[already:]
    out, times = [], []

    t0 = time.perf_counter()
    logits = model.forward(todo, cache)[-1]            # prefill: the prompt's new part
    tok = int(np.argmax(logits))
    times.append(time.perf_counter() - t0)
    processed = len(todo)
    out.append(tok)

    for _ in range(n_new - 1):                          # decode: one token at a time
        t0 = time.perf_counter()
        logits = model.forward([tok], cache)[-1]
        tok = int(np.argmax(logits))
        times.append(time.perf_counter() - t0)
        processed += 1
        out.append(tok)

    # Note: the last generated token has not been written to the cache yet
    # (it hasn't been fed back in). Callers that continue the conversation
    # include it in the next prompt, and it gets prefilled then.
    return GenResult(out, times[0], times, processed, logits), cache
