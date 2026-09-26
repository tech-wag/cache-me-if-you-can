"""
Prefix caching: reuse KV across *requests*, not just across tokens.

A plain KV cache lives for one request and is thrown away. But in a real
chatbot most of every request is text the server has already seen:

    [ system prompt (same for every user) ][ turn 1 ][ turn 2 ] ... [ new message ]

Because attention is causal, the K/V for any prefix are identical no matter
what comes after it. So if the server keeps finished caches around, a new
request only has to prefill the part after the longest prefix it has seen.
This is the idea behind vLLM's automatic prefix caching, SGLang's RadixAttention
and the prompt-caching features of hosted LLM APIs. This file is the smallest
honest version of it: longest-common-prefix lookup over an LRU of snapshots.
"""
from __future__ import annotations

from collections import OrderedDict

from .model import KVCache


def _common_prefix_len(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    def __init__(self, max_entries: int = 16):
        self.max_entries = max_entries
        self._entries: OrderedDict[int, tuple[tuple[int, ...], KVCache]] = OrderedDict()
        self._next_id = 0
        self.hits = 0
        self.misses = 0

    def lookup(self, tokens: list[int]) -> tuple[KVCache | None, int]:
        """Return (a private copy of the best matching cache, reused length)."""
        best_id, best_len = None, 0
        for eid, (toks, _) in self._entries.items():
            n = _common_prefix_len(toks, tokens)
            if n > best_len:
                best_id, best_len = eid, n
        # Always leave at least one token to prefill: we need its logits.
        best_len = min(best_len, len(tokens) - 1)
        if best_id is None or best_len <= 0:
            self.misses += 1
            return None, 0
        self.hits += 1
        self._entries.move_to_end(best_id)             # LRU touch
        return self._entries[best_id][1].copy(upto=best_len), best_len

    def insert(self, tokens: list[int], cache: KVCache) -> None:
        """Store a finished cache. `tokens[:cache.length]` are the tokens it holds."""
        key = tuple(tokens[: cache.length])
        self._entries[self._next_id] = (key, cache)
        self._next_id += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)          # evict least recently used

    def __len__(self) -> int:
        return len(self._entries)
