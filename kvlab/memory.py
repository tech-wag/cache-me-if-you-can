"""KV cache memory math for real, published model shapes.

bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value

Grouped-query attention (GQA) shares each K/V head across several query heads,
which is why Llama 3 8B needs a quarter of the cache that Llama 2 7B does
despite being a larger model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    attention: str  # "MHA" or "GQA"


MODELS = [
    Shape("GPT-2 small (124M)", 12, 12, 64, "MHA"),
    Shape("Llama 2 7B", 32, 32, 128, "MHA"),
    Shape("Llama 3 8B", 32, 8, 128, "GQA"),
    Shape("Mistral 7B", 32, 8, 128, "GQA"),
    Shape("Llama 3 70B", 80, 8, 128, "GQA"),
]


def bytes_per_token(s: Shape, bytes_per_value: int = 2) -> int:
    return 2 * s.layers * s.kv_heads * s.head_dim * bytes_per_value


def cache_gib(s: Shape, context_tokens: int, batch: int = 1, bytes_per_value: int = 2) -> float:
    return bytes_per_token(s, bytes_per_value) * context_tokens * batch / 2 ** 30
