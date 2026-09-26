"""Byte-level tokenizer: one token per UTF-8 byte (vocab = 256).

Real tokenizers (BPE) pack ~4 characters per token, so a 1,500-character
system prompt is ~375 tokens in GPT-4-class models and ~1,500 here. The cache
behaves identically; only the token count per sentence differs.
"""


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")
