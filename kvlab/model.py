"""
TinyGPT: a decoder-only transformer written in plain NumPy, with an optional KV cache.

Nothing here is hidden behind a framework. Every matrix multiply that an LLM
inference server does for one generated token is in this file, so you can see
exactly which work the KV cache removes.

The weights are random. That is deliberate: this project measures the
*inference engine* (how much work each token costs), not what the model knows.
The math per token is identical to a trained GPT-2-style model of the same shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256      # byte-level tokenizer: one token per UTF-8 byte
    max_seq: int = 4096        # longest context the model (and cache) can hold
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# --------------------------------------------------------------------------- #
# The KV cache                                                                #
# --------------------------------------------------------------------------- #
class KVCache:
    """Keys and values for every layer, every head, every position seen so far.

    Shape per tensor: (n_layer, n_head, max_seq, head_dim). Memory is allocated
    once up front (like a real server's cache pool) and `length` says how many
    positions are filled.

    Why this is safe to reuse: attention is causal, so the key/value vectors at
    position i depend only on tokens 0..i. Tokens that arrive later can never
    change them. Compute them once, keep them, and read them back forever.
    """

    def __init__(self, cfg: Config, dtype=np.float32):
        shape = (cfg.n_layer, cfg.n_head, cfg.max_seq, cfg.head_dim)
        self.cfg = cfg
        self.k = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.length = 0

    def nbytes_used(self) -> int:
        """Bytes actually holding data (K and V, all layers and heads)."""
        c = self.cfg
        return 2 * c.n_layer * c.n_head * self.length * c.head_dim * self.k.itemsize

    def copy(self, upto: int | None = None) -> "KVCache":
        """Snapshot the first `upto` positions (a prefix) into a fresh cache."""
        upto = self.length if upto is None else min(upto, self.length)
        new = KVCache(self.cfg, self.k.dtype)
        new.k[:, :, :upto] = self.k[:, :, :upto]
        new.v[:, :, :upto] = self.v[:, :, :upto]
        new.length = upto
        return new


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def _layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _softmax_(x):
    """Numerically stable softmax, computed in place (saves big temporaries)."""
    x -= x.max(-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(-1, keepdims=True)
    return x


class TinyGPT:
    def __init__(self, cfg: Config = Config(), seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d, V = cfg.d_model, cfg.vocab_size

        def w(*shape, scale=0.02):
            return (rng.standard_normal(shape) * scale).astype(np.float32)

        self.wte = w(V, d)                      # token embeddings (tied to output head)
        self.wpe = w(cfg.max_seq, d, scale=0.01)  # learned position embeddings
        self.layers = []
        for _ in range(cfg.n_layer):
            self.layers.append(dict(
                ln1_g=np.ones(d, np.float32), ln1_b=np.zeros(d, np.float32),
                w_qkv=w(d, 3 * d), b_qkv=np.zeros(3 * d, np.float32),
                w_o=w(d, d), b_o=np.zeros(d, np.float32),
                ln2_g=np.ones(d, np.float32), ln2_b=np.zeros(d, np.float32),
                w_fc=w(d, 4 * d), b_fc=np.zeros(4 * d, np.float32),
                w_proj=w(4 * d, d), b_proj=np.zeros(d, np.float32),
            ))
        self.lnf_g = np.ones(d, np.float32)
        self.lnf_b = np.zeros(d, np.float32)

    def n_params(self) -> int:
        n = self.wte.size + self.wpe.size + self.lnf_g.size + self.lnf_b.size
        for L in self.layers:
            n += sum(a.size for a in L.values())
        return n

    # ------------------------------------------------------------------ #
    def forward(self, tokens, cache: KVCache | None = None) -> np.ndarray:
        """Run the transformer on `tokens` and return logits, shape (T, vocab).

        cache=None  -> no cache. `tokens` must be the WHOLE sequence, and every
                       position's keys/values are recomputed from scratch.
        cache given -> `tokens` are only the NEW tokens. Their K/V are written
                       into the cache; attention reads old K/V straight back.
        """
        cfg = self.cfg
        tokens = np.asarray(tokens, dtype=np.int64)
        T = len(tokens)
        start = cache.length if cache is not None else 0
        end = start + T
        if end > cfg.max_seq:
            raise ValueError(f"sequence of {end} exceeds max_seq={cfg.max_seq}")

        H, hd = cfg.n_head, cfg.head_dim
        x = self.wte[tokens] + self.wpe[start:end]          # (T, d)

        # Causal mask for T new queries against all `end` keys:
        # query at absolute position start+i may see keys 0..start+i.
        mask = np.triu(np.full((T, end), -1e9, dtype=np.float32), k=start + 1)  # (T, end)

        for li, L in enumerate(self.layers):
            # ---- attention ------------------------------------------------ #
            h = _layer_norm(x, L["ln1_g"], L["ln1_b"])
            qkv = h @ L["w_qkv"] + L["b_qkv"]                # (T, 3d)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, hd).transpose(1, 0, 2)      # (H, T, hd)
            k = k.reshape(T, H, hd).transpose(1, 0, 2)
            v = v.reshape(T, H, hd).transpose(1, 0, 2)

            if cache is not None:
                # Write the new keys/values once ...
                cache.k[li, :, start:end] = k
                cache.v[li, :, start:end] = v
                # ... and attend over everything cached so far.
                K = cache.k[li, :, :end]
                Vv = cache.v[li, :, :end]
            else:
                K, Vv = k, v                                  # recomputed every call

            att = q @ K.transpose(0, 2, 1)                   # (H, T, end)
            att *= np.float32(1.0 / np.sqrt(hd))
            att += mask
            _softmax_(att)
            y = (att @ Vv).transpose(1, 0, 2).reshape(T, cfg.d_model)
            x = x + (y @ L["w_o"] + L["b_o"])

            # ---- MLP ------------------------------------------------------ #
            h = _layer_norm(x, L["ln2_g"], L["ln2_b"])
            x = x + (_gelu(h @ L["w_fc"] + L["b_fc"]) @ L["w_proj"] + L["b_proj"])

        if cache is not None:
            cache.length = end

        x = _layer_norm(x, self.lnf_g, self.lnf_b)
        return x @ self.wte.T                                 # (T, vocab)
