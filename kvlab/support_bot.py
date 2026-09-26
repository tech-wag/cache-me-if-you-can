"""
The real-usage scenario: a customer-support copilot serving several customers.

Every request the server receives is:  system prompt + chat history + new message.
We serve the exact same conversations three ways and measure what each costs:

  none    - no KV cache at all; each generated token re-reads the whole prompt
  kv      - a standard per-request KV cache (what every modern server does)
  prefix  - KV cache + prefix cache shared across turns and across customers
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .generate import generate_cached, generate_no_cache
from .model import TinyGPT
from .prefix_cache import PrefixCache
from .tokenizer import encode

# Fictional company and policy text, written for this demo.
SYSTEM_PROMPT = """You are Peach, the support copilot for Peachtree Card Services, a fictional credit card issuer.
Follow these rules on every reply.
1. Identity. Never ask for a full card number, PIN, CVV or online banking password. To verify a customer, ask for the last four digits of the card and the billing ZIP code only.
2. Disputes. A cardholder may dispute a charge within 60 days of the statement date. Collect: merchant name, transaction date, amount, and the reason (not received, not as described, duplicate, or unrecognized). Issue a provisional credit within 10 business days and tell the customer the investigation can take up to 90 days.
3. Lost or stolen cards. Lock the card immediately, confirm the last three transactions with the customer, and order a replacement. Standard delivery is 5 to 7 business days; expedited delivery is 2 business days at no charge for Platinum members.
4. Payments. Payments made before 8 PM Eastern post the same day. A returned payment carries a fee of up to 25 dollars. Offer to set up autopay for the minimum due, the statement balance, or a fixed amount.
5. Credit limit requests. You may not approve or deny a credit limit increase. Explain that the request is reviewed within 2 business days and may include a soft credit inquiry that does not affect the credit score.
6. Fees. Late fee is up to 30 dollars. Foreign transaction fee is 0 percent on Platinum and 3 percent on Classic. Annual fee is 0 on Classic and 95 dollars on Platinum.
7. Tone. Be warm, brief and specific. Use plain language, no jargon. Confirm the next step and the timeline at the end of every reply.
8. Escalation. Hand off to a human agent for fraud over 1,000 dollars, bankruptcy, a deceased cardholder, or any legal threat.
"""

CONVERSATIONS = {
    "maya": [
        "Hi, I see a charge for 89.40 from StreamMax on Sept 3 that I never signed up for.",
        "Last four are 4417, ZIP 30309. Can you dispute it?",
        "Great. How long until I see the credit back on my account?",
    ],
    "dev": [
        "I think I left my card at a restaurant last night. Can you lock it?",
        "Yes, the last three were groceries, gas and the restaurant. I'm Platinum.",
        "Perfect, please send the replacement expedited to my address on file.",
    ],
}

# Order the server actually sees requests arrive in (interleaved customers).
ARRIVAL_ORDER = [("maya", 0), ("maya", 1), ("dev", 0), ("maya", 2), ("dev", 1), ("dev", 2)]


def _wrap_system(text: str) -> list[int]:
    return encode(f"<|system|>\n{text}")


def _wrap_user(text: str) -> list[int]:
    return encode(f"\n<|user|>\n{text}\n<|assistant|>\n")


@dataclass
class TurnMetrics:
    strategy: str
    customer: str
    turn: int
    prompt_tokens: int      # full request length the model conditions on
    reused_tokens: int      # served from prefix cache (no compute)
    prefilled_tokens: int   # prompt tokens actually computed before first reply token
    ttft_s: float           # time to first token (what the user feels as "lag")
    total_s: float          # whole reply
    tokens_processed: int   # every token-position pushed through the network


class SupportServer:
    def __init__(self, model: TinyGPT, strategy: str, reply_tokens: int = 32):
        assert strategy in {"none", "kv", "prefix"}
        self.model = model
        self.strategy = strategy
        self.reply_tokens = reply_tokens
        self.histories: dict[str, list[int]] = {}
        self.prefix_cache = PrefixCache(max_entries=16) if strategy == "prefix" else None
        self.system_tokens = _wrap_system(SYSTEM_PROMPT)

    def handle(self, customer: str, turn: int, message: str) -> TurnMetrics:
        history = self.histories.get(customer, list(self.system_tokens))
        prompt = history + _wrap_user(message)

        reused = 0
        if self.strategy == "none":
            res = generate_no_cache(self.model, prompt, self.reply_tokens)
        elif self.strategy == "kv":
            res, _ = generate_cached(self.model, prompt, self.reply_tokens)
        else:
            t0 = time.perf_counter()
            cache, reused = self.prefix_cache.lookup(prompt)
            lookup_s = time.perf_counter() - t0     # copying the snapshot is not free: count it
            res, cache = generate_cached(self.model, prompt, self.reply_tokens, cache)
            res.step_times_s[0] += lookup_s
            res.ttft_s += lookup_s
            self.prefix_cache.insert(prompt + res.tokens, cache)

        self.histories[customer] = prompt + res.tokens
        return TurnMetrics(
            strategy=self.strategy, customer=customer, turn=turn,
            prompt_tokens=len(prompt), reused_tokens=reused,
            prefilled_tokens=len(prompt) - reused,
            ttft_s=res.ttft_s, total_s=res.total_s,
            tokens_processed=res.tokens_processed,
        )


def run_scenario(model: TinyGPT, strategy: str, reply_tokens: int = 32) -> list[TurnMetrics]:
    server = SupportServer(model, strategy, reply_tokens)
    return [server.handle(c, t, CONVERSATIONS[c][t]) for c, t in ARRIVAL_ORDER]
