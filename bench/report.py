"""Print the headline numbers from results/results.json as Markdown."""
import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parents[1] / "results" / "results.json").read_text())
lat, chat = R["latency"], R["chat"]
names = {"none": "No cache", "kv": "KV cache", "prefix": "KV + prefix cache"}

print(f"**Model:** {R['model']['n_layer']} layers, d={R['model']['d_model']}, "
      f"{R['model']['n_head']} heads, {R['model']['params'] / 1e6:.1f}M params, NumPy on "
      f"{R['machine']['cpus']} CPU cores\n")

print(f"### One request: {lat['generated']} tokens after a {lat['prompt_tokens']}-token prompt\n")
print("| | Total time | Token positions computed | Last token |")
print("|---|---|---|---|")
for k, label in (("no_cache", "No cache"), ("kv_cache", "KV cache")):
    d = lat[k]
    print(f"| {label} | {d['total_s']:.2f}s | {d['tokens_processed']:,} | {d['step_ms'][-1]:.1f} ms |")
print(f"\nSpeed-up: **{lat['no_cache']['total_s'] / lat['kv_cache']['total_s']:.0f}x**, "
      f"identical output tokens: **{lat['identical_tokens']}**\n")

print(f"### Support copilot: {len(chat['kv'])} requests, {R.get('chat_reply_tokens', 32)} reply tokens each\n")
print("| Strategy | Total time | Token positions computed | Avg time to first token |")
print("|---|---|---|---|")
for s in ("none", "kv", "prefix"):
    t = chat[s]
    print(f"| {names[s]} | {sum(x['total_s'] for x in t):.2f}s | "
          f"{sum(x['tokens_processed'] for x in t):,} | "
          f"{sum(x['ttft_s'] for x in t) / len(t) * 1e3:.0f} ms |")
warm = chat["prefix"][1:]
print(f"\nPrefix cache, requests 2-6: avg time to first token "
      f"**{sum(x['ttft_s'] for x in warm) / len(warm) * 1e3:.0f} ms**, "
      f"avg {sum(x['reused_tokens'] / x['prompt_tokens'] for x in warm) / len(warm):.0%} of each prompt reused.\n")

print("### KV cache memory, fp16\n")
print("| Model | Attention | KiB / token | 8K context | 32 chats x 8K |")
print("|---|---|---|---|---|")
for m in R["memory"]:
    print(f"| {m['model']} | {m['attention']} | {m['kib_per_token_fp16']:.0f} | "
          f"{m['gib_8k_context']:.2f} GiB | {m['gib_32_users_x_8k']:.0f} GiB |")
