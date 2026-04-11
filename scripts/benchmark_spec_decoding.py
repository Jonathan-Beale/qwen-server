#!/usr/bin/env python3
"""
Benchmark speculative decoding configurations.

Runs prompts at two token-length tiers and records tok/s and latency.
Results are saved to a JSON file so configs can be compared across runs.

Usage:
    # Benchmark current server and save as "eagle"
    python benchmark_spec_decoding.py --label eagle

    # Show comparison of all saved results
    python benchmark_spec_decoding.py --compare

    # Override server URL or results file
    python benchmark_spec_decoding.py --label baseline --url http://localhost:8001
    python benchmark_spec_decoding.py --compare --results results.json
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime

DEFAULT_URL = "http://localhost:8000"
DEFAULT_RESULTS = "benchmark_results.json"
MODEL = "/models/Qwen2-7B-Instruct-AWQ"

# Short prompts — output capped at 128 tokens
SHORT_PROMPTS = [
    ("short", "What is the capital of France?", 128),
    ("short", "Name the three primary colors.", 128),
    ("short", "What does HTTP stand for?", 128),
    ("short", "What is the time complexity of quicksort on average?", 128),
    ("short", "Explain the difference between TCP and UDP in two sentences.", 128),
    ("short", "What is the boiling point of water at sea level in Celsius?", 128),
]

# Medium prompts — output capped at 512 tokens
MEDIUM_PROMPTS = [
    ("medium",
     "Write a Python function that takes a list of integers and returns a new list "
     "with duplicates removed, preserving the original order.",
     512),
    ("medium",
     "Explain how garbage collection works in Python. Cover reference counting, "
     "the cyclic garbage collector, and when each triggers.",
     512),
    ("medium",
     "Write a SQL query to find the top 5 customers by total order value from tables "
     "`orders` (id, customer_id, amount) and `customers` (id, name).",
     512),
    ("medium",
     "Debug this Python code and explain the fix:\n"
     "def flatten(lst):\n    result = []\n    for item in lst:\n"
     "        if isinstance(item, list):\n            result.extend(flatten(item))\n"
     "        result.append(item)\n    return result",
     512),
]

# Long prompts — output capped at 1500 tokens, pushing near the 2048 context limit
LONG_PROMPTS = [
    ("long",
     "Write a complete Python implementation of a red-black tree with insert, delete, "
     "and search operations. Include a Node class, full docstrings, and a brief usage example.",
     1500),
    ("long",
     "Write a detailed technical explanation of how the transformer attention mechanism works. "
     "Cover scaled dot-product attention, multi-head attention, positional encoding, and why "
     "each design choice matters. Include concrete numerical examples where helpful.",
     1500),
    ("long",
     "Implement a complete HTTP/1.1 server in Python using only the standard library. "
     "Support GET and POST, JSON request/response bodies, routing via a decorator, "
     "and proper error responses (400, 404, 500).",
     1500),
    ("long",
     "Write a comprehensive guide to async/await in Python. Cover the event loop, "
     "coroutines, Tasks, gather, asyncio.run, and common pitfalls. "
     "Include working code examples for each concept.",
     1500),
]

ALL_PROMPTS = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS


def chat(url: str, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        elapsed = time.perf_counter() - t0
        result = json.loads(resp.read())
    usage = result.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    return {
        "elapsed": elapsed,
        "tokens": tokens,
        "tps": tokens / elapsed if elapsed > 0 else 0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def run_suite(url: str, label: str, runs: int) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")

    tier_results = {"short": [], "medium": [], "long": []}

    for tier, prompt, max_tokens in ALL_PROMPTS:
        prompt_results = []
        short_prompt = prompt[:70] + ("..." if len(prompt) > 70 else "")
        print(f"\n[{tier}] {short_prompt}")
        for i in range(runs):
            r = chat(url, prompt, max_tokens)
            prompt_results.append(r)
            print(f"  run {i+1}: {r['tokens']:4d} tok out  "
                  f"{r['prompt_tokens']:3d} tok in  "
                  f"{r['total_tokens']:4d} total  "
                  f"{r['elapsed']:.1f}s  {r['tps']:.1f} tok/s")
        avg_tps = statistics.mean(p["tps"] for p in prompt_results)
        avg_lat = statistics.mean(p["elapsed"] for p in prompt_results)
        avg_tok = statistics.mean(p["tokens"] for p in prompt_results)
        avg_total = statistics.mean(p["total_tokens"] for p in prompt_results)
        print(f"  avg:  {avg_tok:.0f} tok out  {avg_total:.0f} total  "
              f"{avg_lat:.1f}s  {avg_tps:.1f} tok/s")
        tier_results[tier].append({
            "prompt": prompt[:80],
            "avg_tps": avg_tps,
            "avg_latency": avg_lat,
            "avg_tokens": avg_tok,
            "avg_total_tokens": avg_total,
        })

    all_tiers = [r for tier in tier_results.values() for r in tier]
    return {
        "label": label,
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "short": tier_results["short"],
        "medium": tier_results["medium"],
        "long": tier_results["long"],
        "summary": {
            "short_tps": statistics.mean(r["avg_tps"] for r in tier_results["short"]),
            "medium_tps": statistics.mean(r["avg_tps"] for r in tier_results["medium"]),
            "long_tps": statistics.mean(r["avg_tps"] for r in tier_results["long"]),
            "overall_tps": statistics.mean(r["avg_tps"] for r in all_tiers),
            "short_latency": statistics.mean(r["avg_latency"] for r in tier_results["short"]),
            "medium_latency": statistics.mean(r["avg_latency"] for r in tier_results["medium"]),
            "long_latency": statistics.mean(r["avg_latency"] for r in tier_results["long"]),
        },
    }


def save_result(result: dict, path: str):
    if os.path.exists(path):
        all_results = json.load(open(path))
    else:
        all_results = {}
    all_results[result["label"]] = result
    json.dump(all_results, open(path, "w"), indent=2)
    print(f"\nSaved to {path}")


def print_comparison(path: str):
    if not os.path.exists(path):
        print(f"No results file found at {path}")
        return
    all_results = json.load(open(path))
    if not all_results:
        print("No results to compare.")
        return

    labels = list(all_results.keys())
    print(f"\n{'='*72}")
    print("  COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Config':<22} {'Short tok/s':>11} {'Med tok/s':>10} {'Long tok/s':>11} "
          f"{'Overall tok/s':>14} {'Speedup':>8}")
    print(f"  {'-'*22} {'-'*11} {'-'*10} {'-'*11} {'-'*14} {'-'*8}")

    baseline_overall = None
    for label in labels:
        s = all_results[label]["summary"]
        if baseline_overall is None:
            baseline_overall = s["overall_tps"]
        speedup = s["overall_tps"] / baseline_overall if baseline_overall else 1.0
        speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
        med = s.get("medium_tps", s.get("long_tps", 0))
        print(f"  {label:<22} {s['short_tps']:>9.1f}   {med:>8.1f}   {s['long_tps']:>9.1f}   "
              f"{s['overall_tps']:>12.1f}   {speedup_str:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None,
                        help="Label for this run (e.g. 'baseline', 'eagle')")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--runs", type=int, default=2, help="Runs per prompt (default: 2)")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help=f"Results JSON file (default: {DEFAULT_RESULTS})")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table of all saved results and exit")
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.results)
        return

    if not args.label:
        parser.error("--label is required when benchmarking (e.g. --label eagle)")

    result = run_suite(args.url, args.label, args.runs)
    save_result(result, args.results)

    s = result["summary"]
    print(f"\n{'='*64}")
    print(f"  SUMMARY — {args.label}")
    print(f"{'='*64}")
    print(f"  Short prompts:   {s['short_tps']:.1f} tok/s avg  |  {s['short_latency']:.2f}s latency")
    print(f"  Medium prompts:  {s.get('medium_tps', 0):.1f} tok/s avg  |  {s.get('medium_latency', 0):.2f}s latency")
    print(f"  Long prompts:    {s['long_tps']:.1f} tok/s avg   |  {s['long_latency']:.2f}s latency")
    print(f"  Overall:         {s['overall_tps']:.1f} tok/s avg")
    print(f"\nRun with --compare to see all configs side-by-side.")


if __name__ == "__main__":
    main()
