"""Drop SFT examples that exceed max_length when tokenized with Qwen's chat template.

This avoids the dirty truncation that ms-swift would otherwise do, which can cut
off the middle of reasoning traces and degrade training signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sft_train.jsonl")
    parser.add_argument("--output", default="data/processed/sft_train_8k.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--keep_pct_min", type=float, default=0.5,
                        help="Warn if we keep less than this fraction")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)

    items = [json.loads(line) for line in Path(args.input).open()]
    print(f"Loaded {len(items)} examples from {args.input}")

    kept = []
    dropped_by_source = {"s1k": 0, "pcmb": 0}
    length_stats = []

    for item in items:
        text = tok.apply_chat_template(item["messages"], tokenize=False)
        n_tokens = len(tok.encode(text))
        length_stats.append(n_tokens)
        if n_tokens <= args.max_length:
            kept.append(item)
        else:
            dropped_by_source[item.get("source", "unknown")] = (
                dropped_by_source.get(item.get("source", "unknown"), 0) + 1
            )

    pct_kept = len(kept) / len(items)
    print(f"\nKept: {len(kept)} ({pct_kept:.1%})")
    print(f"Dropped: {len(items) - len(kept)}")
    print(f"Dropped by source: {dropped_by_source}")

    if pct_kept < args.keep_pct_min:
        print(f"\n!!! WARNING: kept fraction {pct_kept:.1%} is below {args.keep_pct_min:.1%}")
        print("    Consider raising max_length or filtering differently.")

    # Show length distribution
    import statistics
    print(f"\nLength stats (all examples):")
    print(f"  median: {statistics.median(length_stats):.0f}")
    print(f"  p90:    {sorted(length_stats)[int(len(length_stats)*0.9)]:.0f}")
    print(f"  max:    {max(length_stats):.0f}")

    # Show by source in kept set
    src_counts = {}
    for item in kept:
        s = item.get("source", "unknown")
        src_counts[s] = src_counts.get(s, 0) + 1
    print(f"\nKept by source: {src_counts}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} examples to {args.output}")


if __name__ == "__main__":
    main()