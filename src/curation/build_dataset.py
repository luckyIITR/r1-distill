"""Merge s1K-1.1 + filtered PCMB into the SFT training set.

Output: data/processed/sft_train.jsonl

Each line is a dict:
  {
    "messages": [
      {"role": "user", "content": ...},
      {"role": "assistant", "content": "<think>...</think>\n\n..."}
    ],
    "source": "s1k" | "pcmb",
    "subject": str,
    "id": str,
  }

This format is what ms-swift expects (--dataset_type messages).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset

from src.generation.prompts import format_options
from src.utils import get_logger, set_seed

log = get_logger(__name__)


# ---- s1K-1.1 loading ----

def find_field(row: dict, *candidates: str) -> str | None:
    """Return the value of the first candidate field that exists and is non-empty."""
    for c in candidates:
        v = row.get(c)
        if v:
            return v
    return None


def load_s1k(dataset_name: str = "simplescaling/s1K-1.1") -> list[dict]:
    """Load s1K-1.1 and convert each row to our internal format."""
    log.info(f"Loading {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    log.info(f"  {len(ds)} rows; columns: {ds.column_names}")

    out = []
    skipped = 0
    for row in ds:
        question = find_field(row, "question")
        reasoning = find_field(row, "deepseek_thinking_trajectory", "deepseek_reasoning", "thinking_trajectories", "reasoning")
        answer = find_field(row, "deepseek_attempt", "attempt", "answer")
        subject = find_field(row, "cot_type", "category") or "unknown"

        if not (question and reasoning and answer):
            skipped += 1
            continue

        # Build the assistant response: <think>reasoning</think>\n\nanswer
        assistant_text = f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"

        # Stable ID from question hash
        qid = hashlib.sha256(question.encode()).hexdigest()[:16]

        out.append({
            "id": f"s1k-{qid}",
            "source": "s1k",
            "subject": subject,
            "user": question.strip(),
            "assistant": assistant_text,
        })

    log.info(f"  Loaded {len(out)} s1K examples; skipped {skipped} (missing fields)")
    if skipped > len(out) * 0.05:
        log.warning(f"  >5% skipped — check field names. First row: {list(ds[0].keys())}")
    return out


# ---- PCMB loading ----

def load_pcmb(traces_path: Path, source_path: Path) -> list[dict]:
    """Load filtered PCMB traces and join with original questions for the prompt."""
    log.info(f"Loading PCMB traces from {traces_path}")

    # Load source for option text (traces already have it, but defensive)
    src_by_id = {}
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            src_by_id[row["id"]] = row

    out = []
    with traces_path.open() as f:
        for line in f:
            t = json.loads(line)
            src = src_by_id.get(t["id"])
            if src is None:
                log.warning(f"  PCMB trace {t['id']} has no matching source row, skipping")
                continue

            # User message in the same format as during generation
            user_text = (
                f"Question: {t['question']}\n\n"
                f"Options:\n{format_options(src['options'])}\n\n"
                f"Solve this step by step. End your response with: "
                f'"The answer is (X)" where X is the correct letter.'
            )

            # Assistant message: wrap reasoning in <think> and append the answer
            assistant_text = (
                f"<think>\n{t['reasoning'].strip()}\n</think>\n\n"
                f"{t['answer_text'].strip()}"
            )

            out.append({
                "id": f"pcmb-{t['id']}",
                "source": "pcmb",
                "subject": t["subject"],
                "user": user_text,
                "assistant": assistant_text,
            })

    log.info(f"  Loaded {len(out)} PCMB examples")
    return out


# ---- Merging ----

def to_messages_format(item: dict) -> dict:
    """Convert internal format to ms-swift's messages format."""
    return {
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "source": item["source"],
        "subject": item["subject"],
        "id": item["id"],
    }


def dedupe_by_id(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg["seed"])

    # Load both sources
    s1k = load_s1k(cfg["s1k_dataset"])
    pcmb = load_pcmb(
        Path(cfg["pcmb_traces_path"]),
        Path(cfg["pcmb_source_path"]),
    )

    # Merge, dedupe, shuffle
    merged = s1k + pcmb
    merged = dedupe_by_id(merged)
    log.info(f"After dedup: {len(merged)} examples")

    random.shuffle(merged)
    log.info("Shuffled")

    # Convert to ms-swift schema
    final = [to_messages_format(item) for item in merged]

    # Write JSONL
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    by_source = Counter(item["source"] for item in final)
    by_subject = Counter(item["subject"] for item in final)
    log.info(f"\nWrote {len(final)} examples to {out_path}")
    log.info(f"By source: {dict(by_source)}")
    log.info(f"By subject: {dict(by_subject)}")

    # Approximate token stats
    avg_user_chars = sum(len(it["messages"][0]["content"]) for it in final) / len(final)
    avg_assist_chars = sum(len(it["messages"][1]["content"]) for it in final) / len(final)
    log.info(f"Avg user message chars: {avg_user_chars:.0f}")
    log.info(f"Avg assistant message chars: {avg_assist_chars:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)