"""Source PCMB-500 from MMLU-Pro, decontaminated against GPQA.

Usage:
    python -m src.curation.source_pcmb --config configs/pcmb_source.yaml

Output schema (JSONL):
    {
      "id": str,                # stable hash of question text
      "question": str,
      "options": list[str],     # the answer choices
      "answer_index": int,      # 0-indexed
      "answer_letter": str,     # "A", "B", ...
      "subject": str,           # physics|chemistry|biology|math
      "source": "mmlu-pro",
      "source_id": str,         # original ID from MMLU-Pro
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset

from src.utils import get_logger, set_seed
from src.config import settings

log = get_logger(__name__)

# MMLU-Pro uses "category" field; map to our subject names.
# "math" subject doesn't exist in PCMB conventionally but it's part of P-C-M-B.
SUBJECT_MAP = {
    "physics": "physics",
    "chemistry": "chemistry",
    "biology": "biology",
    "math": "math",
}


def stable_id(text: str) -> str:
    """Stable 16-char hex hash of the question text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For dedup/decontam."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(text: str, n: int) -> set[str]:
    tokens = normalize(text).split()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_gpqa_ngrams(dataset_name: str, config: str, n: int) -> list[set[str]]:
    """Load GPQA and return list of n-gram sets for each question."""
    log.info(f"Loading GPQA for decontamination: {dataset_name}/{config}")
    ds = load_dataset(dataset_name, config, split="train")
    grams = []
    for row in ds:
        q = row.get("Question") or row.get("question") or ""
        grams.append(ngrams(q, n))
    log.info(f"Built n-grams for {len(grams)} GPQA questions")
    return grams


def is_contaminated(question: str, gpqa_grams: list[set[str]], threshold: float, n: int) -> bool:
    """Return True if this question is too similar to any GPQA question."""
    q_grams = ngrams(question, n)
    if not q_grams:
        return False
    return any(jaccard(q_grams, g) >= threshold for g in gpqa_grams)


def sample_subject(
    rows: list[dict],
    subject: str,
    k: int,
    gpqa_grams: list[set[str]],
    threshold: float,
    n: int,
    seed: int,
) -> list[dict]:
    """Sample k questions from this subject, after decontamination."""
    import random
    rng = random.Random(seed + hash(subject) % 10000)

    candidates = [r for r in rows if r["category"].lower() == subject]
    log.info(f"  {subject}: {len(candidates)} candidates in MMLU-Pro")
    rng.shuffle(candidates)

    out, skipped = [], 0
    for r in candidates:
        if len(out) >= k:
            break
        if is_contaminated(r["question"], gpqa_grams, threshold, n):
            skipped += 1
            continue
        out.append(r)

    log.info(f"  {subject}: kept {len(out)}, decontam-skipped {skipped}")
    if len(out) < k:
        log.warning(f"  {subject}: only got {len(out)}/{k} after decontamination")
    return out


def to_output_schema(row: dict, subject: str) -> dict:
    return {
        "id": stable_id(row["question"]),
        "question": row["question"],
        "options": row["options"],
        "answer_index": row["answer_index"],
        "answer_letter": row["answer"],
        "subject": subject,
        "source": "mmlu-pro",
        "source_id": str(row.get("question_id", "")),
    }


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg["seed"])

    log.info(f"Loading MMLU-Pro: {cfg['source_dataset']}")
    ds = load_dataset(cfg["source_dataset"], split=cfg["source_split"])
    rows = list(ds)
    log.info(f"Loaded {len(rows)} MMLU-Pro rows total")

    # Decontaminate against GPQA
    gpqa_grams = []
    if cfg["decontam"]["enabled"]:
        gpqa_grams = build_gpqa_ngrams(
            cfg["decontam"]["gpqa_dataset"],
            cfg["decontam"]["gpqa_config"],
            cfg["decontam"]["ngram_size"],
        )

    # Sample per subject, dedup by stable_id across subjects
    selected = []
    seen_ids: set[str] = set()
    counts = defaultdict(int)
    cross_subject_dups = 0

    # Oversample slightly per subject so we still hit target after dedup
    target_per_subject = cfg["samples_per_subject"]
    oversample = int(target_per_subject * 1.2)  # 20% buffer

    for subject in cfg["subjects"]:
        sub_rows = sample_subject(
            rows,
            subject,
            oversample,  # request more, dedup down
            gpqa_grams,
            cfg["decontam"]["similarity_threshold"],
            cfg["decontam"]["ngram_size"],
            cfg["seed"],
        )
        kept_this_subject = 0
        for r in sub_rows:
            if kept_this_subject >= target_per_subject:
                break
            item = to_output_schema(r, SUBJECT_MAP[subject])
            if item["id"] in seen_ids:
                cross_subject_dups += 1
                continue
            seen_ids.add(item["id"])
            selected.append(item)
            counts[subject] += 1
            kept_this_subject += 1

    log.info(f"Cross-subject dedup removed: {cross_subject_dups} duplicates")
    log.info(f"Final counts: {dict(counts)}")
    log.info(f"Total selected: {len(selected)}")

    # Write JSONL
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(selected)} examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)