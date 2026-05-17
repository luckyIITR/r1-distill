"""Generate DeepSeek-R1 reasoning traces for PCMB-500.

Usage:
    python -m src.generation.generate_traces --config configs/generation.yaml

Key properties:
- Resumable: re-running skips question IDs already in the output file
- Streaming: each trace is fsync'd to disk immediately
- Best-of-N: generates multiple candidates per question, keeps the one
  that matches ground truth (if any)
- Stats: logs running totals to W&B and stdout
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from src.generation.client import DeepSeekClient
from src.generation.parser import (
    detect_repetition_loop,
    extract_answer_letter,
    extract_reasoning,
    passes_length_bounds,
    strip_reasoning,
)
from src.generation.prompts import build_messages
from src.generation.schema import (
    GenerationCandidate,
    GenerationStats,
    PCMBQuestion,
    VerifiedTrace,
)
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def load_questions(path: str) -> list[PCMBQuestion]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(PCMBQuestion.model_validate_json(line))
    return out


def load_existing_ids(output_path: Path) -> set[str]:
    """Return set of question IDs already in the output file."""
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def parse_candidate(c: GenerationCandidate) -> GenerationCandidate:
    """Fill in reasoning/answer fields by parsing the raw response."""
    c.reasoning = extract_reasoning(c.raw_response)
    c.answer_text = strip_reasoning(c.raw_response)
    c.predicted_letter = extract_answer_letter(c.raw_response)
    return c


def verify_candidate(c: GenerationCandidate, q: PCMBQuestion) -> str | None:
    """Return None if valid, else a string reason for rejection."""
    if c.predicted_letter is None:
        return "no_answer_extracted"
    if c.predicted_letter != q.answer_letter:
        return "answer_mismatch"
    if not passes_length_bounds(c.reasoning):
        return "length_out_of_bounds"
    if c.reasoning and detect_repetition_loop(c.reasoning):
        return "repetition_loop"
    return None


async def process_question(
    q: PCMBQuestion,
    client: DeepSeekClient,
    cfg: dict,
    stats: GenerationStats,
) -> VerifiedTrace | None:
    """Generate up to N candidates for q, return first verified one or None."""
    messages = build_messages(q)
    last_reject = None

    for attempt in range(cfg["samples_per_question"]):
        try:
            cand = await client.generate(
                messages=messages,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                max_tokens=cfg["max_tokens"],
            )
        except Exception as e:
            log.warning(f"  [{q.id}] attempt {attempt + 1} API error: {e}")
            continue

        cand = parse_candidate(cand)
        stats.add_candidate(cand, cfg["cost_input_per_1m"], cfg["cost_output_per_1m"])

        reject = verify_candidate(cand, q)
        if reject is None:
            assert cand.reasoning is not None and cand.predicted_letter is not None
            return VerifiedTrace(
                id=q.id,
                question=q.question,
                options=q.options,
                answer_letter=q.answer_letter,
                subject=q.subject,
                source=q.source,
                reasoning=cand.reasoning,
                answer_text=cand.answer_text or "",
                predicted_letter=cand.predicted_letter,
                teacher_model=cfg["model"],
                reasoning_tokens=cand.output_tokens,  # approximation
                total_tokens=cand.input_tokens + cand.output_tokens,
                temperature=cfg["temperature"],
            )
        last_reject = reject
        if reject == "answer_mismatch":
            stats.answer_mismatches += 1
        elif reject in ("no_answer_extracted", "length_out_of_bounds", "repetition_loop"):
            stats.parse_failures += 1

    log.info(f"  [{q.id}] {q.subject}: rejected after {cfg['samples_per_question']} attempts (last: {last_reject})")
    return None


async def main_async(cfg: dict) -> None:
    set_seed(cfg["seed"])

    questions = load_questions(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(output_path)
    log.info(f"Loaded {len(questions)} questions; {len(existing_ids)} already done; "
             f"{len(questions) - len(existing_ids)} to go.")

    todo = [q for q in questions if q.id not in existing_ids]
    if not todo:
        log.info("Nothing to do. Exiting.")
        return

    client = DeepSeekClient(
        model=cfg["model"],
        max_concurrent=cfg["max_concurrent"],
        requests_per_minute=cfg["requests_per_minute"],
    )
    stats = GenerationStats(total_questions=len(todo))
    t_start = time.time()

    # Open output in append mode; flush+fsync per line for crash safety
    with output_path.open("a", buffering=1) as out_f:
        async def worker(q: PCMBQuestion):
            trace = await process_question(q, client, cfg, stats)
            if trace is not None:
                line = trace.model_dump_json() + "\n"
                out_f.write(line)
                out_f.flush()
                os.fsync(out_f.fileno())
                stats.verified += 1

        # Drive all questions concurrently — the client's semaphore bounds parallelism
        await tqdm.gather(*(worker(q) for q in todo), desc="generating")

    stats.elapsed_seconds = time.time() - t_start

    log.info("=" * 60)
    log.info(f"  Verified:        {stats.verified} / {stats.total_questions}")
    log.info(f"  Parse failures:  {stats.parse_failures}")
    log.info(f"  Answer mismatch: {stats.answer_mismatches}")
    log.info(f"  Total attempts:  {stats.attempted}")
    log.info(f"  Input tokens:    {stats.total_input_tokens:,}")
    log.info(f"  Output tokens:   {stats.total_output_tokens:,}")
    log.info(f"  Total cost:      ${stats.total_cost_usd:.2f}")
    log.info(f"  Elapsed:         {stats.elapsed_seconds:.0f}s")
    log.info("=" * 60)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()