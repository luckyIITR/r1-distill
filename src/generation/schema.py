"""Pydantic models for the generation pipeline.

Type safety is the cheapest bug prevention you'll ever get. Everything
that crosses an I/O boundary (file, API, network) is validated here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PCMBQuestion(BaseModel):
    """A single PCMB MCQ from data/raw/pcmb_500.jsonl."""
    id: str
    question: str
    options: list[str]
    answer_index: int
    answer_letter: str
    subject: Literal["physics", "chemistry", "biology", "math"]
    source: str
    source_id: str


class GenerationCandidate(BaseModel):
    """A single completion attempt from the teacher model."""
    raw_response: str
    reasoning: str | None         # extracted <think>...</think> content
    answer_text: str | None       # extracted final answer (whatever R1 said)
    predicted_letter: str | None  # parsed answer letter A-J or None
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: int


class VerifiedTrace(BaseModel):
    """A trace that passed all quality filters and matched ground truth."""
    id: str                        # same as PCMBQuestion.id
    question: str
    options: list[str]
    answer_letter: str             # ground truth
    subject: str
    source: str
    # Generation outputs
    reasoning: str
    answer_text: str
    predicted_letter: str
    # Metadata
    teacher_model: str
    reasoning_tokens: int
    total_tokens: int
    temperature: float
    # Quality fields
    verification_status: Literal["matched"] = "matched"


class GenerationStats(BaseModel):
    total_questions: int = 0
    attempted: int = 0
    parse_failures: int = 0
    answer_mismatches: int = 0
    verified: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0

    def add_candidate(self, c: GenerationCandidate, cost_input_per_1m: float, cost_output_per_1m: float) -> None:
        self.attempted += 1
        self.total_input_tokens += c.input_tokens
        self.total_output_tokens += c.output_tokens
        self.total_cost_usd += (
            c.input_tokens / 1_000_000 * cost_input_per_1m
            + c.output_tokens / 1_000_000 * cost_output_per_1m
        )