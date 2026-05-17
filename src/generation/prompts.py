"""Prompts used for teacher inference.

The format here matters more than people realize. We want R1 to:
1. Produce a long, detailed reasoning trace
2. End with a clearly extractable answer letter

We use a minimal, standard MCQ prompt. Over-prompting (e.g., "think step
by step in detail then conclude with \\boxed{X}") sometimes makes R1
*less* effective because it tries to follow the format rather than
genuinely reason.
"""
from __future__ import annotations

from src.generation.schema import PCMBQuestion


def format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def build_user_prompt(q: PCMBQuestion) -> str:
    return (
        f"Question: {q.question}\n\n"
        f"Options:\n{format_options(q.options)}\n\n"
        f"Solve this step by step. End your response with: "
        f"\"The answer is (X)\" where X is the correct letter."
    )


def build_messages(q: PCMBQuestion) -> list[dict]:
    return [{"role": "user", "content": build_user_prompt(q)}]