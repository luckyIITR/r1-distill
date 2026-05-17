"""Parse <think>...</think> blocks and extract final answer letter.

R1's response format varies. We try several parsers in order, from
strictest to loosest. Each parser is cheap; we run all of them and
take the first hit.
"""
from __future__ import annotations

import re

# Reasoning extraction
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Answer extraction patterns, from most explicit to most lenient.
# Each must capture a single letter A-J in group(1).
_ANSWER_PATTERNS = [
    # "The answer is (D)", "answer: D", "final answer: D"
    re.compile(r"(?:final\s+)?answer\s*(?:is|:)\s*\(?\s*([A-J])\b", re.IGNORECASE),
    # \boxed{D} or \boxed{\text{D}}
    re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*([A-J])\b"),
    # "**D**" or "**(D)**" at the end of a line
    re.compile(r"\*\*\(?\s*([A-J])\s*\)?\*\*"),
    # "(D)" as standalone near end of text
    re.compile(r"\(\s*([A-J])\s*\)"),
    # Lone letter on a line near the end ("D" or "D.")
    re.compile(r"(?m)^\s*([A-J])\.?\s*$"),
]


def extract_reasoning(text: str) -> str | None:
    """Return the contents of the LAST <think>...</think> block, or None."""
    matches = _THINK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def strip_reasoning(text: str) -> str:
    """Return the response with <think> blocks removed."""
    return _THINK_RE.sub("", text).strip()


def extract_answer_letter(text: str) -> str | None:
    """Try to extract the final answer letter A-J from the response.

    We search only in the post-<think> portion (the 'final answer' section).
    If no <think> block, we search the whole text.
    """
    post_think = strip_reasoning(text)
    # Search the tail first — answers usually appear at the end.
    # Try patterns in order of strictness.
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(post_think)
        if matches:
            return matches[-1].upper()  # last match — likely the final answer
    return None


def detect_repetition_loop(text: str, ngram: int = 8, repeats: int = 6) -> bool:
    """Detect degenerate repetition (R1 sometimes loops on long generations).

    Returns True if any n-gram appears more than `repeats` times.
    """
    tokens = text.split()
    if len(tokens) < ngram * repeats:
        return False
    counts: dict[str, int] = {}
    for i in range(len(tokens) - ngram + 1):
        gram = " ".join(tokens[i : i + ngram])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] > repeats:
            return True
    return False


def passes_length_bounds(reasoning: str | None, min_tokens: int = 100, max_tokens: int = 16000) -> bool:
    """Reject empty/tiny or absurdly long reasoning."""
    if reasoning is None:
        return False
    approx_tokens = len(reasoning.split())
    return min_tokens <= approx_tokens <= max_tokens