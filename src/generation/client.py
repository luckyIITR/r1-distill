"""Async DeepSeek client with retries, rate limiting, and cost tracking.

Uses the OpenAI-compatible /chat/completions endpoint.
DeepSeek's `deepseek-reasoner` model returns reasoning in `reasoning_content`
(separate from `content`) — we handle both that format and inline <think> tags.
"""
from __future__ import annotations

import os
import time

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.generation.schema import GenerationCandidate
from src.utils import get_logger

log = get_logger(__name__)


class DeepSeekClient:
    """Thin wrapper around AsyncOpenAI configured for DeepSeek-R1."""

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 8,
        requests_per_minute: int = 60,
        timeout: float = 600.0,  # R1 reasoning can take a while
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=timeout,
        )
        self.rate_limiter = AsyncLimiter(requests_per_minute, time_period=60)
        # Bounded concurrency
        import asyncio
        self.sem = asyncio.Semaphore(max_concurrent)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_api(self, messages: list[dict], temperature: float, top_p: float, max_tokens: int):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 16000,
    ) -> GenerationCandidate:
        async with self.sem, self.rate_limiter:
            t0 = time.time()
            try:
                resp = await self._call_api(messages, temperature, top_p, max_tokens)
            except Exception as e:
                log.error(f"API call failed after retries: {e}")
                raise
            latency_ms = int((time.time() - t0) * 1000)

            choice = resp.choices[0]
            msg = choice.message

            # DeepSeek-R1 returns reasoning separately in `reasoning_content`.
            # Some endpoints / proxies may return it inline in `content` as <think>...</think>.
            reasoning = getattr(msg, "reasoning_content", None)
            content = msg.content or ""

            # If reasoning is in its own field, the answer is in content.
            # If not, the answer + <think> are both in content.
            if reasoning is not None:
                raw_response = f"<think>{reasoning}</think>\n{content}"
            else:
                raw_response = content

            return GenerationCandidate(
                raw_response=raw_response,
                reasoning=None,        # parser fills these
                answer_text=None,
                predicted_letter=None,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                finish_reason=choice.finish_reason or "unknown",
                latency_ms=latency_ms,
            )