from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from email.utils import parsedate_to_datetime
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI


logger = logging.getLogger(__name__)
T = TypeVar("T")


class EmbeddingUnavailable(RuntimeError):
    """The ingestion run can resume once the provider is available again."""


class DailyEmbeddingQuotaExhausted(EmbeddingUnavailable):
    """OpenRouter rejected the request because its daily quota was exhausted."""


def _retry_after(error: APIStatusError) -> float:
    headers = error.response.headers
    delays = [0.0]
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (ValueError, OverflowError, OSError):
                pass

    reset = headers.get("x-ratelimit-reset")
    if error.status_code == 429 and reset:
        try:
            delays.append(float(reset) / 1000 - time.time())
        except ValueError:
            pass
    return max(delays)


class OpenAIEmbeddingGenerator:
    """Use one model and the same vector transformation for passages and queries."""

    transformation = "prefix-truncate-then-l2-normalize-v1"

    @staticmethod
    def configured_model() -> str:
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model or model.lower() in {"auto", "openrouter/auto"}:
            raise ValueError("EMBEDDING_MODEL must specify a fixed embedding model in .env.")
        return model

    @staticmethod
    def configured_dimensions() -> int:
        value = os.getenv("EMBEDDING_DIMENSIONS")
        if not value or int(value) < 1:
            raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer in .env.")
        return int(value)

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        client: OpenAI | None = None,
    ):
        self.model = self.configured_model()
        if model is not None and model != self.model:
            raise ValueError("The model argument must match EMBEDDING_MODEL in .env.")
        self.dimensions = self.configured_dimensions()
        self.batch_size = (
            int(os.getenv("EMBEDDING_BATCH_SIZE", "2")) if batch_size is None else batch_size
        )
        if self.batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1.")

        self.request_delay = float(os.getenv("EMBEDDING_REQUEST_DELAY_SECONDS", "2"))
        self.max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
        self.retry_base_seconds = float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "5"))
        self.retry_max_seconds = float(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "60"))
        for delay in (self.request_delay, self.retry_base_seconds, self.retry_max_seconds):
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Embedding delay settings must be finite and nonnegative.")
        if self.max_retries < 0 or self.retry_base_seconds > self.retry_max_seconds:
            raise ValueError("Invalid embedding retry count or backoff range.")

        if client is None:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("LLM_BASE_URL")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is missing from .env.")
            if not base_url:
                raise RuntimeError("LLM_BASE_URL is missing from .env.")
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0)
        # Keep retries here, including for caller-supplied SDK clients.
        self.client = client.with_options(max_retries=0)
        self._next_request_at = 0.0

    def run_with_retries(self, operation: Callable[[], T]) -> T:
        """Retry an operation that makes at most one request and commits its result."""
        backoff = self.retry_base_seconds
        for attempt in range(self.max_retries + 1):
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                return operation()
            except (APIConnectionError, APIStatusError) as exc:
                if isinstance(exc, APIStatusError):
                    if exc.status_code not in {429, 500, 502, 503, 504}:
                        raise
                    error_text = str(exc).lower().replace("_", "-")
                    if exc.status_code == 429 and "free-models-per-day" in error_text:
                        raise DailyEmbeddingQuotaExhausted(
                            "OpenRouter daily free-model quota exhausted. Rerun after it resets."
                        ) from exc
                    retry_after = _retry_after(exc)
                else:
                    retry_after = 0.0

                if attempt == self.max_retries:
                    raise EmbeddingUnavailable(
                        f"Embedding provider request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                if retry_after > self.retry_max_seconds:
                    raise EmbeddingUnavailable(
                        f"Embedding provider requested a {retry_after:.0f}s wait, "
                        "above EMBEDDING_RETRY_MAX_SECONDS. Rerun after the provider's wait."
                    ) from exc

                delay = max(retry_after, min(backoff * random.uniform(1, 1.25), self.retry_max_seconds))
                self._next_request_at = max(self._next_request_at, time.monotonic() + delay)
                logger.warning(
                    "Embedding request failed (%s). Retry %d/%d in %.1fs.",
                    type(exc).__name__, attempt + 1, self.max_retries,
                    self._next_request_at - time.monotonic(),
                )
                backoff = min(backoff * 2, self.retry_max_seconds)

        raise AssertionError("Embedding retry loop exited without a result.")

    def embed_batch(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        """Make one request. The caller can retry the enclosing database transaction."""
        if not 1 <= len(texts) <= self.batch_size:
            raise ValueError("Embedding batch is empty or exceeds EMBEDDING_BATCH_SIZE.")
        if input_type not in {"passage", "query"}:
            raise ValueError("input_type must be either 'passage' or 'query'.")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=list(texts),
                extra_body={"input_type": input_type},
            )
        finally:
            self._next_request_at = time.monotonic() + self.request_delay

        if response.model.removesuffix(":free") != self.model.removesuffix(":free"):
            raise ValueError(f"Embedding API returned model {response.model!r}; expected {self.model!r}.")
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("Embedding API did not return exactly one vector per input index.")

        vectors = []
        for item in ordered:
            if len(item.embedding) < self.dimensions:
                raise ValueError(
                    f"Model {self.model!r} returned {len(item.embedding)} dimensions, "
                    f"but {self.dimensions} are required."
                )
            vector = item.embedding[:self.dimensions]
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Received a non-finite or zero-norm embedding vector.")
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_texts(
        self,
        texts: Sequence[str],
        input_type: str = "passage",
    ) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self.run_with_retries(lambda: self.embed_batch(batch, input_type)))
        return vectors
