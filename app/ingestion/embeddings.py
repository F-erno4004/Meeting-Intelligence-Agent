from __future__ import annotations

import math
import os
from collections.abc import Sequence

from openai import OpenAI


class OpenAIEmbeddingGenerator:
    """Generate normalized embeddings through OpenRouter in bounded batches."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 8,
        client: OpenAI | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model = model or os.getenv("EMBEDDING_MODEL")
        if not self.model:
            raise RuntimeError("EMBEDDING_MODEL is not configured.")

        dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        if not dimensions:
            raise RuntimeError("EMBEDDING_DIMENSIONS is not configured.")

        self.dimensions = int(dimensions)

        if self.dimensions < 1:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive.")

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if client is None and not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        if client is None and not base_url:
            raise RuntimeError("LLM_BASE_URL is not configured.")

        self.batch_size = batch_size

        self.client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=3,
            timeout=120.0,
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        input_type: str = "passage",
    ) -> list[list[float]]:

        if not texts:
            return []

        if input_type not in {"passage", "query"}:
            raise ValueError(
                "input_type must be either 'passage' or 'query'."
            )

        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                extra_body={"input_type": input_type},
            )

            ordered = sorted(response.data, key=lambda item: item.index)

            if len(ordered) != len(batch):
                raise RuntimeError(
                    "Embedding API returned a different number of "
                    "vectors than the number of input texts."
                )

            for item in ordered:
                original_vector = item.embedding

                if len(original_vector) < self.dimensions:
                    raise ValueError(
                        f"Model {self.model!r} returned "
                        f"{len(original_vector)} dimensions, but "
                        f"{self.dimensions} are required."
                    )

                vector = original_vector[:self.dimensions]

                norm = math.sqrt(sum(value * value for value in vector))

                if norm == 0:
                    raise ValueError(
                        "Received a zero-length embedding vector."
                    )

                vectors.append([value / norm for value in vector])

        return vectors
