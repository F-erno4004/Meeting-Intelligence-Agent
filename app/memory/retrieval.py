
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion.embeddings import OpenAIEmbeddingGenerator
from app.memory.models import Meeting, TranscriptChunk


@dataclass(frozen=True)
class RetrievedChunk:
    meeting_id: str
    source_filename: str
    chunk_index: int
    speaker: str | None
    content: str
    similarity: float


def retrieve_relevant_chunks(
    query: str,
    db: Session,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Retrieve transcript chunks using vector cosine similarity.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    generator = (
        embedding_generator
        or OpenAIEmbeddingGenerator()
    )

    query_vectors = generator.embed_texts(
        [query],
        input_type="query",
    )

    if len(query_vectors) != 1:
        raise RuntimeError(
            "Expected exactly one query embedding."
        )

    query_vector = query_vectors[0]

    distance_expression = (
        TranscriptChunk.embedding.cosine_distance(query_vector)
    )

    statement = (
        select(
            TranscriptChunk,
            Meeting,
            distance_expression.label("distance"),
        )
        .join(
            Meeting,
            TranscriptChunk.meeting_id == Meeting.id,
        )
        .where(
            TranscriptChunk.embedding.is_not(None)
        )
        .order_by(distance_expression)
        .limit(top_k)
    )

    rows = db.execute(statement).all()

    results: list[RetrievedChunk] = []

    for chunk, meeting, distance in rows:
        results.append(
            RetrievedChunk(
                meeting_id=meeting.meeting_id,
                source_filename=meeting.source_filename,
                chunk_index=chunk.chunk_index,
                speaker=chunk.speaker,
                content=chunk.content,
                similarity=1.0 - float(distance),
            )
        )

    return results
