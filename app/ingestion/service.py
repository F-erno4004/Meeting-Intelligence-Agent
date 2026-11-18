from __future__ import annotations

import hashlib
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.memory.models import Meeting, TranscriptChunk

from .chunker import chunk_transcript
from .embeddings import OpenAIEmbeddingGenerator
from .parser import parse_transcript


@dataclass(frozen=True)
class IngestionResult:
    source_filename: str
    meeting_id: str | None
    status: Literal["ingested", "duplicate", "error"]
    chunk_count: int = 0
    word_count: int = 0
    transcript_hash: str | None = None
    message: str | None = None


def _read_source(path: Path) -> tuple[bytes, str]:
    raw_bytes = path.read_bytes()
    # utf-8-sig accepts a UTF-8 BOM if present. newline="" behavior is not
    # needed here because bytes are decoded directly and line endings remain.
    text = raw_bytes.decode("utf-8-sig")
    return raw_bytes, text


def ingest_transcript_file(
    path: str | Path,
    db: Session,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    max_chunk_tokens: int = 550,
    overlap_turns: int = 1,
    embedding_batch_size: int = 64,
) -> IngestionResult:
    """
    Ingest one original meeting file.

    Deduplication is based on SHA-256 of the original file bytes. If a meeting
    ID already exists with different content, this function raises rather
    than deleting existing extracted decisions/actions/evidence.
    """
    path = Path(path)
    raw_bytes, text = _read_source(path)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    transcript = parse_transcript(text, path.name)

    # Fast duplicate/conflict check before spending tokens on embeddings.
    existing_hash = db.scalar(
        select(Meeting).where(Meeting.transcript_hash == digest)
    )
    if existing_hash is not None:
        db.rollback()
        return IngestionResult(
            source_filename=path.name,
            meeting_id=transcript.meeting_id,
            status="duplicate",
            transcript_hash=digest,
            message=f"Exact file content already ingested as {existing_hash.meeting_id}",
        )

    existing_meeting = db.scalar(
        select(Meeting).where(Meeting.meeting_id == transcript.meeting_id)
    )
    if existing_meeting is not None:
        old_hash = existing_meeting.transcript_hash
        db.rollback()
        if old_hash == digest:
            return IngestionResult(
                source_filename=path.name,
                meeting_id=transcript.meeting_id,
                status="duplicate",
                transcript_hash=digest,
                message="Meeting ID and content hash already exist",
            )
        raise ValueError(
            f"Meeting ID {transcript.meeting_id!r} already exists with a "
            "different transcript hash. Refusing to overwrite it because "
            "derived decisions/actions/evidence may depend on the old transcript."
        )

    # End the read transaction before the potentially slow embedding API call.
    db.rollback()

    chunks = chunk_transcript(
        transcript,
        max_tokens=max_chunk_tokens,
        overlap_turns=overlap_turns,
    )
    if not chunks:
        raise ValueError(f"{path.name}: parser produced no chunks")

    generator = embedding_generator or OpenAIEmbeddingGenerator(
        batch_size=embedding_batch_size
    )
    
    vectors = generator.embed_texts(
        [chunk.content for chunk in chunks],
        input_type="passage",
    )

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} embeddings, received {len(vectors)}"
        )
        
    dimensions_config = os.getenv("EMBEDDING_DIMENSIONS")

    if not dimensions_config:
        raise RuntimeError("EMBEDDING_DIMENSIONS is not configured.")

    expected_dimensions = int(dimensions_config)

    if vectors and len(vectors[0]) != expected_dimensions:
        raise ValueError(
            f"Received {len(vectors[0])}-dimension embeddings, "
            f"but EMBEDDING_DIMENSIONS is configured as "
            f"{expected_dimensions}."
        )

    if any(len(vector) != expected_dimensions for vector in vectors):
        raise ValueError("Embedding vectors have inconsistent dimensions.")

    # Recheck inside the write transaction to reduce duplicate-ingestion races.
    with db.begin():
        duplicate = db.scalar(
            select(Meeting).where(Meeting.transcript_hash == digest)
        )
        if duplicate is not None:
            return IngestionResult(
                source_filename=path.name,
                meeting_id=transcript.meeting_id,
                status="duplicate",
                transcript_hash=digest,
                message=f"Concurrent ingestion already stored {duplicate.meeting_id}",
            )

        meeting_with_id = db.scalar(
            select(Meeting).where(Meeting.meeting_id == transcript.meeting_id)
        )
        if meeting_with_id is not None:
            raise ValueError(
                f"Meeting ID {transcript.meeting_id!r} was inserted concurrently"
            )

        meeting = Meeting(
            meeting_id=transcript.meeting_id,
            team_id=transcript.team_id,
            meeting_sequence=transcript.meeting_sequence,
            source_filename=transcript.source_filename,
            transcript_hash=digest,
            word_count=len(text.split()),
        )
        db.add(meeting)
        db.flush()  # Obtain the database primary key for chunk foreign keys.

        for chunk, vector in zip(chunks, vectors, strict=True):
            db.add(
                TranscriptChunk(
                    meeting_id=meeting.id,
                    chunk_index=chunk.chunk_index,
                    speaker=chunk.speaker,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    embedding=vector,
                )
            )

    return IngestionResult(
        source_filename=path.name,
        meeting_id=transcript.meeting_id,
        status="ingested",
        chunk_count=len(chunks),
        word_count=len(text.split()),
        transcript_hash=digest,
        message="Transcript and embeddings stored successfully",
    )


def ingest_transcript_directory(
    directory: str | Path,
    session_factory: sessionmaker,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    pattern: str = "*.txt",
    max_chunk_tokens: int = 550,
    overlap_turns: int = 1,
    embedding_batch_size: int = 64,
) -> list[IngestionResult]:
    """
    Process each .txt file independently, using a fresh DB session per file.
    Files are sorted to make runs deterministic.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {directory}")

    transcript_paths = sorted(directory.glob(pattern))
    if not transcript_paths:
        raise FileNotFoundError(
            f"No transcript files matching {pattern!r} found in {directory}"
        )

    results: list[IngestionResult] = []
    for path in transcript_paths:
        db = session_factory()
        try:
            result = ingest_transcript_file(
                path,
                db,
                embedding_generator,
                max_chunk_tokens=max_chunk_tokens,
                overlap_turns=overlap_turns,
                embedding_batch_size=embedding_batch_size,
            )
            results.append(result)
        except Exception as exc:
            db.rollback()
            results.append(
                IngestionResult(
                    source_filename=path.name,
                    meeting_id=None,
                    status="error",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            db.close()

    return results
