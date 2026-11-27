from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app.memory.embedding_space import EmbeddingSpaceError, require_embedding_space
from app.memory.models import Meeting, TranscriptChunk

from .chunker import chunk_transcript
from .embeddings import EmbeddingUnavailable, OpenAIEmbeddingGenerator
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
    stopped: bool = False


def ingest_transcript_file(
    path: str | Path,
    db: Session,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    max_chunk_tokens: int | None = None,
    overlap_turns: int | None = None,
    embedding_batch_size: int | None = None,
) -> IngestionResult:
    """Stage a complete chunk plan and commit each embedding batch using an idle session."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    with db.begin():
        # meeting_id is unique. This lock also serializes identical source bytes
        # ingested under different filenames when the meeting ID is filename-derived.
        lock_key = int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=True)
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        existing = db.execute(
            select(Meeting.id, Meeting.meeting_id, Meeting.word_count)
            .where(Meeting.transcript_hash == digest)
            .order_by(Meeting.id)
            .limit(1)
        ).first()
        if existing is not None:
            meeting_pk, meeting_id, word_count = existing
            chunk_count, embedded_count = db.execute(
                select(func.count(), func.count(TranscriptChunk.embedding))
                .where(TranscriptChunk.meeting_id == meeting_pk)
            ).one()
            if chunk_count and chunk_count == embedded_count:
                return IngestionResult(
                    path.name, meeting_id, "duplicate", chunk_count, word_count or 0, digest,
                    "Transcript already complete",
                )
        else:
            chunk_count = embedded_count = 0

        generator = embedding_generator or OpenAIEmbeddingGenerator(batch_size=embedding_batch_size)
        require_embedding_space(db, generator)

        if not chunk_count:
            # Decode directly to preserve the source offsets, including CRLF.
            source = raw.decode("utf-8-sig")
            transcript = parse_transcript(source, path.name)
            if existing is None:
                meeting_id = transcript.meeting_id
                word_count = len(source.split())
                meeting_pk = db.scalar(
                    pg_insert(Meeting)
                    .values(
                        meeting_id=meeting_id,
                        team_id=transcript.team_id,
                        meeting_sequence=transcript.meeting_sequence,
                        source_filename=transcript.source_filename,
                        transcript_hash=digest,
                        word_count=word_count,
                    )
                    .on_conflict_do_nothing(index_elements=[Meeting.meeting_id])
                    .returning(Meeting.id)
                )
                if meeting_pk is None:
                    raise ValueError(
                        f"Meeting ID {meeting_id!r} already exists with a different transcript hash."
                    )

            chunks = chunk_transcript(
                transcript,
                max_tokens=(
                    int(os.getenv("CHUNK_MAX_TOKENS", "550"))
                    if max_chunk_tokens is None else max_chunk_tokens
                ),
                overlap_turns=(
                    int(os.getenv("CHUNK_OVERLAP_TURNS", "1"))
                    if overlap_turns is None else overlap_turns
                ),
            )
            if not chunks:
                raise ValueError(f"{path.name}: parser produced no chunks")
            db.execute(
                insert(TranscriptChunk),
                [
                    {
                        "meeting_id": meeting_pk,
                        "chunk_index": chunk.chunk_index,
                        "speaker": chunk.speaker,
                        "content": chunk.content,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    }
                    for chunk in chunks
                ],
            )
            chunk_count = len(chunks)

    def store_next_batch() -> int:
        with db.begin():
            # Acquire the meeting lock before looking for missing vectors. A
            # competing worker will see the committed batch before making a request.
            db.execute(select(Meeting.id).where(Meeting.id == meeting_pk).with_for_update()).scalar_one()
            pending = db.scalars(
                select(TranscriptChunk)
                .where(
                    TranscriptChunk.meeting_id == meeting_pk,
                    TranscriptChunk.embedding.is_(None),
                )
                .order_by(TranscriptChunk.chunk_index)
                .limit(generator.batch_size)
            ).all()
            if not pending:
                return 0
            vectors = generator.embed_batch([chunk.content for chunk in pending], input_type="passage")
            for chunk, vector in zip(pending, vectors, strict=True):
                chunk.embedding = vector
            return len(pending)

    remaining = chunk_count - embedded_count
    stored = 0
    while stored < remaining:
        batch_count = generator.run_with_retries(store_next_batch)
        if not batch_count:
            break
        stored += batch_count

    status = "ingested" if stored else "duplicate"
    message = "Transcript and embeddings stored successfully" if stored else "Transcript already complete"
    return IngestionResult(path.name, meeting_id, status, chunk_count, word_count or 0, digest, message)


def transcript_paths(directory: str | Path, pattern: str = "*.txt") -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {directory}")
    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No transcript files matching {pattern!r} found in {directory}")
    return paths


def iter_ingestion_results(
    paths: Iterable[Path],
    session_factory: sessionmaker,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    max_chunk_tokens: int | None = None,
    overlap_turns: int | None = None,
    embedding_batch_size: int | None = None,
) -> Iterator[IngestionResult]:
    """Process one source at a time and stop the run on an unavailable provider."""
    generator = embedding_generator or OpenAIEmbeddingGenerator(batch_size=embedding_batch_size)
    for path in paths:
        try:
            with session_factory() as db:
                result = ingest_transcript_file(
                    path, db, generator,
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_turns=overlap_turns,
                )
        except Exception as exc:
            result = IngestionResult(
                source_filename=path.name,
                meeting_id=None,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                stopped=isinstance(exc, (EmbeddingUnavailable, EmbeddingSpaceError)),
            )
        yield result
        if result.stopped:
            break


def ingest_transcript_directory(
    directory: str | Path,
    session_factory: sessionmaker,
    embedding_generator: OpenAIEmbeddingGenerator | None = None,
    *,
    pattern: str = "*.txt",
    max_chunk_tokens: int | None = None,
    overlap_turns: int | None = None,
    embedding_batch_size: int | None = None,
) -> list[IngestionResult]:
    """Ingest files in deterministic order using a fresh session per file."""
    return list(
        iter_ingestion_results(
            transcript_paths(directory, pattern), session_factory, embedding_generator,
            max_chunk_tokens=max_chunk_tokens,
            overlap_turns=overlap_turns,
            embedding_batch_size=embedding_batch_size,
        )
    )
