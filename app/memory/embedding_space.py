from __future__ import annotations

import os

from sqlalchemy import select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.ingestion.embeddings import OpenAIEmbeddingGenerator
from app.memory.database import SessionLocal, engine
from app.memory.models import EMBEDDING_DIMENSIONS, EmbeddingSpace, TranscriptChunk


class EmbeddingSpaceError(RuntimeError):
    """The configured generator is incompatible with the corpus."""


def _check_dimensions(dimensions: int) -> None:
    if dimensions != EMBEDDING_DIMENSIONS:
        raise EmbeddingSpaceError(
            f"EMBEDDING_DIMENSIONS must be {EMBEDDING_DIMENSIONS} for the current pgvector schema."
        )


def _check_space(space: EmbeddingSpace, model: str, dimensions: int, transformation: str) -> None:
    if (space.model, space.dimensions, space.transformation) != (model, dimensions, transformation):
        raise EmbeddingSpaceError(
            f"Database embeddings use {space.model!r}, {space.dimensions} dimensions, "
            f"and {space.transformation!r}. Configured embeddings use {model!r}, "
            f"{dimensions} dimensions, and {transformation!r}."
        )


def require_embedding_space(db: Session, generator: OpenAIEmbeddingGenerator) -> None:
    _check_dimensions(generator.dimensions)
    setup_message = "Initialize the database embedding space with python -m app.memory.embedding_space."
    try:
        space = db.get(EmbeddingSpace, 1)
    except ProgrammingError as exc:
        if getattr(exc.orig, "sqlstate", None) == "42P01":
            raise EmbeddingSpaceError(setup_message) from exc
        raise
    if space is None:
        raise EmbeddingSpaceError(setup_message)
    _check_space(space, generator.model, generator.dimensions, generator.transformation)


def initialize_embedding_space() -> None:
    """Create and register the corpus embedding space without rewriting any vectors."""
    model = OpenAIEmbeddingGenerator.configured_model()
    dimensions = OpenAIEmbeddingGenerator.configured_dimensions()
    _check_dimensions(dimensions)
    transformation = OpenAIEmbeddingGenerator.transformation

    with SessionLocal.begin() as db:
        # Serialize concurrent attempts to create and seed the singleton table.
        db.execute(text("SELECT pg_advisory_xact_lock(849621, 1)"))
        EmbeddingSpace.__table__.create(bind=db.connection(), checkfirst=True)
        space = db.get(EmbeddingSpace, 1)
        if space is not None:
            _check_space(space, model, dimensions, transformation)
            return

        has_vectors = db.scalar(
            select(select(TranscriptChunk.id).where(TranscriptChunk.embedding.is_not(None)).exists())
        )
        if has_vectors and os.getenv("EMBEDDING_LEGACY_MODEL", "").strip() != model:
            raise EmbeddingSpaceError(
                "Existing vectors have no stored model metadata. Confirm the model used to "
                "generate them with EMBEDDING_LEGACY_MODEL in .env. It must match EMBEDDING_MODEL."
            )

        db.add(EmbeddingSpace(id=1, model=model, dimensions=dimensions, transformation=transformation))


def main() -> None:
    try:
        initialize_embedding_space()
    except (EmbeddingSpaceError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        engine.dispose()
    print("Database embedding space is registered.")


if __name__ == "__main__":
    main()
