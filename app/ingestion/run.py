from __future__ import annotations

import logging
import os

from app.ingestion.embeddings import OpenAIEmbeddingGenerator
from app.ingestion.service import iter_ingestion_results, transcript_paths
from app.memory.database import PROJECT_ROOT, SessionLocal
from app.memory.embedding_space import EmbeddingSpaceError, require_embedding_space


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = transcript_paths(PROJECT_ROOT / os.getenv("TRANSCRIPT_DIR", "data/transcripts"))
    generator = OpenAIEmbeddingGenerator()
    try:
        with SessionLocal.begin() as db:
            require_embedding_space(db, generator)
    except EmbeddingSpaceError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Found {len(paths)} transcripts in {paths[0].parent}", flush=True)
    print(f"Embedding model {generator.model} | {generator.dimensions} dimensions | "
          f"batch size {generator.batch_size}", flush=True)

    counts = {"ingested": 0, "duplicate": 0, "error": 0}
    for index, result in enumerate(iter_ingestion_results(paths, SessionLocal, generator), start=1):
        counts[result.status] += 1
        meeting_details = (
            f"{result.meeting_id} | {result.chunk_count} chunks | " if result.meeting_id else ""
        )
        print(
            f"[{index}/{len(paths)}] {result.source_filename} | {result.status.upper()} | "
            f"{meeting_details}{result.message}",
            flush=True,
        )
        if result.stopped:
            print("Ingestion stopped. Committed batches are saved for the next run.", flush=True)

    processed = sum(counts.values())
    print(
        f"Processed {processed}/{len(paths)} | Ingested {counts['ingested']} | "
        f"Duplicates {counts['duplicate']} | Errors {counts['error']}"
    )
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
