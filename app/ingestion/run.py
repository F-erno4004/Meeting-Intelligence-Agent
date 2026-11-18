from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"

    if not env_file.exists():
        raise FileNotFoundError(f".env file not found: {env_file}")

    load_dotenv(dotenv_path=env_file, override=True)

    required_vars = [
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "DATABASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
    ]

    for variable in required_vars:
        if not os.getenv(variable):
            raise RuntimeError(f"{variable} is missing or empty in .env")

    # Import application modules only after loading .env.
    from app.ingestion.embeddings import OpenAIEmbeddingGenerator
    from app.ingestion.service import ingest_transcript_directory
    from app.ingestion.service import ingest_transcript_file
    from app.memory.database import SessionLocal

    transcript_dir = Path(
        os.getenv("TRANSCRIPT_DIR", "data/transcripts")
    )

    if not transcript_dir.is_absolute():
        transcript_dir = project_root / transcript_dir

    generator = OpenAIEmbeddingGenerator(
        model=os.getenv("EMBEDDING_MODEL"),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    )

    print("Environment file:", env_file)
    print("Embedding model:", generator.model)
    print("Embedding dimensions:", generator.dimensions)
    print("Embedding API URL:", os.getenv("LLM_BASE_URL"))

    previously_ingested = {"ES2002a.txt"}

    transcript_files = sorted(
        path
        for path in transcript_dir.glob("*.txt")
        if path.name not in previously_ingested
    )

    results = []

    for transcript_file in transcript_files:
        with SessionLocal() as db:
            result = ingest_transcript_file(
                transcript_file,
                db,
                embedding_generator=generator,
                max_chunk_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "550")),
                overlap_turns=int(os.getenv("CHUNK_OVERLAP_TURNS", "1")),
            )

        results.append(result)

    print("\nINGESTION REPORT")
    print("=" * 72)

    for result in results:
        print(
            f"{result.status.upper():9} "
            f"{result.source_filename:24} "
            f"meeting={result.meeting_id or '-':12} "
            f"chunks={result.chunk_count:4} "
            f"words={result.word_count:7} "
            f"{result.message or ''}"
        )

    ingested = sum(item.status == "ingested" for item in results)
    duplicates = sum(item.status == "duplicate" for item in results)
    errors = sum(item.status == "error" for item in results)

    print("-" * 72)
    print(
        f"Total files: {len(results)} | "
        f"Ingested: {ingested} | "
        f"Duplicates: {duplicates} | "
        f"Errors: {errors}"
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()