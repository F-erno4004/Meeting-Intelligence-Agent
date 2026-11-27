# Meeting Intelligence Agent

## Transcript ingestion

Configure the project `.env` with the database, the existing embedding model ID, and the provider settings. Keep the exact model ID used for the stored vectors. The application also accepts the existing `OPENAI_API_KEY` setting as a fallback for `OPENROUTER_API_KEY`.

    OPENROUTER_API_KEY=
    LLM_BASE_URL=https://openrouter.ai/api/v1
    DATABASE_URL=
    EMBEDDING_MODEL=
    EMBEDDING_DIMENSIONS=1536
    EMBEDDING_BATCH_SIZE=4
    EMBEDDING_REQUEST_DELAY_SECONDS=5
    EMBEDDING_MAX_RETRIES=8
    EMBEDDING_RETRY_BASE_SECONDS=15
    EMBEDDING_RETRY_MAX_SECONDS=300
    TRANSCRIPT_DIR=data/transcripts
    CHUNK_MAX_TOKENS=550
    CHUNK_OVERLAP_TURNS=1

`EMBEDDING_BATCH_SIZE` is the maximum number of chunks per request. Set it within your provider's batch limit. `CHUNK_MAX_TOKENS` is the chunker's token target. A single longer turn remains intact. Relative `TRANSCRIPT_DIR` values are resolved from the project root.

### Register the embedding model

For a new database, first run `python -m app.memory.init_db` to create the base schema. For an existing database, verify the historical model using the configuration or request logs from the original ingestion, then set `EMBEDDING_LEGACY_MODEL` in `.env` to that exact model ID. It must match `EMBEDDING_MODEL`.

Run the additive migration once from the project root.

    python -m app.memory.embedding_space

The migration creates and seeds the `embedding_space` metadata table. The legacy model setting can be removed after registration. The previous and current generators both take the first 1536 components and apply L2 normalization, using `passage` input for ingestion and `query` input for retrieval. Ingestion and retrieval check the registered model, dimensions, and transformation before making embedding requests.

### Run or resume

    python -m app.ingestion.run

The pipeline stages one meeting and its full chunk plan in one transaction. It then commits each embedding batch. A staged meeting is complete when it has at least one chunk and every chunk has a vector. Retrieval exposes only complete meetings. The `ingested_at` timestamp records when the meeting row was created.

On daily quota exhaustion the command exits with an error and stops sending requests. Run the same command after quota reset. SHA-256 matches skip completed transcripts and resume partial transcripts using their saved chunks. A changed file with the same meeting ID produces a conflict. If a crash interrupts the transaction for the current batch, that batch may be requested again. Previously committed batches are reused.

To inspect progress in PostgreSQL, count stored and missing vectors by meeting.

    SELECT m.meeting_id,
           count(c.id) AS total_chunks,
           count(c.embedding) AS embedded_chunks,
           count(c.id) - count(c.embedding) AS missing_embeddings
    FROM meetings AS m
    LEFT JOIN transcript_chunks AS c ON c.meeting_id = m.id
    GROUP BY m.id, m.meeting_id
    ORDER BY m.meeting_id;

### Test

Run `python -m pytest tests/test_ingestion_parser.py tests/test_ingestion_resume.py` for the offline checks. Set `MIA_TEST_DATABASE_URL` to a test PostgreSQL database with pgvector installed to run the integration tests. Each integration test creates and removes its own schema.
