import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: MIA/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables before importing application modules.
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

print("Environment file:", ENV_FILE)
print("Environment file exists:", ENV_FILE.exists())
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Base URL:", os.getenv("LLM_BASE_URL"))

required_vars = [
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
]

for variable in required_vars:
    if not os.getenv(variable):
        raise RuntimeError(f"{variable} is missing from .env")

# Import the actual embedding generator.
from app.ingestion.embeddings import OpenAIEmbeddingGenerator

generator = OpenAIEmbeddingGenerator(
    model=os.getenv("EMBEDDING_MODEL"),
    batch_size=1,
)

print("\nEmbedding configuration:")
print("Model:", generator.model)
print("Expected dimensions:", generator.dimensions)

# Generate an embedding through the application implementation.
vectors = generator.embed_texts(
    ["Testing OpenRouter embedding generation."]
)

if len(vectors) != 1:
    raise RuntimeError(
        f"Expected 1 embedding, received {len(vectors)}"
    )

embedding = vectors[0]

print("\nConnection successful!")
print("Actual dimensions:", len(embedding))

norm = sum(value * value for value in embedding) ** 0.5
print("Vector norm:", norm)

if len(embedding) != generator.dimensions:
    raise RuntimeError("Embedding dimension validation failed.")

print("\nEmbedding test passed!")
