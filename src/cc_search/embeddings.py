"""Embeddings generation using sentence-transformers."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_TOKENS = 512


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Get the sentence transformer model (cached)."""
    return SentenceTransformer(MODEL_NAME)


def encode_text(text: str) -> list[float]:
    """Encode a single text string to an embedding."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode multiple texts to embeddings efficiently."""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return [e.tolist() for e in embeddings]
