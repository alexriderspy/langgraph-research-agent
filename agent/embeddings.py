"""A small local embedding model, loaded once and shared."""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """One row per text, normalised, so a dot product is the cosine similarity."""
    return _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
