"""Integration tests for cross-lingual embedding behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import get_settings
from src.services.embeddings import EmbeddingService


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    denom = math.sqrt(float(np.dot(a, a))) * math.sqrt(float(np.dot(b, b)))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@pytest.mark.integration
def test_spanish_sentence_scores_highest_for_equivalent_english_phrase() -> None:
    """Ensure OpenAI cross-lingual embeddings rank matching Spanish text highest."""

    settings = get_settings()
    embedder = EmbeddingService(settings)

    english_phrase = "The rabies vaccine protects dogs and other animals."
    spanish_candidates = [
        "La vacuna contra la rabia protege a los perros y otros animales.",  # semantic match
        "El cielo de Madrid es azul y despejado hoy.",
        "Me gusta conducir coches rápidos los fines de semana.",
    ]

    english_vector = embedder.embed(english_phrase)
    spanish_vectors = [embedder.embed(text) for text in spanish_candidates]

    similarities = [_cosine_similarity(english_vector, vector) for vector in spanish_vectors]

    best_index = int(np.argmax(similarities))
    assert best_index == 0, "Equivalent Spanish sentence should score highest"
    assert similarities[0] - similarities[1] > 0.05
