"""Audience and species detection helpers for semantic compliance."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Minimal fallback terms (used if config file missing)
_DEFAULT_AUDIENCE_TERMS: dict[str, tuple[str, ...]] = {
    "veterinarians": ("veterinario", "vet", "veterinarian"),
    "producers": ("ganadero", "farmer", "producer"),
    "pet_owners": ("propietario", "pet owner"),
    "investors": ("inversionista", "investor"),
}

_DEFAULT_SPECIES_TERMS: dict[str, tuple[str, ...]] = {
    "swine": ("porcino", "cerdo", "swine"),
    "bovine": ("bovino", "vaca", "cattle"),
    "poultry": ("avicola", "pollo", "poultry"),
    "companion": ("mascota", "perro", "gato", "pet"),
    "equine": ("equino", "caballo", "equine"),
    "aqua": ("acuicultura", "aqua", "pez"),
}


def _load_detection_terms() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Load detection terms from config file, falling back to defaults."""
    config_path = Path("data/detection_terms.json")

    if not config_path.exists():
        logger.warning(
            "Detection terms config not found at %s, using minimal defaults",
            config_path
        )
        return _DEFAULT_AUDIENCE_TERMS, _DEFAULT_SPECIES_TERMS

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        audience_terms = {
            k: tuple(v) for k, v in data.get("audience_terms", {}).items()
        }
        species_terms = {
            k: tuple(v) for k, v in data.get("species_terms", {}).items()
        }

        logger.info(
            "Loaded detection terms: %d audience categories (%d total terms), %d species categories (%d total terms)",
            len(audience_terms),
            sum(len(v) for v in audience_terms.values()),
            len(species_terms),
            sum(len(v) for v in species_terms.values())
        )
        return audience_terms, species_terms

    except Exception as exc:
        logger.error("Failed to load detection terms from %s: %s", config_path, exc)
        return _DEFAULT_AUDIENCE_TERMS, _DEFAULT_SPECIES_TERMS


# Module-level initialization
AUDIENCE_TERMS, SPECIES_TERMS = _load_detection_terms()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower()


def _find_matches(text: str, patterns: Iterable[str]) -> bool:
    snapshot = _normalize_text(text)
    return any(re.search(rf"\b{re.escape(_normalize_text(term))}\b", snapshot) for term in patterns)


def detect_audiences(text: str) -> set[str]:
    """Return normalized audience labels detected in text."""

    detected: set[str] = set()
    for audience, synonyms in AUDIENCE_TERMS.items():
        if _find_matches(text, synonyms):
            detected.add(audience)
    return detected


def detect_species(text: str) -> set[str]:
    """Return normalized species labels detected in text."""

    detected: set[str] = set()
    for species, synonyms in SPECIES_TERMS.items():
        if _find_matches(text, synonyms):
            detected.add(species)
    return detected
