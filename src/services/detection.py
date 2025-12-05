"""Audience and species detection helpers for semantic compliance."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimal fallback terms (used if config file missing)
_DEFAULT_AUDIENCE_TERMS: dict[str, tuple[str, ...]] = {
    "veterinarians": (
        "veterinario",
        "veterinaria",
        "vet",
        "veterinarian",
        "profesionales veterinarios",
    ),
    "producers": ("producer", "producers", "productor", "productores", "ganadero", "ganaderos"),
    "pet owners": ("pet owner", "pet owners", "propietario", "propietarios", "tutores de mascotas"),
    "investors": ("investor", "investors", "inversionista", "inversionistas"),
    "farmers": ("farmers", "farmer", "ganadero", "ganaderos", "agricultor", "agricultores"),
    "general public": ("general public", "publico general", "público general"),
    "media": ("media", "prensa", "press", "medios"),
    "students": ("students", "student", "estudiantes", "alumnos"),
}

_DEFAULT_SPECIES_TERMS: dict[str, tuple[str, ...]] = {
    "swine": ("porcino", "porcina", "cerdo", "swine"),
    "cattle": ("cattle", "ganado", "vacuno", "reses"),
    "beef cattle": ("beef cattle", "ganado de carne", "bovino de carne"),
    "dairy cattle": ("dairy cattle", "ganado lechero", "vacas lecheras"),
    "poultry": ("avicola", "avícola", "pollo", "poultry"),
    "equine": ("equino", "caballo", "equine"),
    "canine": ("canino", "perro", "dog", "canine"),
    "feline": ("felino", "gato", "feline"),
    "aqua": ("acuicultura", "aqua", "pez"),
    "sheep": ("oveja", "ovejas", "sheep", "ovino"),
    "small companion animals": (
        "small companion animals",
        "pequeños animales de compania",
        "small pets",
    ),
}


def _load_detection_terms() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Load detection terms from config file, falling back to defaults."""
    config_path = Path("data/detection_terms.json")

    if not config_path.exists():
        logger.warning(
            "Detection terms config not found at %s, using minimal defaults", config_path
        )
        return _DEFAULT_AUDIENCE_TERMS, _DEFAULT_SPECIES_TERMS

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        audience_terms = {k: tuple(v) for k, v in data.get("audience_terms", {}).items()}
        species_terms = {k: tuple(v) for k, v in data.get("species_terms", {}).items()}

        logger.info(
            "Loaded detection terms: %d audience categories (%d total terms), %d species categories (%d total terms)",
            len(audience_terms),
            sum(len(v) for v in audience_terms.values()),
            len(species_terms),
            sum(len(v) for v in species_terms.values()),
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


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract product names, conditions, and other entities from text.

    This is a basic implementation that can be extended with more sophisticated
    NER or dictionary-based extraction.

    Args:
        text: Text to extract entities from.

    Returns:
        Dictionary with keys 'products', 'conditions', 'technologies' containing lists of found entities.
    """
    entities: dict[str, list[str]] = {
        "products": [],
        "conditions": [],
        "technologies": [],
    }

    # Basic product name patterns (can be extended with a dictionary)
    # Look for trademark symbols and common product naming patterns
    product_patterns = [
        r"\b([A-Z][a-z]+(?:®|™|®))\b",  # ProductName®
        r"\b([A-Z][a-z]+(?:®|™)?\s+[A-Z][a-z]+)\b",  # Product Name
    ]

    # Common condition/disease patterns
    condition_patterns = [
        r"\b(ileítis|leishmaniosis|criptosporidiosis|rinotraqueítis)\b",
        r"\b(ileitis|leishmaniasis|cryptosporidiosis|rhinotracheitis)\b",
    ]

    # Technology patterns
    tech_patterns = [
        r"\b(IDAL|intradermal|monitorización|monitoring)\b",
    ]

    normalized_text = _normalize_text(text)

    for pattern in product_patterns:
        # Keep case-sensitive matching so we only capture deliberate product-style capitalization.
        matches = re.findall(pattern, text)
        entities["products"].extend([m.strip() for m in matches if m.strip()])

    for pattern in condition_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        entities["conditions"].extend([m.strip() for m in matches if m.strip()])

    for pattern in tech_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        entities["technologies"].extend([m.strip() for m in matches if m.strip()])

    # Deduplicate
    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities
