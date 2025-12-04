"""Audience and species detection helpers for semantic compliance."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

AUDIENCE_TERMS: dict[str, tuple[str, ...]] = {
    "veterinarians": (
        "veterinario",
        "veterinaria",
        "veterinarios",
        "vet",
        "veterinarian",
        "veterinary",
        "profesional sanitario",
    ),
    "producers": (
        "ganadero",
        "ganaderos",
        "ganaderia",
        "farmer",
        "producers",
        "agricultor",
        "producer",
        "industria porcina",
        "avicultor",
    ),
    "pet_owners": (
        "propietario",
        "dueño",
        "tutor",
        "pet owner",
        "familia",
        "hogar",
    ),
    "investors": (
        "gerente",
        "director",
        "management",
        "inversionista",
        "responsable",
    ),
}

SPECIES_TERMS: dict[str, tuple[str, ...]] = {
    "swine": (
        "porcino",
        "porcina",
        "porcinos",
        "porcinas",
        "cerdo",
        "cerdos",
        "swine",
        "hog",
        "lechon",
        "lechones",
    ),
    "bovine": (
        "bovino",
        "vaca",
        "vacuno",
        "ganado",
        "cattle",
        "novilla",
        "ternero",
    ),
    "poultry": (
        "avicola",
        "avícola",
        "avicolas",
        "avícolas",
        "avicultura",
        "pollo",
        "pollos",
        "gallina",
        "gallinas",
        "broiler",
        "poultry",
    ),
    "companion": (
        "mascota",
        "perro",
        "gato",
        "pet",
        "companion animal",
    ),
    "equine": (
        "equino",
        "caballo",
        "equine",
    ),
    "aqua": (
        "acuicultura",
        "aqua",
        "pez",
        "piscicultura",
    ),
}


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
