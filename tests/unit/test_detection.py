"""Tests for audience/species detection helpers."""

from src.services.detection import detect_audiences, detect_species


def test_detect_audiences_multilingual() -> None:
    text = "Los veterinarios y ganaderos participan en la jornada"
    result = detect_audiences(text)
    assert "veterinarians" in result
    assert "producers" in result


def test_detect_audiences_none() -> None:
    text = "Contenido general para el público"
    assert detect_audiences(text) == set()


def test_detect_species_spanish_terms() -> None:
    text = "Bioseguridad en explotaciones porcinas y avícolas"
    result = detect_species(text)
    assert result == {"swine", "poultry"}


def test_detect_species_companion_animals() -> None:
    text = "Consejos para cuidar a tu mascota y perro"
    result = detect_species(text)
    assert "companion" in result
