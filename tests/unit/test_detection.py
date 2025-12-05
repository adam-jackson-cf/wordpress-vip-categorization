"""Tests for audience/species detection helpers."""

from src.services.detection import AUDIENCE_TERMS, SPECIES_TERMS, detect_audiences, detect_species


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


def test_detection_terms_loaded_from_config() -> None:
    """Verify terms are loaded from external config."""
    # Should have expanded terms from config file
    assert "ganaderia extensiva" in AUDIENCE_TERMS["producers"]
    assert "tutores de mascotas" in AUDIENCE_TERMS["pet owners"]
    assert "profesionales veterinarios" in AUDIENCE_TERMS["veterinarians"]

    assert "aves acuaticas" in SPECIES_TERMS["poultry"]
    assert "piscifactoria" in SPECIES_TERMS["aqua"]
    assert "ganaderia porcina" in SPECIES_TERMS["swine"]


def test_detect_spanish_audience_variants() -> None:
    """Test expanded Spanish audience terms."""
    text = "El evento está dirigido a profesionales veterinarios y ganaderia extensiva"
    result = detect_audiences(text)
    assert "veterinarians" in result
    assert "producers" in result


def test_detect_aquatic_species() -> None:
    """Test aquatic and poultry species terms."""
    text = "Manejo de piscifactorias y aves acuaticas"
    result = detect_species(text)
    assert "aqua" in result
    assert "poultry" in result
