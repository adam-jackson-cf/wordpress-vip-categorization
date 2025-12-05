"""Tests for content type detector helpers."""

from src.services.content_type_detector import (
    ContentTypeRule,
    detect_content_type,
    get_content_type_rules,
)


def test_rules_loaded() -> None:
    rules = get_content_type_rules()
    assert isinstance(rules, dict)
    assert "Product Catalogue Listing Page" in rules
    assert isinstance(rules["Product Catalogue Listing Page"], ContentTypeRule)
    assert rules["Product Catalogue Listing Page"].bonus > 0


def test_detect_content_type_matches_keywords() -> None:
    hint = detect_content_type("https://example.com/lista-de-productos/", "lista-de-productos")
    assert hint == "Product Catalogue Listing Page"


def test_detect_content_type_handles_unknown() -> None:
    hint = detect_content_type("https://example.com/random", "random")
    assert hint is None
