"""Tests for configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


def _base_settings_kwargs() -> dict[str, object]:
    return {
        "supabase_url": "https://example.supabase.co/rest/v1",
        "supabase_key": "secret",
        "semantic_api_key": "sem-key",
        "llm_api_key": "llm-key",
        "wordpress_vip_site_tokens": "https://site.com|tokenA, https://site2.com|tokenB",
    }


def test_settings_normalizes_urls_and_tokens(tmp_path: Path) -> None:
    kwargs = _base_settings_kwargs()
    kwargs["taxonomy_file_path"] = tmp_path / "taxonomy.csv"
    settings = Settings(**kwargs)

    assert settings.supabase_url == "https://example.supabase.co"
    assert settings.semantic_base_url.endswith("/v1")
    assert settings.llm_base_url.endswith("/v1")
    assert 0 <= settings.similarity_threshold <= 1

    pairs = settings.get_wordpress_site_tokens()
    assert pairs == [
        ("https://site.com", "tokenA"),
        ("https://site2.com", "tokenB"),
    ]


def test_get_wordpress_site_tokens_requires_token(tmp_path: Path) -> None:
    kwargs = _base_settings_kwargs()
    kwargs["taxonomy_file_path"] = tmp_path / "taxonomy.csv"
    kwargs["wordpress_vip_site_tokens"] = "https://site.com"
    settings = Settings(**kwargs)

    with pytest.raises(ValueError):
        settings.get_wordpress_site_tokens()
