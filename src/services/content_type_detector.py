"""Helpers for detecting structural content types from WordPress metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentTypeRule:
    """Rule describing how to detect and boost a content type."""

    keywords: tuple[str, ...]
    bonus: float


def _load_content_type_rules() -> dict[str, ContentTypeRule]:
    config_path = Path("data/detection_content_types.json")
    if not config_path.exists():
        logger.warning("Content-type detection config missing at %s", config_path)
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - config corruption is operational
        logger.error("Failed to read %s: %s", config_path, exc)
        return {}

    default_bonus = float(payload.get("defaults", {}).get("bonus", 0.02))
    content_types: dict[str, ContentTypeRule] = {}
    for name, raw_rule in payload.get("content_types", {}).items():
        keywords = tuple((raw_rule.get("keywords") or []))
        if not keywords:
            logger.warning("Content-type rule %s has no keywords; skipping", name)
            continue
        bonus = float(raw_rule.get("bonus", default_bonus))
        content_types[name] = ContentTypeRule(keywords=tuple(k.lower() for k in keywords), bonus=bonus)
    return content_types


CONTENT_TYPE_RULES: dict[str, ContentTypeRule] = _load_content_type_rules()


def get_content_type_rules() -> Mapping[str, ContentTypeRule]:
    """Expose the loaded rule mapping for downstream callers/tests."""

    return CONTENT_TYPE_RULES


def detect_content_type(link: str | None, slug: str | None) -> str | None:
    """Infer a content type hint from URL path or slug keywords."""

    if not CONTENT_TYPE_RULES:
        return None

    path = urlparse(link or "").path.lower()
    slug_value = (slug or "").lower()
    for name, rule in CONTENT_TYPE_RULES.items():
        if any(keyword in path for keyword in rule.keywords) or any(
            keyword in slug_value for keyword in rule.keywords
        ):
            return name
    return None
