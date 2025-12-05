"""Analyze WordPress pages that lack detected audience/species signals."""

# ruff: noqa: E402  # requires sys.path mutation before importing project modules

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings
from src.connectors.wordpress_vip import WordPressVIPConnector
from src.data.supabase_client import SupabaseClient
from src.services.detection import (
    AUDIENCE_TERMS,
    SPECIES_TERMS,
    detect_audiences,
    detect_species,
)

logger = logging.getLogger(__name__)


def _fetch_rows(
    client: SupabaseClient, limit: int, content_ids: list[str] | None
) -> list[dict[str, Any]]:
    if content_ids:
        result = (
            client.client.table("wordpress_content").select("*").in_("id", content_ids).execute()
        )
        return result.data

    result = (
        client.client.table("wordpress_content")
        .select("*")
        .order("published_date", desc=True)
        .limit(limit * 4)
        .execute()
    )
    rows: list[dict[str, Any]] = []
    for row in result.data:
        metadata = row.get("metadata") or {}
        audiences = row.get("detected_audiences") or metadata.get("detected_audiences") or []
        species = row.get("detected_species") or metadata.get("detected_species") or []
        if not audiences or not species:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows[:limit]


def _extract_text(url: str) -> str:
    parsed = urlparse(url)
    base = (
        f"{parsed.scheme}://{parsed.netloc}"
        if parsed.scheme and parsed.netloc
        else "https://example.com"
    )
    connector = WordPressVIPConnector(site_url=base)
    verify = os.environ.get("REQUESTS_CA_BUNDLE") or True
    resp = requests.get(url, timeout=30, verify=verify)
    resp.raise_for_status()
    return connector._extract_text_content(resp.text)  # type: ignore[attr-defined]


def _detect(text: str, audience: bool = True) -> set[str]:
    return detect_audiences(text) if audience else detect_species(text)


def _find_snippet(text: str, label: str, term_map: dict[str, tuple[str, ...]]) -> str | None:
    candidates = term_map.get(label, ())
    lowered = text.lower()
    for candidate in candidates:
        idx = lowered.find(candidate.lower())
        if idx == -1:
            continue
        start = max(0, idx - 120)
        end = min(len(text), idx + 120)
        snippet = text[start:end]
        return " ".join(snippet.split())
    return None


def analyze(limit: int, content_ids: list[str] | None, output: Path) -> list[dict[str, Any]]:
    settings = Settings()
    client = SupabaseClient(settings)
    rows = _fetch_rows(client, limit, content_ids)
    logger.info("Analyzing %s content rows", len(rows))

    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for row in rows:
        metadata = row.get("metadata") or {}
        row_audiences = row.get("detected_audiences") or metadata.get("detected_audiences") or []
        row_species = row.get("detected_species") or metadata.get("detected_species") or []
        try:
            full_text = _extract_text(row["url"])
        except Exception as exc:  # pragma: no cover - network inspection
            logger.warning("Failed to fetch %s: %s", row["url"], exc)
            results.append(
                {
                    "id": row.get("id"),
                    "url": row.get("url"),
                    "fetch_error": str(exc),
                }
            )
            continue

        excerpt = metadata.get("excerpt") or ""
        title = row.get("title") or metadata.get("title") or ""
        preview_text = " ".join(filter(None, [title, excerpt, full_text[:1500]]))
        full_scope = " ".join(filter(None, [title, excerpt, full_text]))

        preview_aud = sorted(_detect(preview_text, audience=True))
        preview_spc = sorted(_detect(preview_text, audience=False))
        full_aud = sorted(_detect(full_scope, audience=True))
        full_spc = sorted(_detect(full_scope, audience=False))

        late_aud = sorted(set(full_aud) - set(preview_aud))
        late_spc = sorted(set(full_spc) - set(preview_spc))

        snippets: dict[str, Any] = {
            "audiences": {},
            "species": {},
        }
        for label in late_aud:
            snippet = _find_snippet(full_text, label, AUDIENCE_TERMS)
            if snippet:
                snippets["audiences"][label] = snippet
        for label in late_spc:
            snippet = _find_snippet(full_text, label, SPECIES_TERMS)
            if snippet:
                snippets["species"][label] = snippet

        results.append(
            {
                "id": row.get("id"),
                "url": row.get("url"),
                "site_url": row.get("site_url"),
                "missing_fields": [
                    field
                    for field, values in (("audience", row_audiences), ("species", row_species))
                    if not values
                ],
                "stored_audiences": row_audiences,
                "stored_species": row_species,
                "preview_detected_audiences": preview_aud,
                "preview_detected_species": preview_spc,
                "full_detected_audiences": full_aud,
                "full_detected_species": full_spc,
                "late_audiences": late_aud,
                "late_species": late_spc,
                "content_length": len(full_text),
                "snippets": snippets,
            }
        )

    with output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze missing detection signals")
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to inspect")
    parser.add_argument(
        "--content-ids",
        nargs="*",
        help="Optional specific content UUIDs to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/missing_signal_analysis.json"),
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    results = analyze(args.limit, args.content_ids, args.output)
    print(f"Wrote analysis for {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
