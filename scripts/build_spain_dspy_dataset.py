"""Build a DSPy training dataset using Spain taxonomy and live page content.

Inputs:
- data/Spain_New.csv (canonical taxonomy)
- data/Spain_Pages_to_redirect.csv (live pages; we sample per site)

Output:
- data/dspy_training_dataset.csv (overwrites existing)

Notes:
- Uses stdlib urllib to avoid extra deps.
- Fetches a capped subset of pages per site to keep runtime reasonable.
"""

from __future__ import annotations

import csv
import html
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA_DIR = Path("data")
TAXONOMY_PATH = DATA_DIR / "Spain_New.csv"
PAGES_PATH = DATA_DIR / "Spain_Pages_to_redirect.csv"
OUTPUT_PATH = DATA_DIR / "dspy_training_dataset.csv"

# Limit fetches to keep runtime predictable
MAX_PAGES_PER_SITE = 5
MAX_CANDIDATES_PER_TAXONOMY = 8
MIN_CANDIDATES_PER_TAXONOMY = 6


def load_taxonomy(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_pages(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def strip_html(html_text: str) -> str:
    """Lightweight HTML -> text conversion."""
    # Remove scripts/styles
    html_text = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html_text, flags=re.I | re.S)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html_text)
    # Unescape entities
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def extract_title(html_text: str) -> str | None:
    m = re.search(r"<title>(.*?)</title>", html_text, flags=re.I | re.S)
    if m:
        return html.unescape(m.group(1)).strip()
    return None


def fetch_summary(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """Fetch URL and return (title, preview)."""
    headers = {"User-Agent": "spain-dspy-dataset/1.0"}
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                # Try declared encoding
                charset = resp.headers.get_content_charset() or "utf-8"
                html_text = raw.decode(charset, errors="ignore")
            except Exception:
                html_text = raw.decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:
        return "", f"[fetch-error] {exc}"
    title = extract_title(html_text) or ""
    text = strip_html(html_text)
    preview = text[:400]
    return title, preview


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-ZÀ-ÿ0-9]+", text.lower()) if t]


def score_candidate(topics: Sequence[str], summary: str, title: str, url: str) -> int:
    tokens = tokenize(summary + " " + title + " " + url)
    topic_tokens = [t.lower() for t in topics]
    return sum(tokens.count(t) for t in topic_tokens)


def build_content_summary(index: int, title: str, url: str, preview: str) -> str:
    safe_title = title or urlparse(url).path or url
    return f"{index}. Title: {safe_title}\\n   URL: {url}\\n   Preview: {preview[:200]}..."


def main() -> None:
    taxonomy_rows = load_taxonomy(TAXONOMY_PATH)
    pages = load_pages(PAGES_PATH)

    # Sample a capped number of pages per site to reduce fetch volume
    pages_by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pages:
        pages_by_site[row["Site"]].append(row)
    sampled_pages: list[dict[str, str]] = []
    for site, rows in pages_by_site.items():
        sampled = rows[:MAX_PAGES_PER_SITE]
        sampled_pages.extend(sampled)

    # Fetch summaries once per sampled page
    page_cache: dict[str, dict[str, str]] = {}
    for row in sampled_pages:
        url = row["Page_URL"]
        title, preview = fetch_summary(url)
        page_cache[url] = {
            "title": title,
            "preview": preview,
            "site": row["Site"],
            "page_type": row.get("Page_Type", ""),
        }

    examples: list[dict[str, str]] = []
    random.seed(42)

    for tax in taxonomy_rows:
        topics_raw = tax.get("Key_Topics", "")
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]
        summary = tax.get("Semantic_Summary", "")
        content_type = tax.get("Content_Type", "")

        # Score candidates
        scored: list[tuple[int, str]] = []
        for url, meta in page_cache.items():
            score = score_candidate(topics, meta["preview"], meta["title"], url)
            # small boost if content type word appears in preview/title
            if content_type and content_type.lower() in meta["preview"].lower():
                score += 1
            scored.append((score, url))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            continue

        # Select candidate list
        candidate_urls = [u for _, u in scored[:MAX_CANDIDATES_PER_TAXONOMY]]
        if len(candidate_urls) < MIN_CANDIDATES_PER_TAXONOMY:
            candidate_urls = [u for _, u in scored[:MIN_CANDIDATES_PER_TAXONOMY]]

        # Determine best match index
        best_match_url = candidate_urls[0]
        content_summaries_parts: list[str] = []
        best_index = 0
        for idx, url in enumerate(candidate_urls):
            meta = page_cache[url]
            summary_str = build_content_summary(idx, meta["title"], url, meta["preview"])
            content_summaries_parts.append(summary_str)
            if url == best_match_url:
                best_index = idx

        content_summaries = "\\n\\n".join(content_summaries_parts)

        # Heuristic rubric
        max_score = scored[0][0] or 1
        topic_alignment = min(1.0, max_score / max(1, len(topics))) if topics else 0.6
        intent_fit = 0.8 if topic_alignment >= 0.6 else 0.6
        entity_overlap = topic_alignment
        temporal_relevance = 0.3  # static; real recency unavailable
        decision = "accept" if topic_alignment >= 0.6 else "reject"
        confidence = round(0.75 + (topic_alignment * 0.2), 2)
        reasoning = (
            f"Chosen because it overlaps key topics ({topics_raw}) "
            f"and aligns with content type {content_type}."
        )

        examples.append(
            {
                "taxonomy_content_type": content_type,
                "taxonomy_summary": summary,
                "taxonomy_topics": topics_raw,
                "content_summaries": content_summaries,
                "best_match_index": str(best_index),
                "topic_alignment": f"{topic_alignment:.2f}",
                "intent_fit": f"{intent_fit:.2f}",
                "entity_overlap": f"{entity_overlap:.2f}",
                "temporal_relevance": f"{temporal_relevance:.2f}",
                "decision": decision,
                "confidence": f"{confidence:.2f}",
                "reasoning": reasoning,
            }
        )

    fieldnames = [
        "taxonomy_content_type",
        "taxonomy_summary",
        "taxonomy_topics",
        "content_summaries",
        "best_match_index",
        "topic_alignment",
        "intent_fit",
        "entity_overlap",
        "temporal_relevance",
        "decision",
        "confidence",
        "reasoning",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
