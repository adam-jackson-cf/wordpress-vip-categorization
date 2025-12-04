"""Build a DSPy training dataset from taxonomy rows plus sampled live pages."""

from __future__ import annotations

import argparse
import csv
import html
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxonomy-file",
        required=True,
        type=Path,
        help="Taxonomy CSV with Destination_URL, Content_Type, Semantic_Summary, Key_Topics",
    )
    parser.add_argument(
        "--pages-file",
        required=True,
        type=Path,
        help="CSV containing sampled pages (columns: Site, Page_URL, Page_Type)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/dspy_training_dataset.csv"),
        help="Output CSV path (default: data/dspy_training_dataset.csv)",
    )
    parser.add_argument(
        "--max-pages-per-site",
        type=int,
        default=5,
        help="Limit the number of pages fetched per site (default: 5)",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=6,
        help="Minimum candidate rows per taxonomy entry.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum candidate rows per taxonomy entry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def strip_html(html_text: str) -> str:
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html_text, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_title(html_text: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html_text, flags=re.I | re.S)
    if match:
        return html.unescape(match.group(1)).strip()
    return ""


def fetch_summary(url: str, timeout: float = 10.0) -> tuple[str, str]:
    headers = {"User-Agent": "dspy-dataset/1.0"}
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 (controlled usage)
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            html_text = raw.decode(charset, errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:
        return "", f"[fetch-error] {exc}"
    title = extract_title(html_text)
    preview = strip_html(html_text)[:400]
    return title, preview


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-ZÀ-ÿ0-9]+", text.lower()) if token]


def score_candidate(topics: Sequence[str], summary: str, title: str, url: str, content_type: str) -> int:
    tokens = tokenize(" ".join([summary, title, url]))
    topic_tokens = [topic.lower() for topic in topics]
    score = sum(tokens.count(topic) for topic in topic_tokens)
    if content_type and content_type.lower() in summary.lower():
        score += 1
    return score


def build_content_summary(index: int, title: str, url: str, preview: str) -> str:
    safe_title = title or urlparse(url).path or url
    return f"{index}. Title: {safe_title}\n   URL: {url}\n   Preview: {preview[:200]}..."


def main() -> None:
    args = parse_args()
    taxonomy_rows = load_csv(args.taxonomy_file)
    pages = load_csv(args.pages_file)

    random.seed(args.seed)

    pages_by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for page in pages:
        pages_by_site[page["Site"]].append(page)

    sampled_pages: list[dict[str, str]] = []
    for site, rows in pages_by_site.items():
        sampled_pages.extend(rows[: args.max_pages_per_site])

    page_cache: dict[str, dict[str, str]] = {}
    for row in sampled_pages:
        page_url = row["Page_URL"]
        title, preview = fetch_summary(page_url)
        page_cache[page_url] = {
            "title": title,
            "preview": preview,
            "site": row.get("Site", ""),
            "page_type": row.get("Page_Type", ""),
        }

    examples: list[dict[str, str]] = []

    for taxonomy in taxonomy_rows:
        topics_raw = taxonomy.get("Key_Topics", "")
        topics = [topic.strip() for topic in topics_raw.split(",") if topic.strip()]
        summary = taxonomy.get("Semantic_Summary", "")
        content_type = taxonomy.get("Content_Type", "")

        scored: list[tuple[int, str]] = []
        for url, metadata in page_cache.items():
            score = score_candidate(topics, metadata["preview"], metadata["title"], url, content_type)
            scored.append((score, url))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            continue

        candidate_urls = [url for _, url in scored[: args.max_candidates]]
        if len(candidate_urls) < args.min_candidates:
            candidate_urls = [url for _, url in scored[: args.min_candidates]]

        content_summaries: list[str] = []
        for index, url in enumerate(candidate_urls, start=1):
            meta = page_cache[url]
            content_summaries.append(
                build_content_summary(index, meta["title"], url, meta["preview"])
            )

        examples.append(
            {
                "taxonomy_content_type": content_type,
                "taxonomy_summary": summary,
                "taxonomy_topics": ", ".join(topics),
                "content_summaries": "\n\n".join(content_summaries),
            }
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "taxonomy_content_type",
                "taxonomy_summary",
                "taxonomy_topics",
                "content_summaries",
            ],
        )
        writer.writeheader()
        writer.writerows(examples)


+if __name__ == "__main__":
+    main()
