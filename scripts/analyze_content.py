#!/usr/bin/env python3
"""Analyze downloaded HTML content for semantic matching insights."""

import json
import re
from pathlib import Path
from html import unescape
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup4 not available, using basic text extraction")
    BeautifulSoup = None


def extract_text_basic(html: str) -> dict[str, Any]:
    """Extract text using basic regex when BeautifulSoup unavailable."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group(1)) if title_match else ""

    # Extract meta description
    desc_match = re.search(
        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
        html,
        re.IGNORECASE,
    )
    description = unescape(desc_match.group(1)) if desc_match else ""

    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    text = unescape(text)

    return {
        "title": title.strip(),
        "description": description.strip(),
        "body_text": text[:5000],  # First 5000 chars
        "total_length": len(text),
    }


def extract_text_soup(html: str) -> dict[str, Any]:
    """Extract text using BeautifulSoup for better structure."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and navigation
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Extract structured content
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else ""

    # Try meta description
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = desc_tag.get("content", "").strip() if desc_tag else ""

    # Try OG description as fallback
    if not description:
        og_desc = soup.find("meta", property="og:description")
        description = og_desc.get("content", "").strip() if og_desc else ""

    # Try to find main content area
    main_content = None
    for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
        if isinstance(selector, str) and selector.startswith("."):
            main_content = soup.find(class_=selector[1:])
        elif isinstance(selector, str) and selector.startswith("#"):
            main_content = soup.find(id=selector[1:])
        elif "[" in selector:
            continue  # Skip attribute selectors for now
        else:
            main_content = soup.find(selector)
        if main_content:
            break

    # If no main content found, use body
    if not main_content:
        main_content = soup.find("body")

    body_text = ""
    headings = []
    if main_content:
        # Extract headings
        for h in main_content.find_all(["h1", "h2", "h3"]):
            headings.append(h.get_text().strip())

        # Extract all text
        body_text = main_content.get_text(separator=" ", strip=True)

    return {
        "title": title,
        "description": description,
        "headings": headings[:10],  # Top 10 headings
        "body_text": body_text[:5000],  # First 5000 chars
        "total_length": len(body_text),
    }


def analyze_page(page_num: int, content_dir: Path) -> dict[str, Any]:
    """Analyze a single page."""
    meta_file = content_dir / f"page_{page_num}_meta.json"
    html_file = content_dir / f"page_{page_num}.html"

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Extract text
    if BeautifulSoup:
        extracted = extract_text_soup(html_content)
    else:
        extracted = extract_text_basic(html_content)

    return {
        "page_num": page_num,
        "metadata": metadata,
        "extracted": extracted,
    }


def main() -> None:
    """Analyze all downloaded pages."""
    content_dir = Path("data/examples/content")
    output_file = Path("data/examples/content_analysis.json")

    results = []
    for page_num in range(1, 51):
        try:
            result = analyze_page(page_num, content_dir)
            results.append(result)
            print(f"Analyzed page {page_num}/50")
        except Exception as e:
            print(f"Error analyzing page {page_num}: {e}")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved analysis to {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    total_chars = sum(r["extracted"]["total_length"] for r in results)
    avg_chars = total_chars / len(results) if results else 0
    print(f"Total pages analyzed: {len(results)}")
    print(f"Average content length: {avg_chars:.0f} characters")

    # Category distribution
    categories = {}
    for r in results:
        cat = r["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
