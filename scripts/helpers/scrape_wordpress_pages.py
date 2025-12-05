#!/usr/bin/env python3
"""
Script to scrape WordPress pages and extract clean text content.
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def sanitize_filename(url: str) -> str:
    """
    Create a safe filename from URL.
    Format: domain_path.txt
    """
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    # Get path and sanitize it
    path = parsed.path.strip("/")
    if not path or path == "":
        path = "home"

    # Handle query parameters
    if parsed.query:
        # Extract page_id or p parameter if exists
        if "page_id=" in parsed.query:
            page_id = re.search(r"page_id=(\d+)", parsed.query)
            if page_id:
                path = f"page_{page_id.group(1)}"
        elif "p=" in parsed.query:
            p_id = re.search(r"p=(\d+)", parsed.query)
            if p_id:
                path = f"post_{p_id.group(1)}"
        else:
            path = parsed.query.replace("=", "_").replace("&", "_")

    # Sanitize path: replace slashes and special chars with underscores
    path = re.sub(r"[/\-\s]+", "_", path)
    path = re.sub(r"[^\w_]", "", path)

    # Limit length
    if len(path) > 100:
        path = path[:100]

    return f"{domain}_{path}.txt"


def extract_content(html: str, url: str) -> dict[str, str]:
    """
    Extract meaningful content from HTML.
    Returns dict with title, headings, and body text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup(
        ["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript", "form"]
    ):
        element.decompose()

    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string.strip() if soup.title.string else ""
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Extract all text content
    # Try to find main content area first
    main_content = None
    for selector in [
        "main",
        "article",
        '[role="main"]',
        ".content",
        "#content",
        ".post-content",
        ".entry-content",
        ".page-content",
    ]:
        main_content = soup.select_one(selector)
        if main_content:
            break

    if not main_content:
        main_content = soup.find("body")

    if not main_content:
        main_content = soup

    # Extract headings
    headings = []
    for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text(strip=True)
        if text:
            headings.append(text)

    # Extract paragraphs and text
    paragraphs = []
    for p in main_content.find_all(["p", "li", "div"]):
        text = p.get_text(strip=True)
        if text and len(text) > 20:  # Filter out very short text
            paragraphs.append(text)

    # Remove duplicates while preserving order
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            unique_paragraphs.append(p)

    return {"title": title, "url": url, "headings": headings, "body": unique_paragraphs}


def format_content(content: dict[str, str]) -> str:
    """
    Format extracted content as readable text.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"URL: {content['url']}")
    lines.append("=" * 80)
    lines.append("")

    if content["title"]:
        lines.append(f"TITLE: {content['title']}")
        lines.append("")

    if content["headings"]:
        lines.append("HEADINGS:")
        for heading in content["headings"][:20]:  # Limit headings
            lines.append(f"  - {heading}")
        lines.append("")

    lines.append("CONTENT:")
    lines.append("-" * 80)
    for para in content["body"][:100]:  # Limit paragraphs
        lines.append(para)
        lines.append("")

    return "\n".join(lines)


def scrape_url(url: str, output_dir: Path, timeout: int = 30) -> tuple[bool, str, int]:
    """
    Scrape a single URL and save content.
    Returns (success, message, file_size)
    """
    try:
        # Make request with proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        }

        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Extract content
        content = extract_content(response.text, url)

        # Check if we got meaningful content
        if not content["body"] or len(content["body"]) < 3:
            return False, "No meaningful content extracted", 0

        # Format and save
        formatted = format_content(content)
        filename = sanitize_filename(url)
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted)

        file_size = filepath.stat().st_size
        return True, filename, file_size

    except requests.Timeout:
        return False, f"Timeout after {timeout}s", 0
    except requests.RequestException as e:
        return False, f"Request error: {str(e)}", 0
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path("/Users/adamjackson/Projects/wordpress-vip-categorization")
    urls_file = project_root / "data" / "sample_urls.txt"
    output_dir = project_root / "data" / "examples"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read URLs
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip() and line.strip().startswith("http")]

    print(f"Found {len(urls)} URLs to process")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    # Process each URL
    results = {"success": [], "failed": []}

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url}")

        success, message, size = scrape_url(url, output_dir)

        if success:
            results["success"].append({"url": url, "filename": message, "size": size})
            print(f"  ✓ Saved: {message} ({size:,} bytes)")
        else:
            results["failed"].append({"url": url, "reason": message})
            print(f"  ✗ Failed: {message}")

        # Be respectful to servers
        time.sleep(1)
        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful downloads: {len(results['success'])}")
    print(f"Failed downloads: {len(results['failed'])}")
    print()

    if results["success"]:
        print("SUCCESSFUL DOWNLOADS:")
        print("-" * 80)
        total_size = 0
        for item in results["success"]:
            print(f"  {item['filename']:<60} {item['size']:>10,} bytes")
            total_size += item["size"]
        print(f"\nTotal size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print()

    if results["failed"]:
        print("FAILED DOWNLOADS:")
        print("-" * 80)
        for item in results["failed"]:
            print(f"  URL: {item['url']}")
            print(f"  Reason: {item['reason']}")
            print()

    # Save detailed report
    report_file = output_dir / "_scraping_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("WordPress Page Scraping Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total URLs processed: {len(urls)}\n")
        f.write(f"Successful: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n\n")

        if results["success"]:
            f.write("SUCCESSFUL DOWNLOADS:\n")
            f.write("-" * 80 + "\n")
            for item in results["success"]:
                f.write(f"{item['filename']}: {item['size']:,} bytes\n")
                f.write(f"  Source: {item['url']}\n\n")

        if results["failed"]:
            f.write("\nFAILED DOWNLOADS:\n")
            f.write("-" * 80 + "\n")
            for item in results["failed"]:
                f.write(f"URL: {item['url']}\n")
                f.write(f"Reason: {item['reason']}\n\n")

    print(f"Detailed report saved to: {report_file}")

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
