"""Script to generate DSPy training dataset from WordPress.org/news and taxonomy.csv."""

import csv
import html
import logging
import random
import re
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_taxonomy(taxonomy_path: Path) -> list[dict[str, str]]:
    """Load taxonomy from CSV file."""
    taxonomy = []
    with open(taxonomy_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            taxonomy.append(
                {
                    "category": row["category"],
                    "description": row["description"],
                    "keywords": row["keywords"].replace(
                        ";", ", "
                    ),  # Convert semicolon to comma-separated
                }
            )
    return taxonomy


def fetch_wordpress_posts(site_url: str, num_pages: int = 20) -> list[dict]:
    """Fetch WordPress posts from the site."""
    # Ensure site_url has /news if it's wordpress.org
    if "wordpress.org" in site_url and "/news" not in site_url:
        site_url = f"{site_url.rstrip('/')}/news"

    api_base = f"{site_url.rstrip('/')}/wp-json/wp/v2/"
    all_posts = []

    for page in range(1, num_pages + 1):
        try:
            url = f"{api_base}posts"
            params = {"page": page, "per_page": 10, "status": "publish"}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            posts = response.json()

            if not isinstance(posts, list) or not posts:
                logger.info(f"No more posts at page {page}, stopping")
                break

            all_posts.extend(posts)
            logger.info(f"Fetched page {page}, got {len(posts)} posts, total: {len(all_posts)}")
            if len(posts) < 10:  # Last page
                break
        except Exception as e:
            logger.warning(f"Error fetching page {page}: {e}")
            break

    return all_posts


def format_content_summaries(posts: list[dict], max_posts: int = 10) -> str:
    """Format posts as content summaries for DSPy input."""
    summaries = []
    selected_posts = (
        random.sample(posts, min(max_posts, len(posts))) if len(posts) > max_posts else posts
    )

    for i, post in enumerate(selected_posts):
        title = post.get("title", {}).get("rendered", "Untitled")
        # Decode HTML entities in title
        title = html.unescape(title)
        link = post.get("link", "")
        content = post.get("content", {}).get("rendered", "")
        # Strip HTML tags for preview
        content_text = re.sub(r"<[^>]+>", "", content)
        content_text = html.unescape(content_text)
        preview = content_text[:200].replace("\n", " ").strip()
        summary = f"{i}. Title: {title}\n   URL: {link}\n   Preview: {preview}..."
        summaries.append(summary)

    return "\n\n".join(summaries)


def find_best_match_index(
    taxonomy_category: str,
    taxonomy_keywords: str,
    posts: list[dict],
    selected_indices: list[int],
) -> int:
    """Determine the best match index based on taxonomy and posts."""
    # Simple heuristic: look for keywords in post titles/content
    keywords_lower = taxonomy_keywords.lower()
    category_lower = taxonomy_category.lower()

    best_score = -1
    best_index = 0

    for idx, post_idx in enumerate(selected_indices):
        if post_idx >= len(posts):
            continue
        post = posts[post_idx]
        title = post.get("title", {}).get("rendered", "").lower()
        content = post.get("content", {}).get("rendered", "").lower()

        score = 0
        # Check for category match
        if category_lower in title or category_lower in content[:500]:
            score += 2
        # Check for keyword matches
        for keyword in keywords_lower.split(", "):
            if keyword in title:
                score += 1
            if keyword in content[:500]:
                score += 0.5

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


def generate_confidence_score(best_match_index: int, total_posts: int) -> float:
    """Generate a realistic confidence score."""
    # Base confidence on position and randomness
    base_confidence = 0.75
    if best_match_index == 0:
        base_confidence = 0.85  # First match is usually high confidence
    elif best_match_index < 3:
        base_confidence = 0.80
    else:
        base_confidence = 0.70

    # Add some variation
    variation = random.uniform(-0.05, 0.10)
    confidence = min(0.95, max(0.70, base_confidence + variation))
    return round(confidence, 2)


def generate_reasoning(taxonomy_category: str, taxonomy_keywords: str) -> str:
    """Generate reasoning text for the match."""
    reasons = [
        f"Strong semantic match on {taxonomy_category.lower()} keywords",
        f"Content discusses {taxonomy_category.lower()} topics matching taxonomy",
        f"High relevance to {taxonomy_category.lower()} category based on keywords",
        f"Content aligns with {taxonomy_category.lower()} taxonomy description",
    ]
    return random.choice(reasons)


def generate_dataset(
    taxonomy_path: Path,
    wordpress_site: str,
    output_path: Path,
    num_examples: int = 100,
) -> None:
    """Generate DSPy training dataset."""
    logger.info("Loading taxonomy...")
    taxonomy = load_taxonomy(taxonomy_path)
    logger.info(f"Loaded {len(taxonomy)} taxonomy categories")

    logger.info(f"Fetching WordPress posts from {wordpress_site}...")
    posts = fetch_wordpress_posts(wordpress_site, num_pages=20)
    logger.info(f"Fetched {len(posts)} WordPress posts")

    if len(posts) < 10:
        raise Exception(f"Not enough posts fetched ({len(posts)}). Need at least 10.")

    # Generate examples
    examples = []
    examples_per_category = num_examples // len(taxonomy)
    remainder = num_examples % len(taxonomy)

    logger.info(f"Generating {num_examples} examples...")

    for cat_idx, tax in enumerate(taxonomy):
        num_for_category = examples_per_category + (1 if cat_idx < remainder else 0)

        for _ in range(num_for_category):
            # Select random posts for this example (5-10 posts)
            num_posts = random.randint(5, min(10, len(posts)))
            selected_indices = random.sample(range(len(posts)), num_posts)
            selected_posts = [posts[i] for i in selected_indices]

            # Format content summaries
            content_summaries = format_content_summaries(selected_posts, max_posts=num_posts)

            # Find best match
            best_match_index = find_best_match_index(
                tax["category"],
                tax["keywords"],
                posts,
                selected_indices,
            )

            # Generate confidence and reasoning
            confidence = generate_confidence_score(best_match_index, num_posts)
            reasoning = generate_reasoning(tax["category"], tax["keywords"])

            example = {
                "taxonomy_category": tax["category"],
                "taxonomy_description": tax["description"],
                "taxonomy_keywords": tax["keywords"],
                "content_summaries": content_summaries,
                "best_match_index": str(best_match_index),
                "confidence": str(confidence),
                "reasoning": reasoning,
            }
            examples.append(example)

    logger.info(f"Generated {len(examples)} examples")

    # Write to CSV
    logger.info(f"Writing dataset to {output_path}...")
    fieldnames = [
        "taxonomy_category",
        "taxonomy_description",
        "taxonomy_keywords",
        "content_summaries",
        "best_match_index",
        "confidence",
        "reasoning",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    logger.info(f"✓ Dataset written to {output_path} with {len(examples)} examples")


if __name__ == "__main__":
    import sys

    taxonomy_path = Path("data/taxonomy.csv")
    output_path = Path("data/dspy_training_dataset.csv")
    wordpress_site = "https://wordpress.org/news"

    try:
        generate_dataset(taxonomy_path, wordpress_site, output_path, num_examples=100)
        logger.info("✓ Dataset generation complete!")
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        sys.exit(1)
