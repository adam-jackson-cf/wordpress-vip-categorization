"""Inspect metadata available from a WordPress VIP site."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

import requests

from src.config import Settings
from src.connectors.wordpress_vip import WordPressVIPConnector


def _match_site_token(settings: Settings, site: str) -> str:
    normalized = site.rstrip("/")
    for base, token in settings.get_wordpress_site_tokens():
        if normalized.startswith(base.rstrip("/")):
            return token
    raise ValueError(
        "No token found for the provided site. Use --token to override or update WORDPRESS_VIP_SITE_TOKENS."
    )


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _extract_inline_json(html: str, var_name: str) -> str | None:
    pattern = rf"var\s+{var_name}\s*=\s*(\{{.*?\}});"
    match = re.search(pattern, html, re.S)
    if not match:
        return None
    payload = match.group(1)
    return payload.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect metadata for a WordPress VIP site")
    parser.add_argument("--site", required=True, help="Base site URL, e.g. https://www.msd-animal-health.es")
    parser.add_argument(
        "--token",
        help="Optional token override. Defaults to lookup in WORDPRESS_VIP_SITE_TOKENS",
    )
    parser.add_argument(
        "--content-type",
        choices=["posts", "pages"],
        default="pages",
        help="WordPress REST resource to inspect",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of items to fetch for inspection",
    )
    args = parser.parse_args()

    settings = Settings()
    token = args.token or _match_site_token(settings, args.site)

    connector = WordPressVIPConnector(site_url=args.site, auth_token=token)
    fetcher = connector.get_pages if args.content_type == "pages" else connector.get_posts

    items, _ = fetcher(page=1, per_page=args.limit)
    if not items:
        print("No items returned from the WordPress API. Try lowering limit or checking credentials.")
        return

    print(f"Fetched {len(items)} {args.content_type} from {args.site}\n")

    for idx, item in enumerate(items, start=1):
        link = item.get("link")
        print(f"=== Item {idx}: ID {item.get('id')} ===")
        print(f"Link: {link}")
        print(f"Available keys: {sorted(item.keys())}")
        tax_fields = {
            key: item.get(key)
            for key in ("categories", "tags", "meta", "acf", "page_context", "yoast_head_json")
            if key in item
        }
        if tax_fields:
            print("Selected taxonomy/meta fields:")
            print(_pretty(tax_fields))

        parsed = connector._parse_wordpress_item(item,  # type: ignore[attr-defined]
                                                 "post" if args.content_type == "posts" else "page")
        print("Parsed WordPressContent metadata:")
        print(_pretty(parsed.metadata))

        if link:
            try:
                resp = requests.get(link, timeout=15)
                resp.raise_for_status()
                html = resp.text
                analytics = _extract_inline_json(html, "mahAnalytics")
                if analytics:
                    print("mahAnalytics snippet:")
                    print(analytics[:1000])
                page_context = _extract_inline_json(html, "pageContext")
                if page_context:
                    print("pageContext snippet:")
                    print(page_context[:1000])
            except Exception as exc:  # pragma: no cover - network
                print(f"Failed to fetch HTML for inspection: {exc}")

        print()


if __name__ == "__main__":
    main()
