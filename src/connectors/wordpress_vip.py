"""WordPress VIP API connector for content ingestion."""

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any, cast
from urllib.parse import urljoin

import requests
from pydantic import HttpUrl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.models import WordPressContent

logger = logging.getLogger(__name__)


class WordPressVIPConnector:
    """Connector for WordPress VIP REST API.

    This connector uses the standard WordPress REST API which is available
    on all WordPress VIP sites by default.
    """

    def __init__(
        self,
        site_url: str,
        auth_token: str | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize WordPress VIP connector.

        Args:
            site_url: Base URL of the WordPress site.
            auth_token: Optional authentication token for private sites.
            timeout: Request timeout in seconds.
        """
        self.site_url = site_url.rstrip("/")
        self.api_base = urljoin(self.site_url, "/wp-json/wp/v2/")
        self.timeout = timeout
        self.session = requests.Session()

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

        self.session.headers.update(
            {
                "User-Agent": "WordPress-VIP-Categorization/0.1.0",
                "Accept": "application/json",
            }
        )

        logger.info(f"Initialized WordPress connector for {self.site_url}")

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        include_headers: bool = False,
    ) -> Any:
        """Make HTTP request to WordPress API with retry logic.

        Args:
            endpoint: API endpoint (e.g., 'posts', 'pages').
            params: Query parameters.

        Returns:
            JSON response data.

        Raises:
            requests.exceptions.RequestException: If request fails after retries.
        """
        url = urljoin(self.api_base, endpoint)
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if include_headers:
            return data, response.headers
        return data

    def get_posts(
        self,
        page: int = 1,
        per_page: int = 100,
        status: str = "publish",
        modified_after: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get posts from WordPress site.

        Args:
            page: Page number (1-indexed).
            per_page: Number of posts per page (max 100).
            status: Post status filter (publish, draft, etc.).

        Returns:
            List of post data dictionaries.
        """
        params = {
            "page": page,
            "per_page": min(per_page, 100),
            "status": status,
            "_embed": True,  # Include embedded data like featured images
        }

        if modified_after:
            params["after"] = modified_after.isoformat()

        try:
            data, headers = self._make_request("posts", params, include_headers=True)
            total_pages = int(headers.get("X-WP-TotalPages", "0") or 0)
            if isinstance(data, list):
                return data, total_pages
            return [], total_pages
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Page beyond available content
                return [], 0
            raise

    def get_pages(
        self,
        page: int = 1,
        per_page: int = 100,
        status: str = "publish",
        modified_after: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get pages from WordPress site.

        Args:
            page: Page number (1-indexed).
            per_page: Number of pages per page (max 100).
            status: Page status filter.

        Returns:
            List of page data dictionaries.
        """
        params = {
            "page": page,
            "per_page": min(per_page, 100),
            "status": status,
        }

        if modified_after:
            params["after"] = modified_after.isoformat()

        try:
            data, headers = self._make_request("pages", params, include_headers=True)
            total_pages = int(headers.get("X-WP-TotalPages", "0") or 0)
            if isinstance(data, list):
                return data, total_pages
            return [], total_pages
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                return [], 0
            raise

    def _extract_text_content(self, html_content: str) -> str:
        """Extract plain text from HTML content.

        Args:
            html_content: HTML content string.

        Returns:
            Plain text content.
        """
        # Simple HTML tag removal - in production, use BeautifulSoup or similar
        import re

        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode HTML entities
        import html

        text = html.unescape(text)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_wordpress_item(
        self, item: dict[str, Any], content_type: str = "post"
    ) -> WordPressContent:
        """Parse WordPress API response item into WordPressContent model.

        Args:
            item: WordPress API item dictionary.
            content_type: Type of content (post, page, etc.).

        Returns:
            Parsed WordPressContent object.
        """
        # Extract content
        content_html = item.get("content", {}).get("rendered", "")
        content_text = self._extract_text_content(content_html)

        # Extract title
        title = item.get("title", {}).get("rendered", "Untitled")
        title_text = self._extract_text_content(title)

        # Parse published date
        published_date = None
        date_str = item.get("date_gmt")
        if date_str:
            try:
                published_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Extract metadata
        metadata = {
            "type": content_type,
            "wp_id": item.get("id"),
            "slug": item.get("slug"),
            "excerpt": self._extract_text_content(item.get("excerpt", {}).get("rendered", "")),
            "author": item.get("author"),
            "categories": item.get("categories", []),
            "tags": item.get("tags", []),
        }

        return WordPressContent(
            url=cast(HttpUrl, item.get("link", "")),
            title=title_text,
            content=content_text,
            site_url=cast(HttpUrl, self.site_url),
            published_date=published_date,
            metadata=metadata,
        )

    def fetch_all_posts(
        self,
        max_pages: int | None = None,
        show_progress: bool = True,
        modified_after: datetime | None = None,
    ) -> Iterator[WordPressContent]:
        """Fetch all posts from the WordPress site.

        Args:
            max_pages: Maximum number of pages to fetch (None for all).
            show_progress: Show progress bar.

        Yields:
            WordPressContent objects.
        """
        page = 1
        total_fetched = 0

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(desc=f"Fetching posts from {self.site_url}", unit="posts")

        while True:
            if max_pages and page > max_pages:
                break

            posts, total_pages = self.get_posts(
                page=page, per_page=100, modified_after=modified_after
            )
            if not posts:
                break

            for post_data in posts:
                try:
                    content = self._parse_wordpress_item(post_data, "post")
                    yield content
                    total_fetched += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                except Exception as e:
                    logger.error(f"Error parsing post {post_data.get('id')}: {e}")
                    continue

            if total_pages and page >= total_pages:
                break

            page += 1

        if progress_bar is not None:
            progress_bar.close()

        logger.info(f"Fetched {total_fetched} posts from {self.site_url}")

    def fetch_all_pages(
        self,
        max_pages: int | None = None,
        show_progress: bool = True,
        modified_after: datetime | None = None,
    ) -> Iterator[WordPressContent]:
        """Fetch all pages from the WordPress site.

        Args:
            max_pages: Maximum number of API pages to fetch (None for all).
            show_progress: Show progress bar.

        Yields:
            WordPressContent objects.
        """
        page = 1
        total_fetched = 0

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(desc=f"Fetching pages from {self.site_url}", unit="pages")

        while True:
            if max_pages and page > max_pages:
                break

            pages, total_pages = self.get_pages(
                page=page, per_page=100, modified_after=modified_after
            )
            if not pages:
                break

            for page_data in pages:
                try:
                    content = self._parse_wordpress_item(page_data, "page")
                    yield content
                    total_fetched += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                except Exception as e:
                    logger.error(f"Error parsing page {page_data.get('id')}: {e}")
                    continue

            if total_pages and page >= total_pages:
                break

            page += 1

        if progress_bar is not None:
            progress_bar.close()

        logger.info(f"Fetched {total_fetched} pages from {self.site_url}")

    def fetch_all_content(
        self,
        max_pages: int | None = None,
        show_progress: bool = True,
        modified_after: datetime | None = None,
    ) -> Iterator[WordPressContent]:
        """Fetch all content (posts and pages) from the WordPress site.

        Args:
            max_pages: Maximum number of API pages to fetch per content type.
            show_progress: Show progress bar.

        Yields:
            WordPressContent objects.
        """
        # Fetch posts
        yield from self.fetch_all_posts(
            max_pages=max_pages,
            show_progress=show_progress,
            modified_after=modified_after,
        )

        # Fetch pages
        yield from self.fetch_all_pages(
            max_pages=max_pages,
            show_progress=show_progress,
            modified_after=modified_after,
        )

    def test_connection(self) -> bool:
        """Test connection to WordPress site.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Try to fetch site info
            url = urljoin(self.site_url, "/wp-json/")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Successfully connected to {self.site_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.site_url}: {e}")
            return False
