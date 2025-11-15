"""Unit tests for WordPress VIP connector."""

from datetime import datetime
from unittest.mock import Mock, patch

import requests

from src.connectors.wordpress_vip import WordPressVIPConnector
from src.models import WordPressContent


class TestWordPressVIPConnector:
    """Tests for WordPress VIP connector."""

    def test_init(self) -> None:
        """Test connector initialization."""
        connector = WordPressVIPConnector("https://example.com")
        assert connector.site_url == "https://example.com"
        assert "/wp-json/wp/v2/" in connector.api_base

    def test_init_with_auth(self) -> None:
        """Test connector initialization with auth token."""
        connector = WordPressVIPConnector("https://example.com", auth_token="test-token")
        assert "Authorization" in connector.session.headers
        assert connector.session.headers["Authorization"] == "Bearer test-token"

    @patch("src.connectors.wordpress_vip.requests.Session.get")
    def test_get_posts_success(self, mock_get: Mock) -> None:
        """Test fetching posts successfully."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": 1,
                "link": "https://example.com/post-1",
                "title": {"rendered": "Test Post"},
                "content": {"rendered": "<p>Test content</p>"},
                "date_gmt": "2024-01-01T12:00:00",
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        connector = WordPressVIPConnector("https://example.com")
        posts = connector.get_posts(page=1, per_page=10)

        assert len(posts) == 1
        assert posts[0]["id"] == 1
        assert posts[0]["title"]["rendered"] == "Test Post"

    def test_extract_text_content(self) -> None:
        """Test HTML text extraction."""
        connector = WordPressVIPConnector("https://example.com")

        html = "<p>Hello <strong>world</strong>!</p>"
        text = connector._extract_text_content(html)

        assert text == "Hello world!"
        assert "<p>" not in text
        assert "<strong>" not in text

    def test_parse_wordpress_item(self) -> None:
        """Test parsing WordPress API response."""
        connector = WordPressVIPConnector("https://example.com")

        item = {
            "id": 123,
            "link": "https://example.com/test-post",
            "title": {"rendered": "Test Post"},
            "content": {"rendered": "<p>Test content</p>"},
            "excerpt": {"rendered": "<p>Excerpt</p>"},
            "date_gmt": "2024-01-01T12:00:00",
            "slug": "test-post",
            "author": 1,
            "categories": [5, 10],
            "tags": [3],
        }

        content = connector._parse_wordpress_item(item, "post")

        assert isinstance(content, WordPressContent)
        assert content.title == "Test Post"
        assert content.content == "Test content"
        assert str(content.url) == "https://example.com/test-post"
        assert content.metadata["type"] == "post"
        assert content.metadata["wp_id"] == 123

    @patch("src.connectors.wordpress_vip.requests.Session.get")
    def test_test_connection_success(self, mock_get: Mock) -> None:
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        connector = WordPressVIPConnector("https://example.com")
        result = connector.test_connection()

        assert result is True

    @patch("src.connectors.wordpress_vip.requests.Session.get")
    def test_test_connection_failure(self, mock_get: Mock) -> None:
        """Test failed connection test."""
        mock_get.side_effect = requests.exceptions.RequestException()

        connector = WordPressVIPConnector("https://example.com")
        result = connector.test_connection()

        assert result is False

    def test_fetch_all_posts_respects_modified_after(
        self,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Ensure fetch_all_posts forwards modified_after to get_posts."""

        connector = WordPressVIPConnector("https://example.com")
        cutoff = datetime(2024, 1, 1)

        with (
            patch.object(
                connector,
                "get_posts",
                side_effect=[
                    [
                        {
                            "id": 1,
                            "link": "https://example.com/post",
                            "title": {"rendered": "Title"},
                            "content": {"rendered": "<p>Body</p>"},
                        }
                    ],
                    [],
                ],
            ) as mock_get_posts,
            patch.object(
                connector,
                "_parse_wordpress_item",
                return_value=sample_wordpress_content,
            ),
        ):
            list(
                connector.fetch_all_posts(
                    show_progress=False,
                    modified_after=cutoff,
                )
            )

        first_call_kwargs = mock_get_posts.call_args_list[0].kwargs
        assert first_call_kwargs["page"] == 1
        assert first_call_kwargs["modified_after"] == cutoff
