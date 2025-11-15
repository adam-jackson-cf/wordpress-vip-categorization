"""Unit tests for ingestion service."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.services.ingestion import IngestionService


class TestIngestionService:
    """Tests covering incremental WordPress ingestion flows."""

    @pytest.fixture
    def service(self, mock_settings, mock_supabase_client):
        """Return an ingestion service bound to the mock DB."""
        return IngestionService(mock_settings, mock_supabase_client)

    @pytest.fixture
    def connector(self, sample_wordpress_content):
        """Create a connector mock with deterministic behavior."""
        mock_connector = Mock()
        mock_connector.test_connection.return_value = True
        mock_connector.fetch_all_content.return_value = [sample_wordpress_content]
        return mock_connector

    def test_since_timestamp_passed_to_connector(
        self,
        service: IngestionService,
        mock_supabase_client: Mock,
        connector: Mock,
    ) -> None:
        """Explicit --since timestamps should flow directly into the connector."""
        cutoff = datetime(2025, 1, 1, 12, 0, 0)

        with patch("src.services.ingestion.WordPressVIPConnector", return_value=connector):
            ingested = service.ingest_wordpress_sites(
                ["https://example.com"],
                since=cutoff,
                resume=False,
            )

        assert ingested == 1
        connector.fetch_all_content.assert_called_once()
        kwargs = connector.fetch_all_content.call_args.kwargs
        assert kwargs["modified_after"] == cutoff
        mock_supabase_client.get_latest_published_date.assert_not_called()

    def test_resume_uses_latest_published_date(
        self,
        service: IngestionService,
        mock_supabase_client: Mock,
        connector: Mock,
    ) -> None:
        """Resume mode should look up the last published date per site."""
        last_seen = datetime(2024, 12, 31, 18, 0, 0)
        mock_supabase_client.get_latest_published_date.return_value = last_seen

        with patch("src.services.ingestion.WordPressVIPConnector", return_value=connector):
            ingested = service.ingest_wordpress_sites(
                ["https://example.com"],
                max_pages=1,
                since=None,
                resume=True,
            )

        assert ingested == 1
        mock_supabase_client.get_latest_published_date.assert_called_once_with(
            "https://example.com"
        )
        kwargs = connector.fetch_all_content.call_args.kwargs
        assert kwargs["modified_after"] == last_seen
