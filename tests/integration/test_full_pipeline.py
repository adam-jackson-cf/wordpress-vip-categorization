"""Integration tests for full pipeline using public WordPress API.

These tests use the WordPress.org news blog as a public test endpoint
to verify the end-to-end functionality of the system.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import Settings
from src.connectors.wordpress_vip import WordPressVIPConnector
from src.exporters.csv_exporter import CSVExporter
from src.models import TaxonomyPage
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def public_wordpress_site(self) -> str:
        """Return a public WordPress site for testing."""
        # WordPress.org news blog - public and reliable
        return "https://wordpress.org/news"

    @pytest.fixture
    def mock_db_client(self, mocker) -> Mock:  # type: ignore[misc]
        """Create a mock database client that stores data in memory."""
        mock_db = mocker.Mock()

        # In-memory storage
        mock_db._content = []
        mock_db._taxonomy = []
        mock_db._matchings = []

        # Implement methods with in-memory storage
        def upsert_content(content):
            # Check if exists
            for i, c in enumerate(mock_db._content):
                if c.url == content.url:
                    mock_db._content[i] = content
                    return content
            mock_db._content.append(content)
            return content

        def get_all_content(limit=None):
            if limit:
                return mock_db._content[:limit]
            return mock_db._content[:]

        def upsert_taxonomy(taxonomy):
            for i, t in enumerate(mock_db._taxonomy):
                if t.url == taxonomy.url:
                    mock_db._taxonomy[i] = taxonomy
                    return taxonomy
            mock_db._taxonomy.append(taxonomy)
            return taxonomy

        def get_all_taxonomy():
            return mock_db._taxonomy[:]

        def get_content_by_id(content_id):
            for c in mock_db._content:
                if c.id == content_id:
                    return c
            return None

        def upsert_matching(matching):
            for i, m in enumerate(mock_db._matchings):
                if m.taxonomy_id == matching.taxonomy_id:
                    mock_db._matchings[i] = matching
                    return matching
            mock_db._matchings.append(matching)
            return matching

        def get_best_match_for_taxonomy(taxonomy_id, min_score=0.0):
            for m in mock_db._matchings:
                if m.taxonomy_id == taxonomy_id and m.similarity_score >= min_score:
                    return m
            return None

        def get_categorizations_by_content(content_id):
            return []

        def get_all_matchings():
            return mock_db._matchings[:]

        mock_db.upsert_content = upsert_content
        mock_db.get_all_content = get_all_content
        mock_db.upsert_taxonomy = upsert_taxonomy
        mock_db.get_all_taxonomy = get_all_taxonomy
        mock_db.get_content_by_id = get_content_by_id
        mock_db.upsert_matching = upsert_matching
        mock_db.get_best_match_for_taxonomy = get_best_match_for_taxonomy
        mock_db.get_categorizations_by_content = get_categorizations_by_content
        mock_db.get_all_matchings = get_all_matchings

        return mock_db

    def test_wordpress_connector_with_public_site(self, public_wordpress_site: str) -> None:
        """Test WordPress connector with real public API."""
        connector = WordPressVIPConnector(public_wordpress_site)

        # Test connection
        assert connector.test_connection()

        # Fetch a small number of posts
        posts = connector.get_posts(page=1, per_page=5)

        # Verify we got some posts
        assert len(posts) > 0
        assert "title" in posts[0]
        assert "content" in posts[0]
        assert "link" in posts[0]

    def test_ingestion_with_public_site(
        self,
        public_wordpress_site: str,
        mock_settings: Settings,
        mock_db_client: Mock,
    ) -> None:
        """Test content ingestion from public WordPress site."""
        ingestion_service = IngestionService(mock_settings, mock_db_client)

        # Ingest limited content (5 posts max per API call)
        count = ingestion_service.ingest_wordpress_sites([public_wordpress_site], max_pages=1)

        # Should have ingested some content
        assert count > 0

        # Verify content is stored
        stored_content = mock_db_client.get_all_content()
        assert len(stored_content) > 0

        # Verify content structure
        first_content = stored_content[0]
        assert first_content.title
        assert first_content.content
        assert first_content.url
        assert str(first_content.site_url) == public_wordpress_site

    def test_taxonomy_loading(
        self,
        mock_settings: Settings,
        mock_db_client: Mock,
        tmp_path: Path,
    ) -> None:
        """Test taxonomy loading from CSV."""
        # Create test taxonomy CSV
        taxonomy_csv = tmp_path / "test_taxonomy.csv"
        taxonomy_csv.write_text(
            "url,category,description,keywords\n"
            "https://example.com/page1,News,News articles,wordpress;updates;news\n"
            "https://example.com/page2,Tech,Technology content,technology;development;code\n"
        )

        ingestion_service = IngestionService(mock_settings, mock_db_client)
        count = ingestion_service.load_taxonomy_from_csv(taxonomy_csv)

        assert count == 2

        taxonomy_pages = mock_db_client.get_all_taxonomy()
        assert len(taxonomy_pages) == 2
        assert taxonomy_pages[0].category in ["News", "Tech"]
        assert len(taxonomy_pages[0].keywords) > 0

    @patch("src.services.matching.openai.OpenAI")
    def test_semantic_matching_integration(
        self,
        mock_openai_class: Mock,
        mock_openai_client: Mock,
        mock_settings: Settings,
        mock_db_client: Mock,
        public_wordpress_site: str,
        tmp_path: Path,
    ) -> None:
        """Test semantic matching integration with real WordPress content."""
        mock_openai_class.return_value = mock_openai_client

        # Step 1: Ingest content
        ingestion_service = IngestionService(mock_settings, mock_db_client)
        ingestion_service.ingest_wordpress_sites([public_wordpress_site], max_pages=1)

        # Step 2: Load taxonomy
        taxonomy_csv = tmp_path / "test_taxonomy.csv"
        taxonomy_csv.write_text(
            "url,category,description,keywords\n"
            "https://example.com/wordpress-news,News,WordPress news and updates,wordpress;news;updates;announcements\n"
        )
        ingestion_service.load_taxonomy_from_csv(taxonomy_csv)

        # Step 3: Perform matching
        matching_service = MatchingService(mock_settings, mock_db_client)

        # Use very low threshold to ensure we get matches
        results = matching_service.match_all_taxonomy(min_threshold=0.0, store_results=True)

        # Should have matching results
        assert len(results) > 0

        # Verify matching results structure
        for taxonomy_id, match_result in results.items():
            assert match_result is not None
            assert match_result.taxonomy_id == taxonomy_id

    def test_csv_export_integration(
        self,
        mock_settings: Settings,
        mock_db_client: Mock,
        tmp_path: Path,
    ) -> None:
        """Test CSV export with complete data."""
        # Add test data to mock DB
        taxonomy = TaxonomyPage(
            url="https://example.com/test",
            category="Test",
            description="Test page",
            keywords=["test"],
        )
        mock_db_client.upsert_taxonomy(taxonomy)

        # Export
        exporter = CSVExporter(mock_db_client)
        output_path = tmp_path / "export.csv"

        count = exporter.export_to_csv(output_path)

        assert count > 0
        assert output_path.exists()

        # Verify CSV structure
        with open(output_path) as f:
            lines = f.readlines()
            assert "source_url" in lines[0]
            assert "target_url" in lines[0]
            assert "category" in lines[0]
            assert "similarity_score" in lines[0]

    @pytest.mark.skipif(
        not os.getenv("RUN_SLOW_TESTS"),
        reason="Skipping slow test - set RUN_SLOW_TESTS=1 to run",
    )
    @patch("src.services.matching.openai.OpenAI")
    def test_full_e2e_pipeline(
        self,
        mock_openai_class: Mock,
        mock_openai_client: Mock,
        mock_settings: Settings,
        mock_db_client: Mock,
        public_wordpress_site: str,
        tmp_path: Path,
    ) -> None:
        """Test complete end-to-end pipeline.

        This test runs the full workflow:
        1. Ingest WordPress content
        2. Load taxonomy
        3. Perform semantic matching
        4. Export results to CSV

        Set RUN_SLOW_TESTS=1 environment variable to run this test.
        """
        mock_openai_class.return_value = mock_openai_client

        # 1. Ingest content
        ingestion_service = IngestionService(mock_settings, mock_db_client)
        content_count = ingestion_service.ingest_wordpress_sites(
            [public_wordpress_site], max_pages=1
        )
        assert content_count > 0

        # 2. Load taxonomy
        taxonomy_csv = tmp_path / "taxonomy.csv"
        taxonomy_csv.write_text(
            "url,category,description,keywords\n"
            "https://example.com/wp-news,WordPress News,WordPress announcements,wordpress;release;update;announcement\n"
            "https://example.com/tech,Technology,Tech articles,technology;development;programming\n"
        )
        taxonomy_count = ingestion_service.load_taxonomy_from_csv(taxonomy_csv)
        assert taxonomy_count == 2

        # 3. Perform matching
        matching_service = MatchingService(mock_settings, mock_db_client)
        results = matching_service.match_all_taxonomy_batch(min_threshold=0.0, store_results=True)
        assert len(results) == 2

        # 4. Export results
        exporter = CSVExporter(mock_db_client)
        output_path = tmp_path / "final_export.csv"
        export_count = exporter.export_to_csv(output_path)

        assert export_count == 2
        assert output_path.exists()

        # 5. Verify export content
        with open(output_path) as f:
            content = f.read()
            assert "WordPress News" in content
            assert "Technology" in content

        print("\n✓ Full E2E pipeline test completed successfully!")
        print(f"  - Ingested: {content_count} content items")
        print(f"  - Loaded: {taxonomy_count} taxonomy pages")
        print(f"  - Matched: {len(results)} results")
        print(f"  - Exported: {export_count} rows to {output_path}")
