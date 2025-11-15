"""Unit-level pipeline tests using in-memory Supabase mocks."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.config import Settings
from src.exporters.csv_exporter import CSVExporter
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService


class TestFullPipeline:
    """Pipeline smoke tests operating entirely in memory."""

    @pytest.fixture
    def public_wordpress_site(self) -> str:
        return "https://example.com"

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

        def bulk_upsert_content(contents, chunk_size=200):
            for item in contents:
                upsert_content(item)
            return contents

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

        def bulk_upsert_taxonomy(taxonomies, chunk_size=200):
            for item in taxonomies:
                upsert_taxonomy(item)
            return taxonomies

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

        def bulk_upsert_matchings(matchings, chunk_size=200):
            for item in matchings:
                upsert_matching(item)
            return matchings

        def get_best_match_for_taxonomy(taxonomy_id, min_score=0.0):
            for m in mock_db._matchings:
                if m.taxonomy_id == taxonomy_id and m.similarity_score >= min_score:
                    return m
            return None

        def match_content_by_embedding(_embedding, _threshold, limit):
            # Return top-N stored content rows with a deterministic similarity score
            return [(content, 0.9) for content in mock_db._content[:limit]]

        def get_categorizations_by_content(content_id):
            return []

        def get_all_matchings():
            return mock_db._matchings[:]

        mock_db.upsert_content = upsert_content
        mock_db.bulk_upsert_content = bulk_upsert_content
        mock_db.get_all_content = get_all_content
        mock_db.upsert_taxonomy = upsert_taxonomy
        mock_db.bulk_upsert_taxonomy = bulk_upsert_taxonomy
        mock_db.get_all_taxonomy = get_all_taxonomy
        mock_db.get_content_by_id = get_content_by_id
        mock_db.upsert_matching = upsert_matching
        mock_db.bulk_upsert_matchings = bulk_upsert_matchings
        mock_db.get_best_match_for_taxonomy = get_best_match_for_taxonomy
        mock_db.match_content_by_embedding = match_content_by_embedding
        mock_db.get_categorizations_by_content = get_categorizations_by_content
        mock_db.get_all_matchings = get_all_matchings

        return mock_db

    @pytest.fixture
    def stub_wordpress(self, mocker, sample_wordpress_content):  # type: ignore[misc]
        mocker.patch(
            "src.services.ingestion.WordPressVIPConnector.test_connection",
            return_value=True,
        )

        def _content_generator(*_args, **_kwargs):
            return iter([sample_wordpress_content])

        mocker.patch(
            "src.services.ingestion.WordPressVIPConnector.fetch_all_content",
            side_effect=_content_generator,
        )

    def test_ingestion_with_public_site(
        self,
        public_wordpress_site: str,
        mock_settings: Settings,
        mock_db_client: Mock,
        stub_wordpress,
    ) -> None:
        """Test content ingestion from public WordPress site."""
        ingestion_service = IngestionService(mock_settings, mock_db_client)

        # Ingest limited content (5 posts max per API call)
        count = ingestion_service.ingest_wordpress_sites([public_wordpress_site], max_pages=1)

        assert count == 1
        stored_content = mock_db_client.get_all_content()
        assert len(stored_content) == 1

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

    def test_semantic_matching_integration(
        self,
        mock_settings: Settings,
        mock_db_client: Mock,
        public_wordpress_site: str,
        tmp_path: Path,
        stub_wordpress,
    ) -> None:
        """Test semantic matching integration with real WordPress content."""
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

    def test_full_e2e_pipeline(
        self,
        mock_settings: Settings,
        mock_db_client: Mock,
        public_wordpress_site: str,
        tmp_path: Path,
        stub_wordpress,
    ) -> None:
        """Test complete end-to-end pipeline.

        This test runs the full workflow:
        1. Ingest WordPress content
        2. Load taxonomy
        3. Perform semantic matching
        4. Export results to CSV

        Set RUN_SLOW_TESTS=1 environment variable to run this test.
        """
        # 1. Ingest content
        ingestion_service = IngestionService(mock_settings, mock_db_client)
        content_count = ingestion_service.ingest_wordpress_sites(
            [public_wordpress_site], max_pages=1
        )
        assert content_count == 1

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
