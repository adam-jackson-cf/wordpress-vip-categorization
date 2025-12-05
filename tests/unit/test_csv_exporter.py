"""Unit tests for CSV exporter."""

from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from src.exporters.csv_exporter import CSVExporter
from src.models import ExportRow, MatchingResult, MatchStage


class TestCSVExporter:
    """Tests for CSV exporter."""

    def test_init(self, mock_supabase_client: Mock) -> None:
        """Test exporter initialization."""
        exporter = CSVExporter(mock_supabase_client)
        assert exporter.db == mock_supabase_client

    def test_prepare_export_rows(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        sample_wordpress_content,
        sample_matching_result,
        sample_categorization_result,
    ) -> None:
        """Test export row preparation."""
        # Mock database responses for content-first export
        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = sample_matching_result
        mock_supabase_client.get_taxonomy_by_id.return_value = sample_taxonomy_page
        mock_supabase_client.get_categorizations_by_content.return_value = [
            sample_categorization_result
        ]

        exporter = CSVExporter(mock_supabase_client)
        rows = exporter.prepare_export_rows()

        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, ExportRow)
        assert row.source_url == str(sample_wordpress_content.url)
        assert row.target_url == str(sample_taxonomy_page.destination_url)
        assert row.category == "Veterinary Guidance"
        assert row.similarity_score == 0.85

    def test_url_matching_uses_destination_url(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        sample_wordpress_content,
        sample_matching_result,
    ) -> None:
        sample_matching_result.match_stage = MatchStage.URL_MATCHING
        sample_matching_result.semantic_similarity_score = 1.0

        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = sample_matching_result
        mock_supabase_client.get_taxonomy_by_id.return_value = sample_taxonomy_page
        exporter = CSVExporter(mock_supabase_client)

        rows = exporter.prepare_export_rows()

        assert len(rows) == 1
        assert rows[0].target_url == str(sample_taxonomy_page.destination_url)

    def test_export_to_csv(
        self,
        mock_supabase_client: Mock,
        sample_wordpress_content,
        tmp_path: Path,
    ) -> None:
        """Test CSV export."""
        # Mock with no matches for simplicity
        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = None

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "test_export.csv"
        count = exporter.export_to_csv(output_path)

        assert output_path.exists()
        assert count == 1

        # Verify CSV content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 data row
            assert "source_url" in lines[0]
            assert str(sample_wordpress_content.url) in lines[1]

    def test_export_unmatched_only(
        self,
        mock_supabase_client: Mock,
        sample_wordpress_content,
        tmp_path: Path,
    ) -> None:
        """Test exporting only unmatched content items."""
        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = None

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "unmatched.csv"
        count = exporter.export_unmatched_only(output_path)

        assert output_path.exists()
        assert count == 1

        with open(output_path) as f:
            lines = f.readlines()
            assert "source_url" in lines[0]
            # Category is not included in unmatched export for content items
            assert "category" not in lines[0]

    def test_export_with_min_similarity_filter(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        sample_wordpress_content,
        sample_matching_result,
        tmp_path: Path,
    ) -> None:
        """Test export with similarity threshold filter."""
        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = sample_matching_result
        mock_supabase_client.get_taxonomy_by_id.return_value = sample_taxonomy_page
        mock_supabase_client.get_categorizations_by_content.return_value = []

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "filtered.csv"

        # Export with high threshold (should exclude our 0.85 score match)
        count = exporter.export_to_csv(output_path, min_similarity=0.90)

        assert count == 0  # Match filtered out

        # Export with low threshold (should include our 0.85 score match)
        count = exporter.export_to_csv(output_path, min_similarity=0.80)

        assert count == 1  # Match included

    def test_export_shows_below_threshold_semantic_candidate(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        sample_wordpress_content,
        tmp_path: Path,
    ) -> None:
        """Verify below-threshold matches populate target_url from semantic_taxonomy_id."""
        # Create a match with taxonomy_id=None but semantic_taxonomy_id populated
        # This simulates a below-threshold semantic match
        below_threshold_match = MatchingResult(
            id=uuid4(),
            content_id=sample_wordpress_content.id,
            taxonomy_id=None,  # Not accepted yet
            semantic_taxonomy_id=sample_taxonomy_page.id,  # Best semantic candidate
            semantic_similarity_score=0.65,  # Below 0.7 threshold
            match_stage=MatchStage.NEEDS_LLM_REVIEW,
            failed_at_stage="semantic_matching",
        )

        mock_supabase_client.get_all_content.return_value = [sample_wordpress_content]
        mock_supabase_client.get_best_match_for_content.return_value = below_threshold_match
        mock_supabase_client.get_taxonomy_by_id.return_value = sample_taxonomy_page

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "below_threshold.csv"
        count = exporter.export_to_csv(output_path)

        assert count == 1
        assert output_path.exists()

        # Verify CSV contains target_url and category from semantic candidate
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 data row
            data_line = lines[1]
            assert str(sample_taxonomy_page.destination_url) in data_line
            assert "Veterinary Guidance" in data_line
            assert "0.6500" in data_line  # similarity score
            assert "semantic_matching" in data_line  # failed_at_stage
