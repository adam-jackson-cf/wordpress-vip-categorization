"""Unit tests for CSV exporter."""

from pathlib import Path
from unittest.mock import Mock

from src.exporters.csv_exporter import CSVExporter
from src.models import ExportRow


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
        # Mock database responses
        mock_supabase_client.get_all_taxonomy.return_value = [sample_taxonomy_page]
        mock_supabase_client.get_best_match_for_taxonomy.return_value = sample_matching_result
        mock_supabase_client.get_content_by_id.return_value = sample_wordpress_content
        mock_supabase_client.get_categorizations_by_content.return_value = [
            sample_categorization_result
        ]

        exporter = CSVExporter(mock_supabase_client)
        rows = exporter.prepare_export_rows()

        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, ExportRow)
        assert row.category == "Technology"
        assert row.confidence == 0.95
        assert row.similarity_score == 0.85

    def test_export_to_csv(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        tmp_path: Path,
    ) -> None:
        """Test CSV export."""
        # Mock with no matches for simplicity
        mock_supabase_client.get_all_taxonomy.return_value = [sample_taxonomy_page]
        mock_supabase_client.get_best_match_for_taxonomy.return_value = None

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
            assert str(sample_taxonomy_page.url) in lines[1]

    def test_export_unmatched_only(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        tmp_path: Path,
    ) -> None:
        """Test exporting only unmatched taxonomy."""
        mock_supabase_client.get_all_taxonomy.return_value = [sample_taxonomy_page]
        mock_supabase_client.get_best_match_for_taxonomy.return_value = None

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "unmatched.csv"
        count = exporter.export_unmatched_only(output_path)

        assert output_path.exists()
        assert count == 1

        with open(output_path) as f:
            lines = f.readlines()
            assert "source_url" in lines[0]
            assert "category" in lines[0]

    def test_export_with_min_similarity_filter(
        self,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
        sample_wordpress_content,
        sample_matching_result,
        tmp_path: Path,
    ) -> None:
        """Test export with similarity threshold filter."""
        mock_supabase_client.get_all_taxonomy.return_value = [sample_taxonomy_page]
        mock_supabase_client.get_best_match_for_taxonomy.return_value = sample_matching_result
        mock_supabase_client.get_content_by_id.return_value = sample_wordpress_content
        mock_supabase_client.get_categorizations_by_content.return_value = []

        exporter = CSVExporter(mock_supabase_client)

        output_path = tmp_path / "filtered.csv"

        # Export with high threshold (should exclude our 0.85 score match)
        count = exporter.export_to_csv(output_path, min_similarity=0.90)

        assert count == 0  # Match filtered out

        # Export with low threshold (should include our 0.85 score match)
        count = exporter.export_to_csv(output_path, min_similarity=0.80)

        assert count == 1  # Match included
