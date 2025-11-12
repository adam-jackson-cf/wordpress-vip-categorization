"""CSV exporter for matching results."""

import csv
import logging
from pathlib import Path

from src.data.supabase_client import SupabaseClient
from src.models import ExportRow

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exporter for creating CSV files from matching results."""

    def __init__(self, db_client: SupabaseClient) -> None:
        """Initialize CSV exporter.

        Args:
            db_client: Supabase database client.
        """
        self.db = db_client
        logger.info("Initialized CSV exporter")

    def prepare_export_rows(self) -> list[ExportRow]:
        """Prepare rows for export.

        Returns:
            List of export rows combining taxonomy, content, and matching data.
        """
        rows = []

        # Get all taxonomy pages
        taxonomy_pages = self.db.get_all_taxonomy()

        for taxonomy in taxonomy_pages:
            # Get best match for this taxonomy
            match = self.db.get_best_match_for_taxonomy(taxonomy.id, min_score=0.0)

            # Default values
            target_url = ""
            similarity_score = 0.0
            category = taxonomy.category
            confidence = 0.0

            if match and match.content_id:
                # Get matched content
                content = self.db.get_content_by_id(match.content_id)
                if content:
                    target_url = str(content.url)

                similarity_score = match.similarity_score

                # Get categorization for the content
                categorizations = self.db.get_categorizations_by_content(match.content_id)
                if categorizations:
                    confidence = categorizations[0].confidence

            row = ExportRow(
                source_url=str(taxonomy.url),
                target_url=target_url,
                confidence=confidence,
                category=category,
                similarity_score=similarity_score,
            )
            rows.append(row)

        logger.info(f"Prepared {len(rows)} export rows")
        return rows

    def export_to_csv(
        self,
        output_path: Path,
        include_unmatched: bool = True,
        min_similarity: float | None = None,
    ) -> int:
        """Export matching results to CSV file.

        Args:
            output_path: Path for output CSV file.
            include_unmatched: Whether to include unmatched taxonomy pages.
            min_similarity: Optional minimum similarity threshold for inclusion.

        Returns:
            Number of rows exported.
        """
        rows = self.prepare_export_rows()

        # Filter rows based on criteria
        filtered_rows = []
        for row in rows:
            # Skip unmatched if requested
            if not include_unmatched and not row.target_url:
                continue

            # Skip below similarity threshold if specified
            if min_similarity is not None and row.similarity_score < min_similarity:
                continue

            filtered_rows.append(row)

        # Write to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_url",
                    "target_url",
                    "category",
                    "similarity_score",
                    "confidence",
                ],
            )
            writer.writeheader()

            for row in filtered_rows:
                writer.writerow(
                    {
                        "source_url": row.source_url,
                        "target_url": row.target_url,
                        "category": row.category,
                        "similarity_score": f"{row.similarity_score:.4f}",
                        "confidence": f"{row.confidence:.4f}",
                    }
                )

        logger.info(f"Exported {len(filtered_rows)} rows to {output_path}")
        return len(filtered_rows)

    def export_unmatched_only(self, output_path: Path) -> int:
        """Export only unmatched taxonomy pages.

        Args:
            output_path: Path for output CSV file.

        Returns:
            Number of rows exported.
        """
        rows = self.prepare_export_rows()
        unmatched = [row for row in rows if not row.target_url]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_url", "category"],
            )
            writer.writeheader()

            for row in unmatched:
                writer.writerow(
                    {
                        "source_url": row.source_url,
                        "category": row.category,
                    }
                )

        logger.info(f"Exported {len(unmatched)} unmatched rows to {output_path}")
        return len(unmatched)
