"""CSV exporter for matching results."""

import csv
import logging
from pathlib import Path

from src.data.supabase_client import SupabaseClient
from src.models import ExportRow, MatchingResult, TaxonomyPage

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

    @staticmethod
    def _resolve_target_url(taxonomy: TaxonomyPage, match: MatchingResult | None) -> str:
        return str(taxonomy.destination_url)

    def prepare_export_rows(self) -> list[ExportRow]:
        """Prepare rows for export.

        Returns:
            List of export rows combining content, taxonomy, and matching data.
        """
        rows = []

        # Get all content items (now the primary list)
        content_items = self.db.get_all_content()

        for content in content_items:
            # Get best match for this content item
            match = self.db.get_best_match_for_content(content.id, min_score=0.0)

            # Default values
            target_url = ""
            similarity_score = 0.0
            category = ""
            match_stage: str | None = None
            failed_at_stage: str | None = None

            if match and match.taxonomy_id:
                # Accepted match (passed threshold/rubric)
                taxonomy = self.db.get_taxonomy_by_id(match.taxonomy_id)
                if taxonomy:
                    target_url = self._resolve_target_url(taxonomy, match)
                    category = taxonomy.content_type

                similarity_score = match.semantic_similarity_score

                # Get match stage info
                match_stage = match.match_stage.value if match.match_stage else None
                failed_at_stage = match.failed_at_stage
            elif match and match.semantic_taxonomy_id:
                # Below-threshold or rejected match - show best semantic candidate
                taxonomy = self.db.get_taxonomy_by_id(match.semantic_taxonomy_id)
                if taxonomy:
                    target_url = self._resolve_target_url(taxonomy, match)
                    category = taxonomy.content_type

                similarity_score = match.semantic_similarity_score
                match_stage = match.match_stage.value if match.match_stage else None
                failed_at_stage = match.failed_at_stage
            elif match:
                # Match exists but no candidates at all
                similarity_score = match.semantic_similarity_score
                match_stage = match.match_stage.value if match.match_stage else None
                failed_at_stage = match.failed_at_stage

            row = ExportRow(
                source_url=str(content.url),
                target_url=target_url,
                category=category,
                similarity_score=similarity_score,
                match_stage=match_stage,
                failed_at_stage=failed_at_stage,
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
                    "match_stage",
                    "failed_at_stage",
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
                        "match_stage": row.match_stage or "",
                        "failed_at_stage": row.failed_at_stage or "",
                    }
                )

        logger.info(f"Exported {len(filtered_rows)} rows to {output_path}")
        return len(filtered_rows)

    def export_unmatched_only(self, output_path: Path) -> int:
        """Export only unmatched content items.

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
                fieldnames=["source_url"],
            )
            writer.writeheader()

            for row in unmatched:
                writer.writerow(
                    {
                        "source_url": row.source_url,
                    }
                )

        logger.info(f"Exported {len(unmatched)} unmatched rows to {output_path}")
        return len(unmatched)
