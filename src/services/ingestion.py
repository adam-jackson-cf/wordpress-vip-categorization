"""Ingestion service orchestrating content fetching and storage."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import HttpUrl

from src.config import Settings
from src.connectors.wordpress_vip import WordPressVIPConnector
from src.data.supabase_client import SupabaseClient
from src.models import TaxonomyPage

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for orchestrating content ingestion."""

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize ingestion service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        logger.info("Initialized ingestion service")

    def ingest_wordpress_sites(
        self,
        site_urls: list[str],
        max_pages: int | None = None,
        since: datetime | None = None,
        resume: bool = False,
    ) -> int:
        """Ingest content from WordPress sites.

        Args:
            site_urls: List of WordPress site URLs.
            max_pages: Maximum pages to fetch per site (None for all).
            since: Only ingest content published after this timestamp.
            resume: If True, start from the last known published date per site.

        Returns:
            Total number of content items ingested.
        """
        total_ingested = 0

        for site_url in site_urls:
            logger.info(f"Starting ingestion from {site_url}")

            site_since = since
            if resume and site_since is None:
                site_since = self.db.get_latest_published_date(site_url)
                if site_since:
                    logger.info(
                        "Resuming ingestion for %s from %s",
                        site_url,
                        site_since.isoformat(),
                    )
            elif site_since:
                logger.info(
                    "Ingesting %s content published after %s",
                    site_url,
                    site_since.isoformat(),
                )

            connector = WordPressVIPConnector(
                site_url=site_url,
                auth_token=self.settings.wordpress_vip_auth_token or None,
            )

            # Test connection
            if not connector.test_connection():
                logger.error(f"Failed to connect to {site_url}, skipping")
                continue

            # Fetch and store content
            site_count = 0
            for content in connector.fetch_all_content(
                max_pages=max_pages,
                show_progress=True,
                modified_after=site_since,
            ):
                try:
                    self.db.upsert_content(content)
                    site_count += 1
                except Exception as e:
                    logger.error(f"Error storing content {content.url}: {e}")

            logger.info(f"Ingested {site_count} items from {site_url}")
            total_ingested += site_count

        logger.info(f"Total ingestion completed: {total_ingested} items")
        return total_ingested

    def load_taxonomy_from_csv(self, csv_path: Path) -> int:
        """Load taxonomy from CSV file.

        Expected CSV format:
        url,category,description,keywords

        Where keywords is a semicolon-separated list.

        Args:
            csv_path: Path to taxonomy CSV file.

        Returns:
            Number of taxonomy pages loaded.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {csv_path}")

        count = 0
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse keywords if present
                    keywords = []
                    if "keywords" in row and row["keywords"]:
                        keywords = [kw.strip() for kw in row["keywords"].split(";") if kw.strip()]

                    taxonomy = TaxonomyPage(
                        url=cast(HttpUrl, row["url"]),
                        category=row["category"],
                        description=row["description"],
                        keywords=keywords,
                    )

                    self.db.upsert_taxonomy(taxonomy)
                    count += 1

                except Exception as e:
                    logger.error(f"Error loading taxonomy row {row}: {e}")

        logger.info(f"Loaded {count} taxonomy pages from {csv_path}")
        return count

    def get_ingestion_stats(self) -> dict[str, int]:
        """Get statistics about ingested data.

        Returns:
            Dictionary with counts of various data types.
        """
        stats = {
            "wordpress_content": len(self.db.get_all_content()),
            "taxonomy_pages": len(self.db.get_all_taxonomy()),
            "categorizations": 0,  # Would need separate query
            "matchings": len(self.db.get_all_matchings()),
        }

        logger.info(f"Ingestion stats: {stats}")
        return stats
