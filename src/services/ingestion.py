"""Ingestion service orchestrating content fetching and storage."""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from pydantic import HttpUrl

from src.config import Settings
from src.connectors.wordpress_vip import WordPressVIPConnector
from src.data.supabase_client import SupabaseClient
from src.models import TaxonomyPage, WordPressContent
from src.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for orchestrating content ingestion."""

    def __init__(
        self,
        settings: Settings,
        db_client: SupabaseClient,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize ingestion service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self._content_buffer: list[WordPressContent] = []
        self._taxonomy_buffer: list[TaxonomyPage] = []
        self._seen_content_urls: set[str] = set()
        logger.info("Initialized ingestion service")

    @staticmethod
    def _render_content_embedding_text(content: WordPressContent) -> str:
        preview = content.content[:1000]
        return f"Title: {content.title}\n\nContent: {preview}"

    @staticmethod
    def _render_taxonomy_embedding_text(taxonomy: TaxonomyPage) -> str:
        topics = ", ".join(taxonomy.key_topics) if taxonomy.key_topics else ""
        species = ", ".join(taxonomy.species) if taxonomy.species else ""
        audiences = ", ".join(
            filter(None, [taxonomy.primary_audiance, taxonomy.secondary_audiance])
        )
        parsed = urlparse(str(taxonomy.destination_url))
        url_path = parsed.path
        url_path = url_path.strip("/") or "/"
        parts = [
            f"Content Type: {taxonomy.content_type}",
            f"Summary: {taxonomy.semantic_summary}",
        ]
        if taxonomy.uid:
            parts.append(f"UID: {taxonomy.uid}")
        if taxonomy.english_page_name:
            parts.append(f"English Name: {taxonomy.english_page_name}")
        if taxonomy.local_page_name:
            parts.append(f"Local Name: {taxonomy.local_page_name}")
        if audiences:
            parts.append(f"Audiences: {audiences}")
        parts.append(f"URL Path: {url_path}")
        if topics:
            parts.append(f"Key Topics: {topics}")
        if species:
            parts.append(f"Species: {species}")
        return "\n".join(parts)

    @staticmethod
    def _parse_species_field(raw_value: str | None) -> list[str]:
        if not raw_value:
            return []
        tokens = [token.strip() for token in raw_value.split(",")]
        cleaned: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if not token:
                continue
            lowered = token.lower()
            if lowered in {"n/a", "none"} or lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(token)
        return cleaned

    def ingest_wordpress_sites(
        self,
        site_configs: list[tuple[str, str]],
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

        for site_url, token in site_configs:
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

            connector = WordPressVIPConnector(site_url=site_url, auth_token=token)

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
                    enriched = content
                    if self.embedding_service:
                        try:
                            embedding = self.embedding_service.embed(
                                self._render_content_embedding_text(content)
                            )
                            enriched = content.model_copy(
                                update={
                                    "content_embedding": embedding,
                                    "embedding_updated_at": datetime.now(timezone.utc),
                                }
                            )
                        except Exception as exc:  # pragma: no cover - external API
                            logger.warning(
                                "Embedding generation failed for %s: %s", content.url, exc
                            )

                    content_url = str(enriched.url)
                    if content_url in self._seen_content_urls:
                        logger.warning(
                            "Duplicate content URL detected (%s); skipping subsequent entry",
                            content_url,
                        )
                        continue

                    self._seen_content_urls.add(content_url)
                    self._content_buffer.append(enriched)
                    site_count += 1

                    if len(self._content_buffer) >= self.settings.ingestion_batch_size:
                        self.db.bulk_upsert_content(
                            self._content_buffer,
                            chunk_size=self.settings.ingestion_batch_size,
                        )
                        self._content_buffer.clear()
                except Exception as e:
                    logger.error(f"Error storing content {content.url}: {e}")

            logger.info(f"Ingested {site_count} items from {site_url}")
            total_ingested += site_count

        if self._content_buffer:
            self.db.bulk_upsert_content(
                self._content_buffer,
                chunk_size=self.settings.ingestion_batch_size,
            )
            self._content_buffer.clear()

        logger.info(f"Total ingestion completed: {total_ingested} items")
        return total_ingested

    def load_taxonomy_from_csv(self, csv_path: Path) -> int:
        """Load taxonomy from CSV file.

        Expected CSV format (Spain canonical):
        UID,Destination_URL,English_Page Name,Local_Page_Name,Content_Type,Primary_Audiance,
        Secondary_Audiance,Species,Semantic_Summary,Key_Topics[,Reference_Source]

        Key_Topics is comma-separated.

        Args:
            csv_path: Path to taxonomy CSV file.

        Returns:
            Number of taxonomy pages loaded.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {csv_path}")

        count = 0
        seen_urls: set[str] = set()
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    destination_raw = (row.get("Destination_URL") or "").strip()
                    if not destination_raw:
                        logger.warning("Skipping taxonomy row without Destination_URL: %s", row)
                        continue

                    if destination_raw in seen_urls:
                        logger.warning(
                            "Duplicate taxonomy URL detected (%s); skipping subsequent entry",
                            destination_raw,
                        )
                        continue

                    seen_urls.add(destination_raw)

                    topics = []
                    if row.get("Key_Topics"):
                        topics = [kw.strip() for kw in row["Key_Topics"].split(",") if kw.strip()]

                    species = self._parse_species_field(row.get("Species"))

                    taxonomy = TaxonomyPage(
                        uid=row.get("UID") or None,
                        destination_url=cast(HttpUrl, destination_raw),
                        english_page_name=row.get("English_Page Name") or None,
                        local_page_name=row.get("Local_Page_Name") or None,
                        content_type=row["Content_Type"],
                        primary_audiance=row.get("Primary_Audiance") or None,
                        secondary_audiance=row.get("Secondary_Audiance") or None,
                        species=species,
                        semantic_summary=row["Semantic_Summary"],
                        key_topics=topics,
                    )

                    enriched = taxonomy
                    if self.embedding_service:
                        try:
                            embedding = self.embedding_service.embed(
                                self._render_taxonomy_embedding_text(taxonomy)
                            )
                            enriched = taxonomy.model_copy(
                                update={
                                    "taxonomy_embedding": embedding,
                                    "embedding_updated_at": datetime.now(timezone.utc),
                                }
                            )
                        except Exception as exc:  # pragma: no cover - external API
                            logger.warning(
                                "Embedding generation failed for taxonomy %s: %s",
                                taxonomy.destination_url,
                                exc,
                            )

                    self._taxonomy_buffer.append(enriched)
                    count += 1

                    if len(self._taxonomy_buffer) >= self.settings.ingestion_batch_size:
                        self.db.bulk_upsert_taxonomy(
                            self._taxonomy_buffer,
                            chunk_size=self.settings.ingestion_batch_size,
                        )
                        self._taxonomy_buffer.clear()

                except Exception as e:
                    logger.error(f"Error loading taxonomy row {row}: {e}")

        if self._taxonomy_buffer:
            self.db.bulk_upsert_taxonomy(
                self._taxonomy_buffer,
                chunk_size=self.settings.ingestion_batch_size,
            )
            self._taxonomy_buffer.clear()

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
