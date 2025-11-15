"""Supabase client for data persistence."""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from supabase import Client, create_client

from src.config import Settings
from src.models import (
    CategorizationResult,
    MatchingResult,
    MatchStage,
    TaxonomyPage,
    WordPressContent,
)

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for interacting with Supabase database."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Supabase client.

        Args:
            settings: Application settings containing Supabase credentials.
        """
        self.client: Client = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("Initialized Supabase client")

    async def init_schema(self) -> None:
        """Initialize database schema if not exists."""
        # Note: Supabase Python client doesn't support DDL directly
        # Schema should be created via Supabase dashboard or SQL migrations
        logger.info("Database schema should be initialized via Supabase dashboard")

    # WordPress Content operations
    def insert_content(self, content: WordPressContent) -> WordPressContent:
        """Insert WordPress content into database.

        Args:
            content: WordPress content to insert.

        Returns:
            Inserted content with database-generated fields.
        """
        data = content.model_dump(mode="json")
        result = self.client.table("wordpress_content").insert(data).execute()
        logger.debug(f"Inserted content: {content.url}")
        return WordPressContent.model_validate(result.data[0])

    def upsert_content(self, content: WordPressContent) -> WordPressContent:
        """Upsert WordPress content (insert or update if exists).

        Args:
            content: WordPress content to upsert.

        Returns:
            Upserted content.
        """
        data = content.model_dump(mode="json")
        result = self.client.table("wordpress_content").upsert(data, on_conflict="url").execute()
        logger.debug(f"Upserted content: {content.url}")
        return WordPressContent.model_validate(result.data[0])

    def get_content_by_url(self, url: str) -> WordPressContent | None:
        """Get content by URL.

        Args:
            url: Content URL.

        Returns:
            WordPress content if found, None otherwise.
        """
        result = self.client.table("wordpress_content").select("*").eq("url", url).execute()
        if result.data:
            return WordPressContent.model_validate(result.data[0])
        return None

    def get_content_by_id(self, content_id: UUID) -> WordPressContent | None:
        """Get content by ID.

        Args:
            content_id: Content UUID.

        Returns:
            WordPress content if found, None otherwise.
        """
        result = (
            self.client.table("wordpress_content").select("*").eq("id", str(content_id)).execute()
        )
        if result.data:
            return WordPressContent.model_validate(result.data[0])
        return None

    def get_all_content(self, limit: int | None = None) -> list[WordPressContent]:
        """Get all WordPress content.

        Args:
            limit: Optional limit on number of records.

        Returns:
            List of WordPress content.
        """
        query = self.client.table("wordpress_content").select("*")
        if limit:
            query = query.limit(limit)
        result = query.execute()
        return [WordPressContent.model_validate(item) for item in result.data]

    def get_content_by_site(self, site_url: str) -> list[WordPressContent]:
        """Get all content from a specific site.

        Args:
            site_url: Site URL to filter by.

        Returns:
            List of WordPress content from the site.
        """
        result = (
            self.client.table("wordpress_content").select("*").eq("site_url", site_url).execute()
        )
        return [WordPressContent.model_validate(item) for item in result.data]

    def get_latest_published_date(self, site_url: str) -> datetime | None:
        """Return the most recent published_date for a site, if available."""

        result = (
            self.client.table("wordpress_content")
            .select("published_date")
            .eq("site_url", site_url)
            .order("published_date", desc=True)
            .limit(1)
            .execute()
        )

        if not result.data:
            return None

        value = result.data[0].get("published_date")
        if not value:
            return None

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    # Taxonomy operations
    def insert_taxonomy(self, taxonomy: TaxonomyPage) -> TaxonomyPage:
        """Insert taxonomy page into database.

        Args:
            taxonomy: Taxonomy page to insert.

        Returns:
            Inserted taxonomy.
        """
        data = taxonomy.model_dump(mode="json")
        result = self.client.table("taxonomy_pages").insert(data).execute()
        logger.debug(f"Inserted taxonomy: {taxonomy.url}")
        return TaxonomyPage.model_validate(result.data[0])

    def upsert_taxonomy(self, taxonomy: TaxonomyPage) -> TaxonomyPage:
        """Upsert taxonomy page.

        Args:
            taxonomy: Taxonomy page to upsert.

        Returns:
            Upserted taxonomy.
        """
        data = taxonomy.model_dump(mode="json")
        result = self.client.table("taxonomy_pages").upsert(data, on_conflict="url").execute()
        logger.debug(f"Upserted taxonomy: {taxonomy.url}")
        return TaxonomyPage.model_validate(result.data[0])

    def get_all_taxonomy(self) -> list[TaxonomyPage]:
        """Get all taxonomy pages.

        Returns:
            List of taxonomy pages.
        """
        result = self.client.table("taxonomy_pages").select("*").execute()
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_ids(self, taxonomy_ids: list[UUID]) -> list[TaxonomyPage]:
        """Get taxonomy pages by their UUIDs."""

        if not taxonomy_ids:
            return []

        result = (
            self.client.table("taxonomy_pages")
            .select("*")
            .in_("id", [str(tid) for tid in taxonomy_ids])
            .execute()
        )
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_urls(self, urls: list[str]) -> list[TaxonomyPage]:
        """Get taxonomy pages whose URLs are in the provided list."""

        if not urls:
            return []

        result = self.client.table("taxonomy_pages").select("*").in_("url", urls).execute()
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_id(self, taxonomy_id: UUID) -> TaxonomyPage | None:
        """Get taxonomy by ID.

        Args:
            taxonomy_id: Taxonomy UUID.

        Returns:
            Taxonomy page if found, None otherwise.
        """
        result = (
            self.client.table("taxonomy_pages").select("*").eq("id", str(taxonomy_id)).execute()
        )
        if result.data:
            return TaxonomyPage.model_validate(result.data[0])
        return None

    # Categorization results operations
    def insert_categorization(self, result: CategorizationResult) -> CategorizationResult:
        """Insert categorization result.

        Args:
            result: Categorization result to insert.

        Returns:
            Inserted categorization result.
        """
        data = result.model_dump(mode="json")
        db_result = self.client.table("categorization_results").insert(data).execute()
        logger.debug(f"Inserted categorization for content: {result.content_id}")
        return CategorizationResult.model_validate(db_result.data[0])

    def get_categorizations_by_content(self, content_id: UUID) -> list[CategorizationResult]:
        """Get categorization results for a content item.

        Args:
            content_id: Content UUID.

        Returns:
            List of categorization results.
        """
        result = (
            self.client.table("categorization_results")
            .select("*")
            .eq("content_id", str(content_id))
            .execute()
        )
        return [CategorizationResult.model_validate(item) for item in result.data]

    def get_categorizations_by_batch(self, batch_id: str) -> list[CategorizationResult]:
        """Get all categorization results for a batch.

        Args:
            batch_id: Batch ID.

        Returns:
            List of categorization results.
        """
        result = (
            self.client.table("categorization_results")
            .select("*")
            .eq("batch_id", batch_id)
            .execute()
        )
        return [CategorizationResult.model_validate(item) for item in result.data]

    # Matching results operations
    def insert_matching(self, result: MatchingResult) -> MatchingResult:
        """Insert matching result.

        Args:
            result: Matching result to insert.

        Returns:
            Inserted matching result.
        """
        data = result.model_dump(mode="json")
        db_result = self.client.table("matching_results").insert(data).execute()
        logger.debug(
            f"Inserted matching: taxonomy={result.taxonomy_id}, content={result.content_id}"
        )
        return MatchingResult.model_validate(db_result.data[0])

    def upsert_matching(self, result: MatchingResult) -> MatchingResult:
        """Upsert matching result.

        Args:
            result: Matching result to upsert.

        Returns:
            Upserted matching result.
        """
        data = result.model_dump(mode="json")
        db_result = (
            self.client.table("matching_results")
            .upsert(data, on_conflict="taxonomy_id,content_id")
            .execute()
        )
        return MatchingResult.model_validate(db_result.data[0])

    def get_matchings_by_taxonomy(self, taxonomy_id: UUID) -> list[MatchingResult]:
        """Get matching results for a taxonomy page.

        Args:
            taxonomy_id: Taxonomy UUID.

        Returns:
            List of matching results.
        """
        result = (
            self.client.table("matching_results")
            .select("*")
            .eq("taxonomy_id", str(taxonomy_id))
            .order("similarity_score", desc=True)
            .execute()
        )
        return [MatchingResult.model_validate(item) for item in result.data]

    def get_all_matchings(self) -> list[MatchingResult]:
        """Get all matching results.

        Returns:
            List of all matching results.
        """
        result = (
            self.client.table("matching_results")
            .select("*")
            .order("similarity_score", desc=True)
            .execute()
        )
        return [MatchingResult.model_validate(item) for item in result.data]

    def get_best_match_for_taxonomy(
        self, taxonomy_id: UUID, min_score: float = 0.0
    ) -> MatchingResult | None:
        """Get best matching result for a taxonomy page.

        Args:
            taxonomy_id: Taxonomy UUID.
            min_score: Minimum similarity score threshold.

        Returns:
            Best matching result if found and above threshold, None otherwise.
        """
        result = (
            self.client.table("matching_results")
            .select("*")
            .eq("taxonomy_id", str(taxonomy_id))
            .gte("similarity_score", min_score)
            .order("similarity_score", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return MatchingResult.model_validate(result.data[0])
        return None

    def clear_matching_results(
        self,
        taxonomy_ids: list[UUID] | None = None,
        stages: list[MatchStage] | None = None,
    ) -> int:
        """Delete matching results filtered by taxonomy IDs and/or stages."""

        query = self.client.table("matching_results").delete()

        if taxonomy_ids:
            query = query.in_("taxonomy_id", [str(tid) for tid in taxonomy_ids])

        if stages:
            query = query.in_("match_stage", [stage.value for stage in stages])

        result = query.execute()
        deleted = len(result.data or [])
        logger.info(
            "Cleared %s matching results (taxonomy_ids=%s, stages=%s)",
            deleted,
            taxonomy_ids,
            stages,
        )
        return deleted

    def bulk_insert(self, table: str, records: list[dict[str, Any]]) -> None:
        """Bulk insert records into a table.

        Args:
            table: Table name.
            records: List of records to insert.
        """
        if not records:
            return
        self.client.table(table).insert(records).execute()
        logger.info(f"Bulk inserted {len(records)} records into {table}")
