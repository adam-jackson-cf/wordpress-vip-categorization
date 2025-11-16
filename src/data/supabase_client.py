"""Supabase client for data persistence."""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

import requests
from postgrest import APIError as PostgrestAPIError
from supabase import Client, create_client
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.models import (
    CategorizationResult,
    MatchingResult,
    MatchStage,
    TaxonomyPage,
    WordPressContent,
    WorkflowRun,
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
        self._retryer = Retrying(
            retry=retry_if_exception_type((requests.RequestException, PostgrestAPIError)),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        logger.info("Initialized Supabase client")

    def _with_retry(self, func: Callable[[], Any]) -> Any:
        """Execute a Supabase call with retry/backoff."""

        return self._retryer(func)

    def _demote_current_records(self, taxonomy_ids: list[UUID]) -> None:
        """Mark existing matching rows as non-current before inserting replacements."""

        if not taxonomy_ids:
            return

        id_strings = [str(tid) for tid in taxonomy_ids]
        self._with_retry(
            lambda: self.client.table("matching_results")
            .update({"is_current": False})
            .in_("taxonomy_id", id_strings)
            .eq("is_current", True)
            .execute()
        )

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
        result = self._with_retry(
            lambda: self.client.table("wordpress_content").insert(data).execute()
        )
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
        result = self._with_retry(
            lambda: self.client.table("wordpress_content").upsert(data, on_conflict="url").execute()
        )
        logger.debug(f"Upserted content: {content.url}")
        return WordPressContent.model_validate(result.data[0])

    def bulk_upsert_content(
        self, contents: list[WordPressContent], chunk_size: int = 200
    ) -> list[WordPressContent]:
        """Bulk upsert WordPress content records."""

        if not contents:
            return []

        persisted: list[WordPressContent] = []
        for i in range(0, len(contents), chunk_size):
            chunk = contents[i : i + chunk_size]
            payload = [item.model_dump(mode="json") for item in chunk]

            def _exec_upsert(data: list[dict[str, Any]] = payload) -> Any:
                return (
                    self.client.table("wordpress_content").upsert(data, on_conflict="url").execute()
                )

            result = self._with_retry(_exec_upsert)
            persisted.extend(WordPressContent.model_validate(item) for item in result.data)

        logger.info("Bulk upserted %s content rows", len(persisted))
        return persisted

    def get_content_by_url(self, url: str) -> WordPressContent | None:
        """Get content by URL.

        Args:
            url: Content URL.

        Returns:
            WordPress content if found, None otherwise.
        """
        result = self._with_retry(
            lambda: self.client.table("wordpress_content").select("*").eq("url", url).execute()
        )
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
        result = self._with_retry(
            lambda: self.client.table("wordpress_content")
            .select("*")
            .eq("id", str(content_id))
            .execute()
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
        result = self._with_retry(query.execute)
        return [WordPressContent.model_validate(item) for item in result.data]

    def get_content_by_site(self, site_url: str) -> list[WordPressContent]:
        """Get all content from a specific site.

        Args:
            site_url: Site URL to filter by.

        Returns:
            List of WordPress content from the site.
        """
        result = self._with_retry(
            lambda: self.client.table("wordpress_content")
            .select("*")
            .eq("site_url", site_url)
            .execute()
        )
        return [WordPressContent.model_validate(item) for item in result.data]

    def update_content_embedding(self, content_id: UUID, embedding: list[float]) -> None:
        """Persist embedding vector for a WordPress content row."""

        self._with_retry(
            lambda: self.client.table("wordpress_content")
            .update(
                {
                    "content_embedding": embedding,
                    "embedding_updated_at": datetime.utcnow().isoformat(),
                }
            )
            .eq("id", str(content_id))
            .execute()
        )

    def update_taxonomy_embedding(self, taxonomy_id: UUID, embedding: list[float]) -> None:
        """Persist embedding vector for a taxonomy row."""

        self._with_retry(
            lambda: self.client.table("taxonomy_pages")
            .update(
                {
                    "taxonomy_embedding": embedding,
                    "embedding_updated_at": datetime.utcnow().isoformat(),
                }
            )
            .eq("id", str(taxonomy_id))
            .execute()
        )

    def match_content_by_embedding(
        self,
        embedding: Sequence[float],
        match_threshold: float,
        limit: int,
    ) -> list[tuple[WordPressContent, float]]:
        """Call pgvector helper to fetch the closest content rows."""

        payload = {
            "query_embedding": list(embedding),
            "match_threshold": match_threshold,
            "match_count": limit,
        }

        result = self._with_retry(
            lambda: self.client.rpc("match_wordpress_content", payload).execute()
        )
        rows = cast(list[dict[str, Any]], result.data or [])
        candidates: list[tuple[WordPressContent, float]] = []
        for row in rows:
            similarity = float(row.get("similarity", 0.0))
            content = WordPressContent.model_validate(row)
            candidates.append((content, similarity))
        return candidates

    def get_latest_published_date(self, site_url: str) -> datetime | None:
        """Return the most recent published_date for a site, if available."""

        result = self._with_retry(
            lambda: self.client.table("wordpress_content")
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
        result = self._with_retry(
            lambda: self.client.table("taxonomy_pages").insert(data).execute()
        )
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
        result = self._with_retry(
            lambda: self.client.table("taxonomy_pages").upsert(data, on_conflict="url").execute()
        )
        logger.debug(f"Upserted taxonomy: {taxonomy.url}")
        return TaxonomyPage.model_validate(result.data[0])

    def bulk_upsert_taxonomy(
        self, taxonomies: list[TaxonomyPage], chunk_size: int = 200
    ) -> list[TaxonomyPage]:
        if not taxonomies:
            return []

        persisted: list[TaxonomyPage] = []
        for i in range(0, len(taxonomies), chunk_size):
            chunk = taxonomies[i : i + chunk_size]
            payload = [item.model_dump(mode="json") for item in chunk]

            def _exec_upsert(data: list[dict[str, Any]] = payload) -> Any:
                return self.client.table("taxonomy_pages").upsert(data, on_conflict="url").execute()

            result = self._with_retry(_exec_upsert)
            persisted.extend(TaxonomyPage.model_validate(item) for item in result.data)

        logger.info("Bulk upserted %s taxonomy rows", len(persisted))
        return persisted

    def get_all_taxonomy(self) -> list[TaxonomyPage]:
        """Get all taxonomy pages.

        Returns:
            List of taxonomy pages.
        """
        result = self._with_retry(lambda: self.client.table("taxonomy_pages").select("*").execute())
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_ids(self, taxonomy_ids: list[UUID]) -> list[TaxonomyPage]:
        """Get taxonomy pages by their UUIDs."""

        if not taxonomy_ids:
            return []

        result = self._with_retry(
            lambda: self.client.table("taxonomy_pages")
            .select("*")
            .in_("id", [str(tid) for tid in taxonomy_ids])
            .execute()
        )
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_urls(self, urls: list[str]) -> list[TaxonomyPage]:
        """Get taxonomy pages whose URLs are in the provided list."""

        if not urls:
            return []

        result = self._with_retry(
            lambda: self.client.table("taxonomy_pages").select("*").in_("url", urls).execute()
        )
        return [TaxonomyPage.model_validate(item) for item in result.data]

    def get_taxonomy_by_id(self, taxonomy_id: UUID) -> TaxonomyPage | None:
        """Get taxonomy by ID.

        Args:
            taxonomy_id: Taxonomy UUID.

        Returns:
            Taxonomy page if found, None otherwise.
        """
        result = self._with_retry(
            lambda: self.client.table("taxonomy_pages")
            .select("*")
            .eq("id", str(taxonomy_id))
            .execute()
        )
        if result.data:
            return TaxonomyPage.model_validate(result.data[0])
        return None

    def get_unmatched_taxonomy(self, threshold: float) -> list[TaxonomyPage]:
        """Return taxonomy rows without a canonical match or below similarity threshold."""

        result = self._with_retry(
            lambda: self.client.rpc(
                "get_unmatched_taxonomy",
                {"min_similarity": threshold},
            ).execute()
        )

        rows = cast(list[dict[str, Any]], result.data or [])
        return [TaxonomyPage.model_validate(item) for item in rows]

    # Categorization results operations
    def insert_categorization(self, result: CategorizationResult) -> CategorizationResult:
        """Insert categorization result.

        Args:
            result: Categorization result to insert.

        Returns:
            Inserted categorization result.
        """
        data = result.model_dump(mode="json")
        db_result = self._with_retry(
            lambda: self.client.table("categorization_results").insert(data).execute()
        )
        logger.debug(f"Inserted categorization for content: {result.content_id}")
        return CategorizationResult.model_validate(db_result.data[0])

    def get_categorizations_by_content(self, content_id: UUID) -> list[CategorizationResult]:
        """Get categorization results for a content item.

        Args:
            content_id: Content UUID.

        Returns:
            List of categorization results.
        """
        result = self._with_retry(
            lambda: self.client.table("categorization_results")
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
        result = self._with_retry(
            lambda: self.client.table("categorization_results")
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
        self._demote_current_records([result.taxonomy_id])
        data = result.model_dump(mode="json")
        data["updated_at"] = datetime.utcnow().isoformat()
        data["is_current"] = True
        db_result = self._with_retry(
            lambda: self.client.table("matching_results").insert(data).execute()
        )
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
        self._demote_current_records([result.taxonomy_id])
        data = result.model_dump(mode="json")
        data["updated_at"] = datetime.utcnow().isoformat()
        data["is_current"] = True
        db_result = self._with_retry(
            lambda: self.client.table("matching_results")
            .upsert(data, on_conflict="taxonomy_id")
            .execute()
        )
        return MatchingResult.model_validate(db_result.data[0])

    def bulk_upsert_matchings(
        self, matchings: list[MatchingResult], chunk_size: int = 200
    ) -> list[MatchingResult]:
        if not matchings:
            return []

        persisted: list[MatchingResult] = []
        for i in range(0, len(matchings), chunk_size):
            chunk = matchings[i : i + chunk_size]
            self._demote_current_records([item.taxonomy_id for item in chunk])
            payload = [
                {
                    **item.model_dump(mode="json"),
                    "updated_at": datetime.utcnow().isoformat(),
                    "is_current": True,
                }
                for item in chunk
            ]

            def _exec_upsert(data: list[dict[str, Any]] = payload) -> Any:
                return (
                    self.client.table("matching_results")
                    .upsert(data, on_conflict="taxonomy_id")
                    .execute()
                )

            result = self._with_retry(_exec_upsert)
            persisted.extend(MatchingResult.model_validate(item) for item in result.data)

        logger.info("Bulk upserted %s matching rows", len(persisted))
        return persisted

    # Workflow run operations
    def create_workflow_run(self, run: WorkflowRun) -> WorkflowRun:
        payload = run.model_dump(mode="json")
        result = self._with_retry(
            lambda: self.client.table("workflow_runs").insert(payload).execute()
        )
        return WorkflowRun.model_validate(result.data[0])

    def update_workflow_run(self, run_id: UUID, **fields: Any) -> WorkflowRun:
        fields["updated_at"] = datetime.utcnow().isoformat()
        result = self._with_retry(
            lambda: self.client.table("workflow_runs")
            .update(fields)
            .eq("id", str(run_id))
            .execute()
        )
        return WorkflowRun.model_validate(result.data[0])

    def get_workflow_run_by_key(self, run_key: str) -> WorkflowRun | None:
        result = self._with_retry(
            lambda: self.client.table("workflow_runs")
            .select("*")
            .eq("run_key", run_key)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return WorkflowRun.model_validate(result.data[0])

    def list_workflow_runs(self, limit: int = 20) -> list[WorkflowRun]:
        result = self._with_retry(
            lambda: self.client.table("workflow_runs")
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [WorkflowRun.model_validate(item) for item in result.data]

    def get_matchings_by_taxonomy(self, taxonomy_id: UUID) -> list[MatchingResult]:
        """Get matching results for a taxonomy page.

        Args:
            taxonomy_id: Taxonomy UUID.

        Returns:
            List of matching results.
        """
        result = self._with_retry(
            lambda: self.client.table("matching_results")
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
        result = self._with_retry(
            lambda: self.client.table("matching_results")
            .select("*")
            .eq("is_current", True)
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
        result = self._with_retry(
            lambda: self.client.table("matching_results")
            .select("*")
            .eq("taxonomy_id", str(taxonomy_id))
            .eq("is_current", True)
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

        result = self._with_retry(query.execute)
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
        self._with_retry(lambda: self.client.table(table).insert(records).execute())
        logger.info(f"Bulk inserted {len(records)} records into {table}")
