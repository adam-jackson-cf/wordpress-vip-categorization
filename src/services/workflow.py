"""Orchestration service for cascading semantic matching and LLM categorization workflow."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import (
    MatchStage,
    MatchingResult,
    TaxonomyPage,
    WordPressContent,
    WorkflowRun,
    WorkflowRunStatus,
)
from src.services.categorization import CategorizationService
from src.services.matching import MatchingService

logger = logging.getLogger(__name__)


class WorkflowService:
    """Orchestrates the cascading semantic → LLM categorization workflow.

    This service coordinates the multi-stage matching process:
    1. Semantic matching (if enabled) with configurable threshold
    2. LLM categorization fallback (if enabled) for items below semantic threshold
    3. Mark remaining items for human review

    Both stages can be toggled via configuration flags.
    """

    def __init__(
        self,
        settings: Settings,
        db_client: SupabaseClient,
        matching_service: MatchingService | None = None,
        categorization_service: CategorizationService | None = None,
    ) -> None:
        """Initialize workflow service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
            matching_service: Optional pre-initialized matching service.
            categorization_service: Optional pre-initialized categorization service.
        """
        self.settings = settings
        self.db = db_client
        self.matching_service = matching_service or MatchingService(settings, db_client)
        self.categorization_service = categorization_service or CategorizationService(
            settings, db_client
        )

        logger.info(
            f"Initialized workflow service - "
            f"Semantic: {settings.enable_semantic_matching} (threshold: {settings.similarity_threshold}), "
            f"LLM: {settings.enable_llm_categorization}"
        )

    def run_matching_workflow(
        self,
        taxonomy_pages: list[TaxonomyPage] | None = None,
        content_items: list[WordPressContent] | None = None,
        batch_mode: bool = True,
        run_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Run the complete cascading matching workflow.

        Args:
            taxonomy_pages: Optional list of taxonomy pages. If None, loads from database.
            content_items: Optional list of content items. If None, loads from database.
            batch_mode: Use batch processing for embeddings (faster for large datasets).

        Returns:
            Dictionary with workflow statistics:
                - semantic_matched: Count of items matched via semantic similarity
                - llm_categorized: Count of items matched via LLM categorization
                - needs_review: Count of items requiring human review
                - skipped: Count of items skipped due to disabled stages
        """
        # Validate configuration
        if (
            not self.settings.enable_semantic_matching
            and not self.settings.enable_llm_categorization
        ):
            logger.warning(
                "Both semantic matching and LLM categorization are disabled. "
                "No matching will be performed."
            )
            return {
                "semantic_matched": 0,
                "llm_categorized": 0,
                "needs_review": 0,
                "skipped": len(taxonomy_pages) if taxonomy_pages else 0,
            }

        # Load data if not provided
        if content_items is None:
            logger.info("Loading content items from database...")
            content_items = self.db.get_all_content()

        if taxonomy_pages is None:
            logger.info("Loading taxonomy pages from database...")
            taxonomy_pages = self.db.get_all_taxonomy()

        logger.info(
            f"Starting workflow with {len(content_items)} content items "
            f"and {len(taxonomy_pages)} taxonomy pages"
        )

        stats = {
            "semantic_matched": 0,
            "llm_categorized": 0,
            "needs_review": 0,
            "skipped": 0,
        }

        if run_id:
            self.db.update_workflow_run(
                run_id,
                current_stage="semantic_matching",
                status=WorkflowRunStatus.RUNNING.value,
                stats=stats,
            )

        # Stage 1: Semantic Matching
        unmatched_content: list[WordPressContent] = []

        match_results: dict[UUID, MatchingResult] = {}

        if self.settings.enable_semantic_matching:
            logger.info(
                f"Stage 1: Running semantic matching for {len(content_items)} content items "
                f"(threshold >= {self.settings.similarity_threshold})"
            )

            # Run semantic matching for all content items
            if batch_mode:
                match_results = self.matching_service.match_all_taxonomy_batch(
                    taxonomy_pages,
                    content_items,
                    min_threshold=self.settings.similarity_threshold,
                )
            else:
                match_results = self.matching_service.match_all_taxonomy(
                    taxonomy_pages,
                    content_items,
                    min_threshold=self.settings.similarity_threshold,
                )

            accepted_semantic_stages = {MatchStage.URL_MATCHING, MatchStage.SEMANTIC_MATCHED}
            excluded_ids = {
                content_id
                for content_id, result in match_results.items()
                if result.match_stage == MatchStage.URL_CHECKER_EXCLUDED
            }
            matched_content_ids = {
                content_id
                for content_id, result in match_results.items()
                if result.match_stage in accepted_semantic_stages
            }
            processed_ids = {
                content_id
                for content_id, result in match_results.items()
                if content_id not in excluded_ids
            }
            unmatched_content = [
                c for c in content_items if c.id in processed_ids and c.id not in matched_content_ids
            ]

            stats["skipped"] += len(excluded_ids)

            stats["semantic_matched"] = len(matched_content_ids)
            logger.info(
                f"Semantic matching complete: {stats['semantic_matched']} matched, "
                f"{len(unmatched_content)} unmatched"
            )
            if run_id:
                self.db.update_workflow_run(
                    run_id,
                    current_stage="llm_planning",
                    stats=stats,
                )
        else:
            logger.info("Stage 1: Semantic matching disabled, skipping...")
            unmatched_content = content_items

        # Stage 2: LLM Categorization Fallback
        if self.settings.enable_llm_categorization and unmatched_content:
            logger.info(
                f"Stage 2: Running LLM categorization for {len(unmatched_content)} content items "
                f"(semantic < {self.settings.similarity_threshold})"
            )

            candidate_map = self.matching_service.build_candidate_map(
                unmatched_content,
                taxonomy_pages=taxonomy_pages,
            )

            llm_results = self.categorization_service.categorize_for_matching(
                content_items=unmatched_content,
                candidate_map=candidate_map,
                fallback_taxonomy=taxonomy_pages,
                semantic_results=match_results,
            )

            stats["llm_categorized"] = llm_results.get("matched", 0)
            stats["needs_review"] = llm_results.get(
                "below_threshold", llm_results.get("needs_review", 0)
            )
            if batch_ids := llm_results.get("batch_ids"):
                stats["llm_batch_ids"] = batch_ids

            logger.info(
                f"LLM categorization complete: {stats['llm_categorized']} matched, "
                f"{stats['needs_review']} need review"
            )
            if run_id:
                self.db.update_workflow_run(
                    run_id,
                    current_stage="completed",
                    stats=stats,
                )
        elif not self.settings.enable_llm_categorization:
            logger.info("Stage 2: LLM categorization disabled, skipping...")
            stats["needs_review"] = len(unmatched_content)
        else:
            logger.info("Stage 2: No unmatched items to process")

        # Log final summary
        logger.info(
            f"Workflow complete - "
            f"Semantic: {stats['semantic_matched']}, "
            f"LLM: {stats['llm_categorized']}, "
            f"Review: {stats['needs_review']}"
        )

        return stats

    def run_managed_workflow(
        self,
        run_key: str,
        batch_mode: bool = True,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Execute the workflow while persisting run metadata for resuming."""

        existing = self.db.get_workflow_run_by_key(run_key)
        if existing and not resume:
            raise ValueError(f"Run '{run_key}' already exists. Use resume to continue.")

        if not existing:
            run = WorkflowRun(
                run_key=run_key,
                status=WorkflowRunStatus.RUNNING,
                current_stage="initializing",
                config={"batch_mode": batch_mode},
            )
            run = self.db.create_workflow_run(run)
        else:
            run = self.db.update_workflow_run(
                existing.id,
                status=WorkflowRunStatus.RUNNING.value,
                error=None,
            )

        try:
            stats = self.run_matching_workflow(batch_mode=batch_mode, run_id=run.id)
            self.db.update_workflow_run(
                run.id,
                status=WorkflowRunStatus.COMPLETED.value,
                current_stage="completed",
                stats=stats,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return stats
        except Exception as exc:  # pragma: no cover - error propagation
            self.db.update_workflow_run(
                run.id,
                status=WorkflowRunStatus.FAILED.value,
                current_stage="failed",
                error=str(exc),
            )
            raise
