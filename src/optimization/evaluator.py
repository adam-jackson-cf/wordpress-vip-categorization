"""Evaluation framework for categorization and matching quality."""

import logging
from typing import Any

from src.data.supabase_client import SupabaseClient
from src.models import CategorizationResult, MatchStage, MatchingResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for categorization and matching quality."""

    def __init__(self, db_client: SupabaseClient) -> None:
        """Initialize evaluator.

        Args:
            db_client: Supabase database client.
        """
        self.db = db_client
        logger.info("Initialized evaluator")

    def evaluate_categorization(self, results: list[CategorizationResult]) -> dict[str, Any]:
        """Evaluate categorization results.

        Args:
            results: Categorization results to evaluate.

        Returns:
            Evaluation metrics.
        """
        if not results:
            return {
                "count": 0,
                "unique_categories": 0,
                "category_distribution": {},
            }

        categories = [r.category for r in results]

        metrics = {
            "count": len(results),
            "unique_categories": len(set(categories)),
            "category_distribution": self._get_distribution(categories),
        }

        logger.info(f"Categorization evaluation: {metrics}")
        return metrics

    def evaluate_matching(self, results: list[MatchingResult]) -> dict[str, Any]:
        """Evaluate matching results.

        Args:
            results: Matching results to evaluate.

        Returns:
            Evaluation metrics.
        """
        if not results:
            return {
                "count": 0,
                "matched_count": 0,
                "match_rate": 0.0,
                "avg_similarity": 0.0,
            }

        matched_results = [r for r in results if r.content_id is not None]
        similarities = [r.similarity_score for r in matched_results]

        metrics = {
            "count": len(results),
            "matched_count": len(matched_results),
            "unmatched_count": len(results) - len(matched_results),
            "match_rate": len(matched_results) / len(results) if results else 0.0,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
        }

        logger.info(f"Matching evaluation: {metrics}")
        return metrics

    def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all results in database.

        Returns:
            Combined evaluation metrics.
        """
        # Get all matching results
        matching_results = self.db.get_all_matchings()

        # Get categorization results for matched content
        categorization_results = []
        for match in matching_results:
            if match.content_id:
                cats = self.db.get_categorizations_by_content(match.content_id)
                categorization_results.extend(cats)

        return {
            "categorization": self.evaluate_categorization(categorization_results),
            "matching": self.evaluate_matching(matching_results),
            "rubric": self._summarize_rubric(matching_results),
        }

    def _get_distribution(self, items: list[str]) -> dict[str, int]:
        """Get distribution of items.

        Args:
            items: List of items.

        Returns:
            Dictionary mapping item to count.
        """
        distribution: dict[str, int] = {}
        for item in items:
            distribution[item] = distribution.get(item, 0) + 1
        return distribution

    def get_low_quality_matches(self, min_similarity: float = 0.7) -> list[MatchingResult]:
        """Get matches below quality threshold.

        Args:
            min_similarity: Minimum similarity threshold.

        Returns:
            List of low-quality matches.
        """
        all_matches = self.db.get_all_matchings()
        low_quality = [
            m
            for m in all_matches
            if m.content_id is not None and m.similarity_score < min_similarity
        ]

        logger.info(f"Found {len(low_quality)} low-quality matches below {min_similarity}")
        return low_quality

    def get_unmatched_taxonomy(self) -> list[MatchingResult]:
        """Get taxonomy pages with no matches.

        Returns:
            List of unmatched taxonomy results.
        """
        all_matches = self.db.get_all_matchings()
        unmatched = [m for m in all_matches if m.content_id is None]

        logger.info(f"Found {len(unmatched)} unmatched taxonomy pages")
        return unmatched

    def _summarize_rubric(self, results: list[MatchingResult]) -> dict[str, Any]:
        """Summarize rubric distributions for accepted vs. review items.

        Returns:
            Dictionary with averages for numeric rubric fields per group and counts by decision.
        """
        if not results:
            return {
                "accepted_count": 0,
                "review_count": 0,
                "accepted_avg": {},
                "review_avg": {},
                "decisions": {},
            }

        def _avg(items: list[float]) -> float:
            return sum(items) / len(items) if items else 0.0

        numeric_keys = ("topic_alignment", "intent_fit", "entity_overlap", "temporal_relevance")
        accepted = [r for r in results if r.match_stage == MatchStage.LLM_CATEGORIZED and r.rubric]
        review = [r for r in results if r.match_stage == MatchStage.NEEDS_HUMAN_REVIEW and r.rubric]

        accepted_avg: dict[str, float] = {}
        review_avg: dict[str, float] = {}

        for key in numeric_keys:
            accepted_vals: list[float] = []
            for r in accepted:
                try:
                    accepted_vals.append(float((r.rubric or {}).get(key, 0.0)))
                except (ValueError, TypeError):
                    accepted_vals.append(0.0)
            review_vals: list[float] = []
            for r in review:
                try:
                    review_vals.append(float((r.rubric or {}).get(key, 0.0)))
                except (ValueError, TypeError):
                    review_vals.append(0.0)
            accepted_avg[key] = _avg(accepted_vals)
            review_avg[key] = _avg(review_vals)

        # Decisions distribution across all rubric-bearing results
        decisions: dict[str, int] = {}
        for r in (x for x in results if x.rubric):
            d = str((r.rubric or {}).get("decision", "")).strip().lower()
            if d:
                decisions[d] = decisions.get(d, 0) + 1

        summary = {
            "accepted_count": len(accepted),
            "review_count": len(review),
            "accepted_avg": accepted_avg,
            "review_avg": review_avg,
            "decisions": decisions,
        }
        logger.info("Rubric summary: %s", summary)
        return summary
