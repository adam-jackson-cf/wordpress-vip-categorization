"""Content categorization service using OpenAI Batch API and DSPy-optimized matching."""

import json
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import (
    BatchJobStatus,
    CategorizationResult,
    MatchingResult,
    MatchStage,
    TaxonomyPage,
    WordPressContent,
)
from src.optimization.dspy_optimizer import DSPyOptimizer

logger = logging.getLogger(__name__)

OPENAI_RETRY_EXCEPTIONS = (
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIStatusError,
)


class CategorizationService:
    """Service for categorizing content using OpenAI Batch API.

    The Batch API is cost-effective for large-scale categorization tasks,
    offering 50% discount compared to standard API with 24-hour turnaround.
    """

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize categorization service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        self.client = openai.OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

        # Initialize DSPy optimizer for matching
        self.dspy_optimizer = DSPyOptimizer(settings, db_client)

        # Try to load latest optimized model if it exists
        try:
            loaded = self.dspy_optimizer.load_latest_model()
            if loaded is not None:
                logger.info("Loaded latest optimized DSPy matching model")
            else:
                logger.info("No optimized model found, using unoptimized DSPy module")
        except Exception as e:
            logger.warning(f"Failed to load optimized model, using unoptimized: {e}")

        logger.info(f"Initialized categorization service with base URL: {settings.llm_base_url}")
        self.batch_artifact_root = Path("data/batch")
        self.batch_artifact_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        """Convert API timestamp formats into datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.fromtimestamp(float(value))
                except ValueError:
                    pass
        return datetime.utcfromtimestamp(0)

    @staticmethod
    def _extract_request_count(data: Any, key: str) -> int:
        """Safely extract request count metrics."""
        if data is None:
            return 0

        value: Any
        if isinstance(data, dict):
            value = data.get(key, 0)
        else:
            value = getattr(data, key, 0)

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def create_categorization_prompt(self, content: WordPressContent, categories: list[str]) -> str:
        """Create prompt for content categorization.

        Args:
            content: WordPress content to categorize.
            categories: List of available categories.

        Returns:
            Categorization prompt.
        """
        categories_str = ", ".join(categories)
        prompt = f"""Analyze the following content and categorize it into one of these categories: {categories_str}

Title: {content.title}

Content: {content.content[:2000]}...

Respond with a JSON object in this exact format:
{{
  "category": "the most appropriate category",
  "reasoning": "brief explanation of why this category was chosen"
}}
"""
        return prompt

    def prepare_batch_requests(
        self, content_items: list[WordPressContent], categories: list[str]
    ) -> list[dict[str, Any]]:
        """Prepare batch requests for OpenAI Batch API.

        Args:
            content_items: List of content to categorize.
            categories: Available categories.

        Returns:
            List of batch request objects.
        """
        requests = []
        for content in content_items:
            prompt = self.create_categorization_prompt(content, categories)
            request = {
                "custom_id": str(content.id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.settings.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a content categorization assistant. "
                            "Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                },
            }
            requests.append(request)

        logger.info(f"Prepared {len(requests)} batch requests")
        return requests

    def create_batch_file(self, requests: list[dict[str, Any]]) -> str:
        """Create JSONL file for batch processing.

        Args:
            requests: List of batch requests.

        Returns:
            Path to created JSONL file.
        """
        run_dir = self.batch_artifact_root / f"batch_{int(time.time())}_{uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        file_path = run_dir / "requests.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Created batch file: {file_path}")
        return str(file_path)

    @retry(
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        reraise=True,
    )
    def submit_batch(self, file_path: str, description: str = "") -> str:
        """Submit batch job to OpenAI.

        Args:
            file_path: Path to JSONL batch file.
            description: Optional description for the batch.

        Returns:
            Batch ID.
        """
        # Upload file
        with open(file_path, "rb") as f:
            file_response = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description} if description else {},
        )

        logger.info(f"Submitted batch {batch.id} with file {file_response.id}")
        return str(batch.id)

    def _cleanup_batch_artifacts(self, file_path: str) -> None:
        """Remove temporary batch JSONL files/directories."""

        path = Path(file_path)
        try:
            if path.exists():
                path.unlink()
            parent = path.parent
            if (
                parent.is_dir()
                and parent != self.batch_artifact_root
                and self.batch_artifact_root in parent.parents
            ):
                shutil.rmtree(parent, ignore_errors=True)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to clean up batch artifacts at %s: %s", file_path, exc)

    @retry(
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        reraise=True,
    )
    def get_batch_status(self, batch_id: str) -> BatchJobStatus:
        """Get status of a batch job.

        Args:
            batch_id: Batch ID.

        Returns:
            Batch job status.
        """
        batch = self.client.batches.retrieve(batch_id)

        created_at = self._coerce_datetime(getattr(batch, "created_at", time.time()))
        completed_raw = getattr(batch, "completed_at", None)
        completed_at = self._coerce_datetime(completed_raw) if completed_raw is not None else None
        request_counts = getattr(batch, "request_counts", None)
        counts = {
            "total": self._extract_request_count(request_counts, "total"),
            "completed": self._extract_request_count(request_counts, "completed"),
            "failed": self._extract_request_count(request_counts, "failed"),
        }

        return BatchJobStatus(
            batch_id=str(getattr(batch, "id", batch_id)),
            status=str(getattr(batch, "status", "unknown")),
            created_at=created_at,
            completed_at=completed_at,
            request_counts=counts,
            metadata=getattr(batch, "metadata", {}) or {},
        )

    def wait_for_batch_completion(self, batch_id: str, check_interval: int = 60) -> BatchJobStatus:
        """Wait for batch job to complete.

        Args:
            batch_id: Batch ID.
            check_interval: Seconds between status checks.

        Returns:
            Final batch status.

        Raises:
            TimeoutError: If batch exceeds timeout.
            RuntimeError: If batch fails.
        """
        start_time = time.time()
        timeout = self.settings.llm_batch_timeout

        logger.info(f"Waiting for batch {batch_id} to complete...")
        interval = max(10, check_interval)

        while True:
            status = self.get_batch_status(batch_id)

            if status.status == "completed":
                logger.info(f"Batch {batch_id} completed successfully")
                return status

            if status.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")

            if status.status in ["expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_id} was {status.status}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Batch {batch_id} exceeded timeout of {timeout}s")

            logger.debug(
                f"Batch {batch_id} status: {status.status} "
                f"({status.request_counts.get('completed', 0)}/"
                f"{status.request_counts.get('total', 0)} completed)"
            )

            sleep_for = min(interval, 120) + random.uniform(0, 5)
            logger.debug("Sleeping %.1fs before next batch poll", sleep_for)
            time.sleep(sleep_for)
            interval = min(int(interval * 1.5), 120)

    @retry(
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        reraise=True,
    )
    def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Retrieve results from completed batch.

        Args:
            batch_id: Batch ID.

        Returns:
            List of result objects.
        """
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch.status})")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download results
        file_response = self.client.files.content(batch.output_file_id)
        results = []

        for line in file_response.text.strip().split("\n"):
            if line:
                results.append(json.loads(line))

        logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        return results

    def parse_batch_results(
        self, results: list[dict[str, Any]], batch_id: str
    ) -> list[CategorizationResult]:
        """Parse batch results into categorization results.

        Args:
            results: Raw batch results.
            batch_id: Batch ID.

        Returns:
            List of categorization results.
        """
        categorizations = []

        for result in results:
            try:
                content_id = UUID(result["custom_id"])
                response = result["response"]["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(response)

                categorization = CategorizationResult(
                    content_id=content_id,
                    category=parsed["category"],
                    batch_id=batch_id,
                )
                categorizations.append(categorization)

            except Exception as e:
                logger.error(f"Error parsing result for {result.get('custom_id')}: {e}")
                continue

        logger.info(f"Parsed {len(categorizations)} categorization results")
        return categorizations

    def categorize_content_batch(
        self, content_items: list[WordPressContent], categories: list[str], wait: bool = True
    ) -> str:
        """Categorize content using batch API.

        Args:
            content_items: Content to categorize.
            categories: Available categories.
            wait: Whether to wait for batch completion.

        Returns:
            Batch ID.
        """
        # Prepare batch requests
        requests = self.prepare_batch_requests(content_items, categories)

        # Create batch file
        file_path = self.create_batch_file(requests)

        # Submit batch
        batch_id = self.submit_batch(
            file_path, description=f"Categorize {len(content_items)} content items"
        )
        self._cleanup_batch_artifacts(file_path)

        # Optionally wait for completion
        if wait:
            self.wait_for_batch_completion(batch_id)

            # Retrieve and store results
            results = self.retrieve_batch_results(batch_id)
            categorizations = self.parse_batch_results(results, batch_id)

            # Store in database
            for cat in categorizations:
                self.db.insert_categorization(cat)

            logger.info(f"Stored {len(categorizations)} categorization results")

        return batch_id

    def get_categories_from_taxonomy(self) -> list[str]:
        """Get unique categories from taxonomy pages.

        Returns:
            List of unique category names.
        """
        taxonomy_pages = self.db.get_all_taxonomy()
        categories = list({page.category for page in taxonomy_pages})
        logger.info(f"Found {len(categories)} categories in taxonomy")
        return categories

    def categorize_for_matching(
        self,
        taxonomy_pages: list[TaxonomyPage],
        content_items: list[WordPressContent],
        candidate_map: dict[UUID, list[WordPressContent]] | None = None,
    ) -> dict[str, Any]:
        """Use LLM to match taxonomy pages to content with rubric-based deterministic gating.

        This method provides a fallback for items that didn't match via semantic similarity.
        It uses the LLM to evaluate each taxonomy page against content and determine
        the best match based on semantic understanding rather than embedding similarity.

        Args:
            taxonomy_pages: Taxonomy pages to match (typically unmatched from semantic stage).
            content_items: Available content items to match against (fallback pool).
            candidate_map: Optional per-taxonomy shortlist of semantic candidates.

        Returns:
            Dictionary with statistics:
                - matched: Number of taxonomy pages matched above threshold
                - below_threshold: Number below threshold (need human review)
                - total: Total taxonomy pages processed
        """
        logger.info(
            f"Starting LLM categorization for {len(taxonomy_pages)} taxonomy pages "
            f"against {len(content_items)} content items with rubric gating"
        )

        matched_count = 0
        below_threshold_count = 0

        for taxonomy in taxonomy_pages:
            # Create prompt for LLM to find best match
            pool = []
            if candidate_map:
                pool = candidate_map.get(taxonomy.id, [])

            if not pool:
                pool = content_items

            if not pool:
                logger.warning(
                    "No candidate content available for taxonomy %s; marking for review",
                    taxonomy.id,
                )
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=None,
                    similarity_score=0.0,
                    match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
                    failed_at_stage="llm_categorization",
                )
                below_threshold_count += 1
                self.db.upsert_matching(matching_result)
                continue

            best_match, rubric = self._find_best_match_llm(taxonomy, pool)
            existing_record = self.db.get_best_match_for_taxonomy(taxonomy.id)
            semantic_score = (
                existing_record.candidate_similarity_score
                if existing_record and existing_record.candidate_similarity_score is not None
                else (existing_record.similarity_score if existing_record else 0.0)
            )
            previous_candidate_id = (
                existing_record.candidate_content_id
                if existing_record and existing_record.candidate_content_id
                else (existing_record.content_id if existing_record else None)
            )
            candidate_content_id = best_match.id if best_match else previous_candidate_id
            candidate_score = semantic_score if candidate_content_id is not None else None
            llm_topic_score = float(rubric.get("topic_alignment", 0.0)) if rubric else None

            if best_match and self._accept_by_rubric(rubric):
                # Match found above threshold
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=best_match.id,
                    similarity_score=semantic_score,
                    candidate_content_id=candidate_content_id,
                    candidate_similarity_score=candidate_score,
                    llm_topic_score=llm_topic_score,
                    match_stage=MatchStage.LLM_CATEGORIZED,
                    rubric=rubric,
                )
                matched_count += 1
                logger.info(
                    f"LLM matched taxonomy {taxonomy.url} to {best_match.url} "
                    f"(rubric: topic={rubric.get('topic_alignment', 0.0):.2f}, "
                    f"intent={rubric.get('intent_fit', 0.0):.2f}, "
                    f"entities={rubric.get('entity_overlap', 0.0):.2f}, "
                    f"temporal={rubric.get('temporal_relevance', 0.0):.2f}, "
                    f"decision={rubric.get('decision','')})"
                )
            else:
                # Below threshold or no match - needs human review
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=None,
                    similarity_score=semantic_score,
                    candidate_content_id=candidate_content_id,
                    candidate_similarity_score=candidate_score,
                    llm_topic_score=llm_topic_score,
                    match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
                    failed_at_stage="llm_categorization",
                    rubric=rubric if rubric else None,
                )
                below_threshold_count += 1
                logger.warning(
                    f"LLM match for taxonomy {taxonomy.url} below rubric thresholds "
                    f"(rubric: topic={rubric.get('topic_alignment', 0.0):.2f}, "
                    f"intent={rubric.get('intent_fit', 0.0):.2f}, "
                    f"entities={rubric.get('entity_overlap', 0.0):.2f}, "
                    f"temporal={rubric.get('temporal_relevance', 0.0):.2f}, "
                    f"decision={rubric.get('decision','')})"
                )

            # Store result
            self.db.upsert_matching(matching_result)

        logger.info(
            f"LLM categorization complete: {matched_count} matched, "
            f"{below_threshold_count} need review"
        )

        return {
            "matched": matched_count,
            "below_threshold": below_threshold_count,
            "total": len(taxonomy_pages),
        }

    def _find_best_match_llm(
        self, taxonomy: TaxonomyPage, content_items: list[WordPressContent]
    ) -> tuple[WordPressContent | None, dict[str, float | str]]:
        """Use selector (DSPy) to choose best index, then judge to compute rubric for that candidate.

        Args:
            taxonomy: Taxonomy page to match.
            content_items: Available content items.

        Returns:
            Tuple of (best_match, rubric_dict) or (None, {}) if no good match.
        """
        try:
            votes = max(1, getattr(self.settings, "llm_consensus_votes", 1))
            if votes == 1:
                best_match_index, _ = self.dspy_optimizer.predict_match(taxonomy, content_items)
                if 0 <= best_match_index < len(content_items):
                    best = content_items[best_match_index]
                    rubric = self.dspy_optimizer.judge_candidate(taxonomy, best)
                    return best, rubric
                return None, {}

            # Consensus voting across multiple rubric evaluations
            index_counts: dict[int, int] = {}
            for _ in range(votes):
                idx, _ = self.dspy_optimizer.predict_match(taxonomy, content_items)
                if idx < 0 or idx >= len(content_items):
                    continue
                index_counts[idx] = index_counts.get(idx, 0) + 1
                # Defer rubric scoring to judge on the winning index later

            if not index_counts:
                return None, {}

            # Pick index with most votes
            best_index = max(index_counts.items(), key=lambda kv: kv[1])[0]
            best_candidate = content_items[best_index]
            rubric = self.dspy_optimizer.judge_candidate(taxonomy, best_candidate)
            return best_candidate, rubric

        except Exception as e:
            logger.error(
                "Error in DSPy matching for taxonomy %s: %s", taxonomy.url, e, exc_info=True
            )
            return None, {}

    def _accept_by_rubric(self, rubric: dict[str, float | str]) -> bool:
        """Deterministically accept or reject an LLM-selected match based on rubric scores."""
        try:
            decision = str(rubric.get("decision", "")).strip().lower()
            if decision not in {"accept", "abstain", "reject"}:
                return False
            if decision != "accept":
                return False
            topic = float(rubric.get("topic_alignment", 0.0) or 0.0)
            intent = float(rubric.get("intent_fit", 0.0) or 0.0)
            entity = float(rubric.get("entity_overlap", 0.0) or 0.0)
        except (ValueError, TypeError):
            return False

        # Clamp rubric scores into [0, 1] to satisfy data constraints and avoid overflows
        topic = max(0.0, min(topic, 1.0))
        intent = max(0.0, min(intent, 1.0))
        entity = max(0.0, min(entity, 1.0))

        if topic < self.settings.llm_rubric_topic_min:
            return False
        if intent < self.settings.llm_rubric_intent_min:
            return False
        # Only enforce entity threshold if taxonomy has keywords; infer via >0 expectation
        # If model sets entity to 0 while keywords exist, we enforce; else we treat as N/A.
        enforce_entity = True
        try:
            # Best-effort: if entity was exactly 0.0 and there are no keywords, skip enforcement
            # We cannot access taxonomy here, so we treat very small values as "possibly N/A"
            enforce_entity = entity > 0.0 or self.settings.llm_rubric_entity_min <= 0.0
        except Exception:
            enforce_entity = True
        if enforce_entity and entity < self.settings.llm_rubric_entity_min:
            return False
        return True
