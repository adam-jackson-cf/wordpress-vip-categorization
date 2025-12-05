"""Content categorization service powered by the OpenAI Batch API."""

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
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
from src.optimization.dspy_optimizer import DSPyOptimizer, PromptContext

logger = logging.getLogger(__name__)

OPENAI_RETRY_EXCEPTIONS = (
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIStatusError,
)


@dataclass(frozen=True)
class BatchRequestFile:
    """Metadata describing a rendered JSONL file ready for Batch submission."""

    path: Path
    run_dir: Path
    count: int


@dataclass(slots=True)
class LLMBatchStats:
    matched: int = 0
    needs_review: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "matched": self.matched,
            "needs_review": self.needs_review,
            "total": self.total,
        }


@dataclass(slots=True)
class LLMBatchDecision:
    content_id: UUID
    decision: str
    taxonomy_id: UUID | None
    rubric_payload: dict[str, Any]
    topic_alignment: float | None
    intent_fit: float | None
    entity_overlap: float | None


LLM_SYSTEM_PROMPT = (
    "You are a bilingual (Spanish/English) taxonomy reviewer for MSD Animal Health. "
    "Given WordPress source content plus a short list of candidate taxonomy pages, "
    "pick the single best taxonomy destination only if it clearly satisfies the rubric. "
    "Return strict JSON using the provided schema. Decisions must be one of: accept (confident match), "
    "review (not enough evidence or ambiguous), or reject (none of the candidates are relevant)."
)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


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
        self.dspy_optimizer = DSPyOptimizer(settings, db_client)
        self.prompt_instructions: str | None = None
        self.prompt_demonstrations: list[str] = []
        self._initialize_prompt_context()

        logger.info("Initialized categorization service with base URL: %s", settings.llm_base_url)
        self.batch_artifact_root = settings.llm_batch_artifact_dir
        self.batch_artifact_root.mkdir(parents=True, exist_ok=True)
        self.batch_chunk_size = settings.llm_batch_chunk_size

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

    def _initialize_prompt_context(self) -> None:
        """Load the latest DSPy matcher and cache its prompt metadata."""

        try:
            loaded = self.dspy_optimizer.load_latest_model()
            if loaded is not None:
                logger.info("Loaded optimized DSPy matcher for batch prompts")
            else:
                logger.info("No optimized DSPy matcher found; using default instructions")
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Failed to load optimized DSPy matcher: %s", exc)

        try:
            context: PromptContext = self.dspy_optimizer.get_prompt_context()
            self.prompt_instructions = (
                context.instructions.strip() if context.instructions else None
            )
            self.prompt_demonstrations = [
                demo.strip() for demo in context.demonstrations if demo.strip()
            ]
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Failed to extract DSPy prompt context: %s", exc)
            self.prompt_instructions = None
            self.prompt_demonstrations = []

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

    def _format_content_section(self, content: WordPressContent) -> str:
        metadata = content.metadata or {}
        categories = ", ".join(str(value) for value in metadata.get("categories", []) or [])
        tags = ", ".join(str(value) for value in metadata.get("tags", []) or [])
        excerpt = (metadata.get("excerpt") or content.content[:800] or "").strip()
        published = content.published_date.isoformat() if content.published_date else "unknown"
        detected_audiences = ", ".join(
            sorted(content.detected_audiences or metadata.get("detected_audiences") or [])
        )
        detected_species = ", ".join(
            sorted(content.detected_species or metadata.get("detected_species") or [])
        )
        detected_audiences = detected_audiences or "unknown"
        detected_species = detected_species or "unknown"
        return "\n".join(
            part
            for part in [
                f"Title: {content.title}",
                f"URL: {content.url}",
                f"Site: {content.site_url}",
                f"Published: {published}",
                f"Detected Audiences: {detected_audiences}",
                f"Detected Species: {detected_species}",
                f"Categories: {categories or 'n/a'}",
                f"Tags: {tags or 'n/a'}",
                f"Excerpt: {excerpt[:800]}",
                f"Full Preview: {content.content[:1500]}",
            ]
            if part
        )

    @staticmethod
    def _format_candidate_section(candidates: list[TaxonomyPage]) -> str:
        if not candidates:
            return "No semantic candidates were found; default to review."

        sections: list[str] = []
        for idx, taxonomy in enumerate(candidates, start=1):
            topics = ", ".join(taxonomy.key_topics)
            audiences = ", ".join(
                filter(None, [taxonomy.primary_audiance, taxonomy.secondary_audiance])
            )
            sections.append(
                "\n".join(
                    part
                    for part in [
                        f"{idx}. taxonomy_id: {taxonomy.id}",
                        f"Destination: {taxonomy.destination_url}",
                        f"Content Type: {taxonomy.content_type}",
                        f"Audiences: {audiences or 'n/a'}",
                        f"Semantic Summary: {taxonomy.semantic_summary[:600]}",
                        f"Key Topics: {topics or 'n/a'}",
                    ]
                    if part
                )
            )
        return "\n\n".join(sections)

    def _format_semantic_hint(self, result: MatchingResult | None) -> str:
        if result is None:
            return (
                "Semantic stage produced no confident candidate. "
                "Review all taxonomy options below."
            )

        taxonomy_id = result.semantic_taxonomy_id or UUID(int=0)
        score = result.semantic_similarity_score
        threshold = self.settings.similarity_threshold
        return (
            "Semantic stage best candidate:\n"
            f"- taxonomy_id: {taxonomy_id}\n"
            f"- similarity_score: {score:.3f}\n"
            f"- threshold: {threshold:.3f}\n"
            "This content stayed below the threshold and needs LLM judgment."
        )

    def _llm_response_schema(self) -> dict[str, Any]:
        return {
            "name": "taxonomy_match_result",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["accept", "review", "reject"],
                        "description": "accept if confident match, review if unsure, reject if irrelevant",
                    },
                    "taxonomy_id": {
                        "type": ["string", "null"],
                        "description": "UUID of the chosen taxonomy from the candidate list",
                    },
                    "taxonomy_url": {
                        "type": ["string", "null"],
                        "description": "Destination URL of the chosen taxonomy",
                    },
                    "topic_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Rubric score 0-1 for topical fit",
                    },
                    "intent_fit": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Rubric score 0-1 for intent alignment",
                    },
                    "entity_overlap": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Rubric score 0-1 for entity/keyword overlap",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief justification for the decision",
                    },
                    "confidence": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Optional overall confidence score",
                    },
                },
                "required": [
                    "decision",
                    "taxonomy_id",
                    "taxonomy_url",
                    "topic_alignment",
                    "intent_fit",
                    "entity_overlap",
                    "reasoning",
                ],
            },
        }

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_llm_batch_record(
        self,
        record: dict[str, Any],
        batch_id: str,
    ) -> LLMBatchDecision | None:
        custom_id = record.get("custom_id")
        try:
            content_id = UUID(str(custom_id))
        except (TypeError, ValueError):
            logger.error("Batch %s included invalid custom_id: %s", batch_id, custom_id)
            return None

        parsed = self._extract_message_json(record)
        taxonomy_uuid: UUID | None = None
        decision = "review"
        rubric_payload: dict[str, Any]
        topic_score: float | None = None
        intent_score: float | None = None
        entity_score: float | None = None

        if parsed is not None:
            decision = str(parsed.get("decision", "review")).strip().lower() or "review"
            taxonomy_value = parsed.get("taxonomy_id")
            if taxonomy_value:
                try:
                    taxonomy_uuid = UUID(str(taxonomy_value))
                except (ValueError, TypeError):
                    logger.warning(
                        "Batch %s returned invalid taxonomy_id '%s' for content %s",
                        batch_id,
                        taxonomy_value,
                        content_id,
                    )
                    taxonomy_uuid = None
            topic_score = self._safe_float(parsed.get("topic_alignment"))
            intent_score = self._safe_float(parsed.get("intent_fit"))
            entity_score = self._safe_float(parsed.get("entity_overlap"))
            rubric_payload = {**parsed, "batch_id": batch_id}
        else:
            rubric_payload = {"batch_id": batch_id, "error": "invalid_response"}

        return LLMBatchDecision(
            content_id=content_id,
            decision=decision,
            taxonomy_id=taxonomy_uuid,
            rubric_payload=rubric_payload,
            topic_alignment=topic_score,
            intent_fit=intent_score,
            entity_overlap=entity_score,
        )

    def _build_matching_result(
        self,
        decision: LLMBatchDecision,
        taxonomy_lookup: dict[UUID, TaxonomyPage],
        semantic_results: dict[UUID, MatchingResult],
    ) -> tuple[MatchingResult, bool, TaxonomyPage | None]:
        taxonomy = taxonomy_lookup.get(decision.taxonomy_id) if decision.taxonomy_id else None
        accepted = False
        if decision.decision == "accept" and taxonomy is not None:
            accepted = self._accept_by_rubric(taxonomy, decision.rubric_payload)

        stage = MatchStage.LLM_CATEGORIZED if accepted else MatchStage.NEEDS_HUMAN_REVIEW
        semantic_match = semantic_results.get(decision.content_id)
        semantic_taxonomy_id = semantic_match.semantic_taxonomy_id if semantic_match else None
        semantic_score = semantic_match.semantic_similarity_score if semantic_match else 0.0
        if semantic_taxonomy_id is None and decision.taxonomy_id is not None:
            semantic_taxonomy_id = decision.taxonomy_id

        llm_topic_score = (
            _clamp(decision.topic_alignment) if decision.topic_alignment is not None else None
        )

        failed_at_stage = None
        if stage == MatchStage.NEEDS_HUMAN_REVIEW:
            if semantic_match and semantic_match.failed_at_stage == "semantic_matching":
                failed_at_stage = "both_stages_failed"
            else:
                failed_at_stage = "llm_batch"

        result_row = MatchingResult(
            content_id=decision.content_id,
            taxonomy_id=decision.taxonomy_id if accepted else None,
            semantic_taxonomy_id=semantic_taxonomy_id,
            semantic_similarity_score=semantic_score,
            llm_topic_score=llm_topic_score,
            match_stage=stage,
            failed_at_stage=failed_at_stage,
            rubric=decision.rubric_payload,
        )

        return result_row, accepted, taxonomy

    @staticmethod
    def _extract_message_json(result: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = result["response"]["body"]
            choices = response.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                text = str(content or "")
            if not text.strip():
                return None
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _compose_system_prompt(self) -> str:
        sections = []
        if self.prompt_instructions:
            sections.append(self.prompt_instructions.strip())
        sections.append(LLM_SYSTEM_PROMPT)
        return "\n\n".join(section for section in sections if section).strip()

    def _render_demonstrations(self) -> str:
        if not self.prompt_demonstrations:
            return ""
        return "\n\n".join(self.prompt_demonstrations)

    def prepare_llm_fallback_requests(
        self,
        content_items: list[WordPressContent],
        candidate_map: dict[UUID, list[TaxonomyPage]],
        semantic_results: dict[UUID, MatchingResult] | None = None,
    ) -> list[dict[str, Any]]:
        semantic_results = semantic_results or {}
        schema = self._llm_response_schema()
        requests: list[dict[str, Any]] = []

        system_prompt = self._compose_system_prompt()
        demos_block = self._render_demonstrations()

        for content in content_items:
            candidates = candidate_map.get(content.id, [])
            prompt_sections: list[str] = []
            if demos_block:
                prompt_sections.extend(
                    [
                        "Reference these rubric examples:",
                        demos_block,
                        "",
                    ]
                )
            prompt_sections.extend(
                [
                    "Content to categorize:",
                    self._format_content_section(content),
                    "\nSemantic evidence:",
                    self._format_semantic_hint(semantic_results.get(content.id)),
                    "\nCandidate taxonomy pages (choose at most one):",
                    self._format_candidate_section(candidates),
                    "\nRubric:",
                    (
                        "Score topic_alignment, intent_fit, and entity_overlap between 0 and 1. "
                        "Only output decision='accept' when all rubric minimums are satisfied. "
                        "If no candidate qualifies, set decision='review' and taxonomy_id=null."
                    ),
                ]
            )

            request = {
                "custom_id": str(content.id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.settings.llm_model,
                    "temperature": 0.0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": schema,
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "\n".join(prompt_sections)},
                    ],
                },
            }
            requests.append(request)

        logger.info("Prepared %s LLM fallback batch requests", len(requests))
        return requests

    def _write_llm_request_files(self, requests: list[dict[str, Any]]) -> list[BatchRequestFile]:
        if not requests:
            return []

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.batch_artifact_root / f"llm_match_{timestamp}_{uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[BatchRequestFile] = []
        chunk_size = max(1, self.batch_chunk_size)
        for index in range(0, len(requests), chunk_size):
            chunk = requests[index : index + chunk_size]
            file_path = run_dir / f"match_llm_requests_{index // chunk_size + 1:03d}.jsonl"
            with file_path.open("w", encoding="utf-8") as handle:
                for request in chunk:
                    handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            artifacts.append(BatchRequestFile(path=file_path, run_dir=run_dir, count=len(chunk)))

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_requests": len(requests),
            "files": [
                {"file": artifact.path.name, "count": artifact.count} for artifact in artifacts
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info("Wrote %s batch file(s) to %s", len(artifacts), run_dir)
        return artifacts

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
        completion_window_value = self.settings.llm_batch_completion_window
        if completion_window_value != "24h":
            raise ValueError("llm_batch_completion_window must be '24h' for OpenAI Batch jobs")
        completion_window: Literal["24h"] = "24h"
        batch = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
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

    def wait_for_batch_completion(  # pragma: no cover - network polling
        self, batch_id: str, check_interval: int = 60
    ) -> BatchJobStatus:
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
    def retrieve_batch_results(  # pragma: no cover - network I/O
        self, batch_id: str
    ) -> list[dict[str, Any]]:
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

    def categorize_content_batch(  # pragma: no cover - external Batch API
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
        categories = list({page.content_type for page in taxonomy_pages})
        logger.info(f"Found {len(categories)} categories in taxonomy")
        return categories

    def categorize_for_matching(
        self,
        content_items: list[WordPressContent],
        candidate_map: dict[UUID, list[TaxonomyPage]] | None = None,
        fallback_taxonomy: list[TaxonomyPage] | None = None,
        semantic_results: dict[UUID, MatchingResult] | None = None,
        wait_for_completion: bool = True,
    ) -> dict[str, Any]:
        """Run the LLM fallback exclusively through the OpenAI Batch API."""

        if not content_items:
            return {
                "matched": 0,
                "below_threshold": 0,
                "total": 0,
                "batch_ids": [],
                "needs_review": 0,
            }

        candidate_map = candidate_map or {}
        semantic_results = semantic_results or {}
        if fallback_taxonomy is None:
            fallback_taxonomy = self.db.get_all_taxonomy()

        requests = self.prepare_llm_fallback_requests(
            content_items, candidate_map, semantic_results
        )
        artifacts = self._write_llm_request_files(requests)
        total_items = len(content_items)
        if not artifacts:
            return {
                "matched": 0,
                "below_threshold": 0,
                "total": total_items,
                "batch_ids": [],
                "needs_review": 0,
            }

        taxonomy_lookup: dict[UUID, TaxonomyPage] = {
            taxonomy.id: taxonomy for taxonomy in fallback_taxonomy
        }
        for entries in candidate_map.values():
            for taxonomy in entries:
                taxonomy_lookup.setdefault(taxonomy.id, taxonomy)

        content_lookup = {content.id: content for content in content_items}

        matched = 0
        below_threshold = 0
        batch_ids: list[str] = []

        for chunk_index, artifact in enumerate(artifacts, start=1):
            description = (
                f"LLM fallback chunk {chunk_index}/{len(artifacts)} "
                f"({artifact.count} content items)"
            )
            batch_id = self.submit_batch(str(artifact.path), description=description)
            batch_ids.append(batch_id)
            logger.info(
                "Submitted batch %s for %s unmatched items (file=%s)",
                batch_id,
                artifact.count,
                artifact.path,
            )

            if not wait_for_completion:
                continue

            self.wait_for_batch_completion(batch_id)
            raw_results = self.retrieve_batch_results(batch_id)
            chunk_stats = self.apply_llm_batch_results(
                raw_results,
                batch_id,
                taxonomy_lookup=taxonomy_lookup,
                semantic_results=semantic_results,
                content_lookup=content_lookup,
            )
            matched += chunk_stats.matched
            below_threshold += chunk_stats.needs_review

        if wait_for_completion:
            logger.info(
                "LLM batch categorization complete: %s matched, %s need review",
                matched,
                below_threshold,
            )
        else:
            logger.info(
                "Submitted %s batch job(s); call batch apply/status commands to finalize results",
                len(batch_ids),
            )

        result = {
            "matched": matched,
            "below_threshold": below_threshold,
            "total": total_items,
            "batch_ids": batch_ids,
            "needs_review": below_threshold,
        }
        return result

    def apply_llm_batch_results(
        self,
        results: list[dict[str, Any]],
        batch_id: str,
        taxonomy_lookup: dict[UUID, TaxonomyPage] | None = None,
        semantic_results: dict[UUID, MatchingResult] | None = None,
        content_lookup: dict[UUID, WordPressContent] | None = None,
    ) -> LLMBatchStats:
        taxonomy_lookup = taxonomy_lookup or {}
        semantic_results = semantic_results or {}
        content_lookup = content_lookup or {}

        stats = LLMBatchStats(total=len(results))
        updates: list[MatchingResult] = []

        for record in results:
            decision = self._parse_llm_batch_record(record, batch_id)
            if decision is None:
                stats.needs_review += 1
                continue

            result_row, accepted, taxonomy = self._build_matching_result(
                decision,
                taxonomy_lookup,
                semantic_results,
            )
            updates.append(result_row)

            content_ref = content_lookup.get(decision.content_id)
            content_label = content_ref.url if content_ref else str(decision.content_id)
            if accepted and taxonomy is not None:
                stats.matched += 1
                logger.info(
                    "LLM batch accepted %s → %s (topic=%.2f, intent=%.2f, entity=%.2f)",
                    content_label,
                    taxonomy.destination_url,
                    result_row.llm_topic_score or 0.0,
                    _clamp(decision.intent_fit or 0.0),
                    _clamp(decision.entity_overlap or 0.0),
                )
            else:
                stats.needs_review += 1
                logger.debug(
                    "LLM batch deferred %s for human review (decision=%s)",
                    content_label,
                    decision.decision,
                )

        if updates:
            self.db.bulk_upsert_matchings(updates, chunk_size=self.settings.matching_batch_size)

        return stats

    def apply_batch_job(self, batch_id: str) -> LLMBatchStats:
        """Download, parse, and persist results for an existing batch job."""

        raw_results = self.retrieve_batch_results(batch_id)
        if not raw_results:
            logger.warning("Batch %s returned no results", batch_id)
            return LLMBatchStats(total=0)

        content_ids: list[UUID] = []
        for record in raw_results:
            try:
                content_ids.append(UUID(str(record.get("custom_id"))))
            except (TypeError, ValueError):
                continue

        semantic_lookup = self.db.get_matchings_by_content_ids(content_ids)
        taxonomy_lookup = {taxonomy.id: taxonomy for taxonomy in self.db.get_all_taxonomy()}
        content_lookup = self.db.get_content_by_ids(content_ids)

        return self.apply_llm_batch_results(
            raw_results,
            batch_id,
            taxonomy_lookup=taxonomy_lookup,
            semantic_results=semantic_lookup,
            content_lookup=content_lookup,
        )

    def _accept_by_rubric(self, taxonomy: TaxonomyPage, rubric: dict[str, float | str]) -> bool:
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
        original_topic, original_intent, original_entity = topic, intent, entity
        topic = max(0.0, min(topic, 1.0))
        intent = max(0.0, min(intent, 1.0))
        entity = max(0.0, min(entity, 1.0))

        # Log warning if any scores were clamped to aid debugging
        if original_topic != topic:
            logger.warning(
                "Clamped topic_alignment from %.2f to %.2f for taxonomy %s",
                original_topic,
                topic,
                taxonomy.destination_url,
            )
        if original_intent != intent:
            logger.warning(
                "Clamped intent_fit from %.2f to %.2f for taxonomy %s",
                original_intent,
                intent,
                taxonomy.destination_url,
            )
        if original_entity != entity:
            logger.warning(
                "Clamped entity_overlap from %.2f to %.2f for taxonomy %s",
                original_entity,
                entity,
                taxonomy.destination_url,
            )

        if topic < self.settings.llm_rubric_topic_min:
            return False
        if intent < self.settings.llm_rubric_intent_min:
            return False
        # Only enforce entity threshold when taxonomy defines key topics, otherwise treat entity
        # overlap as optional (many taxonomy pages have no topic metadata yet).
        enforce_entity = bool(taxonomy.key_topics)
        if enforce_entity and entity < self.settings.llm_rubric_entity_min:
            return False
        return True
