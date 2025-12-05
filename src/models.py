"""Pydantic models for data validation and serialization."""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MatchStage(str, Enum):
    """Stages in the matching workflow."""

    URL_MATCHING = "url_matching"
    URL_CHECKER_EXCLUDED = "url_checker_excluded"
    SEMANTIC_MATCHED = "semantic_matched"
    NEEDS_LLM_REVIEW = "needs_llm_review"
    LLM_CATEGORIZED = "llm_categorized"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class WordPressContent(BaseModel):
    """WordPress content item."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    title: str
    content: str
    site_url: HttpUrl
    published_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    detected_audiences: list[str] = Field(default_factory=list)
    detected_species: list[str] = Field(default_factory=list)
    content_embedding: list[float] | None = Field(default=None)
    embedding_updated_at: datetime | None = None
    content_length: int | None = Field(default=None, description="Character count of content")
    exclude: bool = Field(
        default=False, description="Whether this content should be excluded from matching"
    )
    exclude_reason: str | None = Field(default=None, description="Reason for exclusion if excluded")
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("detected_audiences", "detected_species", mode="before")
    @classmethod
    def _parse_detected_list(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            tokens = [str(token).strip() for token in value]
        else:
            return []
        cleaned = [token for token in tokens if token]
        return cleaned

    @field_validator("content_embedding", mode="before")
    @classmethod
    def _parse_content_embedding(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value


class TaxonomyPage(BaseModel):
    """Taxonomy page for matching.

    Represents a source page from the taxonomy that needs to be matched
    to target WordPress content pages.
    """

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    uid: str | None = Field(default=None, description="Optional taxonomy UID from source CSV")
    destination_url: HttpUrl = Field(..., description="Destination URL from taxonomy")
    english_page_name: str | None = Field(default=None, description="English page name")
    local_page_name: str | None = Field(default=None, description="Localized page name")
    content_type: str = Field(..., description="Content type/category for this taxonomy page")
    primary_audiance: str | None = Field(default=None, description="Primary audience")
    secondary_audiance: str | None = Field(default=None, description="Secondary audience")
    reference_source: str | None = Field(
        default=None,
        description="Canonical WordPress URL/path used for stage-0 matching",
    )
    species: list[str] = Field(default_factory=list, description="Target species list")
    semantic_summary: str = Field(..., description="Semantic summary of the page content")
    key_topics: list[str] = Field(default_factory=list, description="Key topics (keywords)")
    taxonomy_embedding: list[float] | None = Field(default=None)
    embedding_updated_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("species", mode="before")
    @classmethod
    def _parse_species(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            tokens = [str(token).strip() for token in value]
        else:
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if not token or token.lower() in {"n/a", "none"}:
                continue
            normalized = token
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned

    @field_validator("taxonomy_embedding", mode="before")
    @classmethod
    def _parse_taxonomy_embedding(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value


class CategorizationResult(BaseModel):
    """Result of content categorization."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    category: str
    batch_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class MatchingResult(BaseModel):
    """Result of content-to-taxonomy matching."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    taxonomy_id: UUID | None = Field(
        default=None, description="Final accepted taxonomy destination for this content"
    )
    semantic_taxonomy_id: UUID | None = Field(
        default=None,
        description="Best semantic candidate taxonomy regardless of acceptance",
    )
    semantic_similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score for the semantic candidate",
    )
    llm_topic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM rubric topic score when available",
    )
    match_stage: MatchStage | None = Field(
        default=None, description="Stage where match was determined"
    )
    failed_at_stage: str | None = Field(
        default=None, description="Stage where matching failed (for debugging)"
    )
    rubric: dict[str, Any] | None = Field(
        default=None, description="Rubric scores and decision from judge"
    )
    is_current: bool = Field(default=True, description="Whether this row is the active match")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime | None = None


class ExportRow(BaseModel):
    """CSV export row format."""

    model_config = ConfigDict(frozen=True)

    source_url: str
    target_url: str
    category: str
    similarity_score: float
    match_stage: str | None = None
    failed_at_stage: str | None = None


class WorkflowRunStatus(str, Enum):
    """Workflow run lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowRun(BaseModel):
    """Persisted workflow run metadata for resumable executions."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    run_key: str
    status: WorkflowRunStatus
    current_stage: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class BatchJobStatus(BaseModel):
    """OpenAI Batch API job status."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    batch_id: str
    status: str  # validating, failed, in_progress, finalizing, completed, expired, cancelled
    created_at: datetime
    completed_at: datetime | None = None
    request_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
