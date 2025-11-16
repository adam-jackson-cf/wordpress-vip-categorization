"""Pydantic models for data validation and serialization."""

import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class MatchStage(str, Enum):
    """Stages in the matching workflow."""

    SEMANTIC_MATCHED = "semantic_matched"
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
    content_embedding: list[float] | None = Field(default=None)
    embedding_updated_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    url: HttpUrl = Field(..., description="Source URL from taxonomy")
    category: str = Field(..., description="Category for this taxonomy page")
    description: str = Field(..., description="Description of the page content/purpose")
    keywords: list[str] = Field(
        default_factory=list, description="Keywords and phrases to match against"
    )
    taxonomy_embedding: list[float] | None = Field(default=None)
    embedding_updated_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MatchingResult(BaseModel):
    """Result of semantic matching."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    taxonomy_id: UUID
    content_id: UUID | None = None
    similarity_score: float = Field(ge=0.0, le=1.0)
    candidate_content_id: UUID | None = Field(
        default=None, description="Best semantic candidate even if below threshold"
    )
    candidate_similarity_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity score for candidate_content_id",
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    started_at: datetime = Field(default_factory=datetime.utcnow)
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
