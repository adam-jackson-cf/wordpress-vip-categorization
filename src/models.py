"""Pydantic models for data validation and serialization."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


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
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CategorizationResult(BaseModel):
    """Result of content categorization."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_schema_serialization_defaults_required=True,
    )

    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExportRow(BaseModel):
    """CSV export row format."""

    model_config = ConfigDict(frozen=True)

    source_url: str
    target_url: str
    confidence: float
    category: str
    similarity_score: float


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
