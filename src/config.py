"""Configuration management for the application."""

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase service role key")

    # Semantic Matching (Embeddings) Configuration
    semantic_api_key: str = Field(
        validation_alias=AliasChoices("SEMANTIC_API_KEY", "OPENAI_API_KEY"),
        description="API key for embeddings-based semantic matching (e.g., OpenRouter/OpenAI).",
    )
    semantic_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("SEMANTIC_BASE_URL", "OPENAI_BASE_URL"),
        description="API base URL for semantic matching provider.",
    )
    semantic_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("SEMANTIC_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL"),
        description="Embedding model used for semantic similarity.",
    )

    # LLM Categorization Configuration
    llm_api_key: str = Field(
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
        description="API key for LLM categorization (chat completions).",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
        description="API base URL for LLM categorization provider.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
        description="LLM model to use for categorization fallback.",
    )
    llm_batch_timeout: int = Field(
        default=86400,
        validation_alias=AliasChoices("LLM_BATCH_TIMEOUT", "OPENAI_BATCH_TIMEOUT"),
        description="Batch API timeout in seconds (24 hours default).",
    )

    # WordPress VIP Configuration
    wordpress_vip_sites: str = Field(..., description="Comma-separated list of WordPress sites")
    wordpress_vip_auth_token: str = Field(default="", description="Optional auth token")

    # Application Configuration
    taxonomy_file_path: Path = Field(
        default=Path("./data/taxonomy.csv"), description="Path to taxonomy CSV file"
    )
    batch_size: int = Field(default=1000, description="Batch processing size")
    log_level: str = Field(default="INFO", description="Logging level")
    ingestion_batch_size: int = Field(
        default=200,
        description="Rows per bulk upsert when ingesting content or taxonomy",
    )
    matching_batch_size: int = Field(
        default=200,
        description="Rows per bulk upsert when persisting matching results",
    )

    # Matching Configuration
    similarity_threshold: float = Field(
        default=0.85, description="Minimum similarity score for semantic matching"
    )
    semantic_candidate_limit: int = Field(
        default=25,
        description="Semantic neighbors fetched from Supabase pgvector per taxonomy row",
    )
    llm_candidate_limit: int = Field(
        default=10,
        description="Max semantic candidates forwarded to the LLM fallback stage",
    )
    llm_candidate_min_score: float = Field(
        default=0.6,
        description="Similarity floor for LLM fallback candidates",
    )
    # Rubric-gated LLM fallback (precision-first)
    llm_rubric_topic_min: float = Field(
        default=0.8, description="Minimum topic_alignment score (0-1) for LLM acceptance"
    )
    llm_rubric_intent_min: float = Field(
        default=0.8, description="Minimum intent_fit score (0-1) for LLM acceptance"
    )
    llm_rubric_entity_min: float = Field(
        default=0.5,
        description="Minimum entity_overlap score (0-1) when applicable (ignored if no entities)",
    )
    llm_consensus_votes: int = Field(
        default=1,
        description="Number of rubric evaluations to run; require majority accept (>=1 disables consensus)",
    )
    llm_match_temperature: float = Field(
        default=0.3, description="Temperature for LLM matching/rubric evaluation"
    )
    enable_semantic_matching: bool = Field(
        default=True, description="Enable semantic matching stage"
    )
    enable_llm_categorization: bool = Field(
        default=True, description="Enable LLM categorization fallback stage"
    )

    # DSPy Configuration
    dspy_optimization_metric: str = Field(
        default="accuracy", description="DSPy optimization metric"
    )
    dspy_num_trials: int = Field(default=50, description="Number of DSPy optimization trials")
    dspy_optimization_budget: str = Field(
        default="medium",
        description="GEPA optimization budget (light/medium/heavy)",
    )
    dspy_train_split_ratio: float = Field(
        default=0.2, description="Training set split ratio for prompt optimizers"
    )
    dspy_optimization_seed: int = Field(
        default=42, description="Random seed for DSPy optimization reproducibility"
    )
    dspy_reflection_model: str = Field(
        default="",
        description="Model for GEPA reflection (defaults to llm_model if empty)",
    )
    dspy_num_threads: int = Field(
        default=1, description="Parallel evaluation threads for DSPy optimization"
    )
    dspy_display_table: int = Field(
        default=5, description="Number of example results to display in evaluation (0 disables)"
    )
    dspy_max_full_evals: int | None = Field(
        default=None, description="Explicit GEPA budget: max full evaluations"
    )
    dspy_max_metric_calls: int | None = Field(
        default=None, description="Explicit GEPA budget: max metric calls"
    )

    @field_validator("wordpress_vip_sites", mode="before")
    @classmethod
    def parse_sites(cls, v: str) -> str:
        """Validate and return sites string."""
        if not v:
            raise ValueError("At least one WordPress site must be specified")
        return v

    def get_wordpress_sites(self) -> list[str]:
        """Parse and return list of WordPress sites."""
        sites = [site.strip() for site in self.wordpress_vip_sites.split(",")]
        return [site for site in sites if site]

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @field_validator("llm_candidate_min_score")
    @classmethod
    def validate_candidate_floor(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("LLM candidate minimum score must be between 0 and 1")
        return v

    @field_validator("llm_rubric_topic_min", "llm_rubric_intent_min", "llm_rubric_entity_min")
    @classmethod
    def validate_rubric_thresholds(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Rubric thresholds must be between 0 and 1")
        return v

    @field_validator("llm_consensus_votes")
    @classmethod
    def validate_consensus_votes(cls, v: int) -> int:
        if v < 1:
            raise ValueError("llm_consensus_votes must be >= 1")
        return v

    @field_validator("llm_match_temperature")
    @classmethod
    def validate_match_temperature(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("llm_match_temperature must be between 0 and 1")
        return v

    @field_validator("dspy_train_split_ratio")
    @classmethod
    def validate_train_split(cls, v: float) -> float:
        """Validate training split ratio is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Training split ratio must be between 0 and 1")
        return v

    @field_validator("dspy_optimization_budget")
    @classmethod
    def validate_budget(cls, v: str) -> str:
        """Validate GEPA budget is one of the allowed values."""
        if v not in ("light", "medium", "heavy"):
            raise ValueError("GEPA budget must be one of: light, medium, heavy")
        return v

    @field_validator(
        "semantic_candidate_limit",
        "llm_candidate_limit",
        "ingestion_batch_size",
        "matching_batch_size",
    )
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()  # type: ignore[call-arg]
