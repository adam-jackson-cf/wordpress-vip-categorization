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

    # Matching Configuration
    similarity_threshold: float = Field(
        default=0.85, description="Minimum similarity score for semantic matching"
    )
    llm_confidence_threshold: float = Field(
        default=0.9, description="Minimum confidence score for LLM categorization"
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

    @field_validator("llm_confidence_threshold")
    @classmethod
    def validate_llm_threshold(cls, v: float) -> float:
        """Validate LLM confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("LLM confidence threshold must be between 0 and 1")
        return v


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()  # type: ignore[call-arg]
