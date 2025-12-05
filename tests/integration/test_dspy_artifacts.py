"""Integration tests that load real DSPy artifacts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.optimization.dspy_optimizer import DSPyOptimizer

ARTIFACT_PATH = Path("prompt-optimiser/models/matcher_v3.json")


@pytest.mark.integration
@pytest.mark.skipif(
    not ARTIFACT_PATH.exists(), reason="matcher_v3.json must exist to validate legacy demos"
)
def test_prompt_context_includes_demonstrations_from_versioned_artifact(
    mock_settings,
    mock_supabase_client,
) -> None:
    """Ensure legacy matcher artifacts still yield prompt context demos."""

    with open(ARTIFACT_PATH, encoding="utf-8") as f:
        artifact = json.load(f)

    demos_data = artifact.get("predict.predict", {}).get("demos", [])
    assert demos_data, "Artifact must contain serialized demos"

    demo_namespaces = [SimpleNamespace(**demo) for demo in demos_data]
    legacy_model = SimpleNamespace(
        predict=SimpleNamespace(instructions=None, demos=demo_namespaces)
    )

    with patch("src.optimization.dspy_optimizer.dspy.LM") as mock_lm:
        mock_lm.return_value = Mock()
        with patch("src.optimization.dspy_optimizer.dspy.configure"):
            optimizer = DSPyOptimizer(mock_settings, mock_supabase_client)

    original_matcher = optimizer.matcher
    optimizer.matcher = legacy_model  # type: ignore[assignment]
    try:
        context = optimizer.get_prompt_context()
    finally:
        optimizer.matcher = original_matcher

    assert context.demonstrations, "Loaded artifact should provide serialized demonstrations"
    assert "taxonomy_content_type" in context.demonstrations[0]
