"""Unit tests for DSPy optimizer."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dspy
import pytest

from src.optimization.dspy_optimizer import DSPyOptimizer, MatchingModule


@pytest.fixture
def sample_dspy_example() -> dspy.Example:
    """Create a sample DSPy example."""
    return dspy.Example(
        taxonomy_category="Technology",
        taxonomy_description="Technology-related content",
        taxonomy_keywords="technology, innovation",
        content_summaries="0. Title: Tech Post\n   URL: https://example.com/tech\n   Preview: Content...",
        best_match_index=0,
        reasoning="Good match",
    ).with_inputs(
        "taxonomy_category",
        "taxonomy_description",
        "taxonomy_keywords",
        "content_summaries",
    )


@pytest.fixture
def mock_dspy_optimizer(mock_settings, mock_supabase_client) -> DSPyOptimizer:
    """Create a DSPy optimizer instance with mocked dependencies."""
    with patch("src.optimization.dspy_optimizer.dspy.LM") as mock_lm:
        mock_lm.return_value = Mock()
        with patch("src.optimization.dspy_optimizer.dspy.configure"):
            optimizer = DSPyOptimizer(mock_settings, mock_supabase_client)
            return optimizer


class TestLoadTrainingDataset:
    """Tests for load_training_dataset method."""

    def test_load_csv_dataset_success(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading a valid CSV dataset."""
        csv_file = tmp_path / "dataset.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "taxonomy_category",
                    "taxonomy_description",
                    "taxonomy_keywords",
                    "content_summaries",
                    "best_match_index",
                    "reasoning",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "taxonomy_category": "Technology",
                    "taxonomy_description": "Tech content",
                    "taxonomy_keywords": "tech, innovation",
                    "content_summaries": "0. Title: Post\n   URL: https://example.com\n   Preview: Content...",
                    "best_match_index": "0",
                    "reasoning": "Good match",
                }
            )

        examples = mock_dspy_optimizer.load_training_dataset(csv_file)

        assert len(examples) == 1
        assert examples[0].taxonomy_category == "Technology"
        assert examples[0].best_match_index == 0
        # Confidence is no longer emitted; ensure attribute is absent
        assert not hasattr(examples[0], "confidence")

    def test_load_json_dataset_success(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading a valid JSON dataset."""
        json_file = tmp_path / "dataset.json"
        data = [
            {
                "taxonomy_category": "Technology",
                "taxonomy_description": "Tech content",
                "taxonomy_keywords": "tech, innovation",
                "content_summaries": "0. Title: Post\n   URL: https://example.com\n   Preview: Content...",
                "best_match_index": 0,
                "reasoning": "Good match",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        examples = mock_dspy_optimizer.load_training_dataset(json_file)

        assert len(examples) == 1
        assert examples[0].taxonomy_category == "Technology"
        assert examples[0].best_match_index == 0

    def test_load_dataset_file_not_found(self, mock_dspy_optimizer) -> None:
        """Test loading a non-existent dataset file."""
        with pytest.raises(FileNotFoundError):
            mock_dspy_optimizer.load_training_dataset(Path("nonexistent.csv"))

    def test_load_dataset_invalid_format(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading a dataset with unsupported format."""
        invalid_file = tmp_path / "dataset.txt"
        invalid_file.write_text("invalid format")

        with pytest.raises(ValueError, match="Unsupported file format"):
            mock_dspy_optimizer.load_training_dataset(invalid_file)

    def test_load_csv_missing_required_columns(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading CSV with missing required columns."""
        csv_file = tmp_path / "dataset.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["taxonomy_category"])
            writer.writeheader()
            writer.writerow({"taxonomy_category": "Tech"})

        with pytest.raises(ValueError, match="CSV must contain columns"):
            mock_dspy_optimizer.load_training_dataset(csv_file)

    def test_load_json_invalid_structure(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading JSON with invalid structure."""
        json_file = tmp_path / "dataset.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="JSON dataset must be a list"):
            mock_dspy_optimizer.load_training_dataset(json_file)

    def test_load_dataset_skips_invalid_rows(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test that invalid rows are skipped with warnings."""
        csv_file = tmp_path / "dataset.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "taxonomy_category",
                    "taxonomy_description",
                    "content_summaries",
                    "best_match_index",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "taxonomy_category": "Tech",
                    "taxonomy_description": "Description",
                    "content_summaries": "Summaries",
                    "best_match_index": "invalid",  # Invalid integer
                }
            )
            writer.writerow(
                {
                    "taxonomy_category": "Tech2",
                    "taxonomy_description": "Description2",
                    "content_summaries": "Summaries2",
                    "best_match_index": "1",
                }
            )

        examples = mock_dspy_optimizer.load_training_dataset(csv_file)

        # Should only have one valid example
        assert len(examples) == 1
        assert examples[0].taxonomy_category == "Tech2"

    def test_load_dataset_empty_file(self, mock_dspy_optimizer, tmp_path: Path) -> None:
        """Test loading an empty dataset file."""
        csv_file = tmp_path / "dataset.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "taxonomy_category",
                    "taxonomy_description",
                    "content_summaries",
                    "best_match_index",
                ],
            )
            writer.writeheader()

        with pytest.raises(ValueError, match="No valid examples found"):
            mock_dspy_optimizer.load_training_dataset(csv_file)


class TestMetricSelection:
    """Tests for configurable optimization metrics."""

    def test_strict_accuracy_requires_accept_decision(
        self,
        mock_settings,
        mock_supabase_client,
        sample_dspy_example,
    ) -> None:
        """Strict accuracy should only score when decision is ACCEPT."""
        mock_settings.dspy_optimization_metric = "strict_accuracy"
        with patch("src.optimization.dspy_optimizer.dspy.LM") as mock_lm:
            mock_lm.return_value = Mock()
            with patch("src.optimization.dspy_optimizer.dspy.configure"):
                optimizer = DSPyOptimizer(mock_settings, mock_supabase_client)

        accept_pred = SimpleNamespace(best_match_index=0, decision="accept")
        reject_pred = SimpleNamespace(best_match_index=0, decision="reject")

        assert optimizer.metric_fn(sample_dspy_example, accept_pred) == 1.0
        assert optimizer.metric_fn(sample_dspy_example, reject_pred) == 0.0

    def test_confidence_weighted_metric_multiplies_rubric_scores(
        self,
        mock_settings,
        mock_supabase_client,
        sample_dspy_example,
    ) -> None:
        """Confidence-weighted accuracy should use rubric averages."""
        mock_settings.dspy_optimization_metric = "confidence_weighted"
        with patch("src.optimization.dspy_optimizer.dspy.LM") as mock_lm:
            mock_lm.return_value = Mock()
            with patch("src.optimization.dspy_optimizer.dspy.configure"):
                optimizer = DSPyOptimizer(mock_settings, mock_supabase_client)

        pred = SimpleNamespace(
            best_match_index=0,
            topic_alignment=0.9,
            intent_fit=0.6,
            entity_overlap=0.3,
        )
        expected = (0.9 + 0.6 + 0.3) / 3
        assert optimizer.metric_fn(sample_dspy_example, pred) == pytest.approx(expected, rel=1e-6)

    def test_unknown_metric_falls_back_to_accuracy(
        self,
        mock_settings,
        mock_supabase_client,
        sample_dspy_example,
    ) -> None:
        """Unsupported metric strings should fall back gracefully."""
        mock_settings.dspy_optimization_metric = "totally-unknown"
        with patch("src.optimization.dspy_optimizer.dspy.LM") as mock_lm:
            mock_lm.return_value = Mock()
            with patch("src.optimization.dspy_optimizer.dspy.configure"):
                optimizer = DSPyOptimizer(mock_settings, mock_supabase_client)

        pred = SimpleNamespace(best_match_index=0)
        assert optimizer.metric_fn(sample_dspy_example, pred) == optimizer.accuracy_metric(
            sample_dspy_example, pred
        )


class TestOptimizeWithGEPA:
    """Tests for optimize_with_gepa method."""

    def test_optimize_with_gepa_no_examples(self, mock_dspy_optimizer) -> None:
        """Test GEPA optimization with no training examples."""
        result = mock_dspy_optimizer.optimize_with_gepa([])

        assert result == mock_dspy_optimizer.matcher

    @patch("src.optimization.dspy_optimizer.GEPA")
    @patch("dspy.evaluate.Evaluate")
    @patch("src.optimization.dspy_optimizer.dspy.LM")
    def test_optimize_with_gepa_success(
        self,
        mock_lm,
        mock_evaluate,
        mock_gepa,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test successful GEPA optimization."""
        # Setup mocks
        mock_optimizer_instance = Mock()
        mock_optimized_module = Mock(spec=MatchingModule)
        mock_optimizer_instance.compile.return_value = mock_optimized_module
        mock_gepa.return_value = mock_optimizer_instance

        mock_evaluator = Mock()
        mock_evaluator.return_value = 0.85
        mock_evaluate.return_value = mock_evaluator

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_gepa(training_examples, budget="light")

        # Successful optimization should return the compiled matcher instance
        assert result == mock_optimized_module
        mock_gepa.assert_called_once()

    @patch("src.optimization.dspy_optimizer.GEPA")
    @patch("src.optimization.dspy_optimizer.dspy.LM")
    def test_optimize_with_gepa_budget_config(
        self,
        mock_lm,
        mock_gepa,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test GEPA optimization with different budget configurations."""
        mock_optimizer_instance = Mock()
        mock_optimized_module = Mock(spec=MatchingModule)
        mock_optimizer_instance.compile.return_value = mock_optimized_module
        mock_gepa.return_value = mock_optimizer_instance

        training_examples = [sample_dspy_example] * 10

        # Test with auto budget
        mock_dspy_optimizer.optimize_with_gepa(training_examples, budget="medium")
        call_kwargs = mock_gepa.call_args[1]
        assert call_kwargs["auto"] == "medium"

        # Test with max_full_evals
        mock_dspy_optimizer.optimize_with_gepa(training_examples, max_full_evals=10)
        call_kwargs = mock_gepa.call_args[1]
        assert call_kwargs["max_full_evals"] == 10

        # Test with max_metric_calls
        mock_dspy_optimizer.optimize_with_gepa(training_examples, max_metric_calls=1000)
        call_kwargs = mock_gepa.call_args[1]
        assert call_kwargs["max_metric_calls"] == 1000

    @patch("src.optimization.dspy_optimizer.GEPA")
    @patch("dspy.evaluate.Evaluate")
    @patch("src.optimization.dspy_optimizer.dspy.LM")
    def test_optimize_with_gepa_handles_exception(
        self,
        mock_lm,
        mock_evaluate,
        mock_gepa,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test GEPA optimization handles exceptions gracefully."""
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.compile.side_effect = Exception("Optimization failed")
        mock_gepa.return_value = mock_optimizer_instance

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_gepa(training_examples)

        # Should return unoptimized model on failure
        assert result == mock_dspy_optimizer.matcher


class TestOptimizeWithDataset:
    """Tests for optimize_with_dataset method."""

    def test_optimize_with_dataset_no_examples(self, mock_dspy_optimizer) -> None:
        """Test dataset optimization with no training examples."""
        result = mock_dspy_optimizer.optimize_with_dataset([])

        assert result == mock_dspy_optimizer.matcher

    @patch("src.optimization.dspy_optimizer.DSPyOptimizer.optimize_with_gepa")
    def test_optimize_with_dataset_gepa(
        self,
        mock_gepa,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test dataset optimization with GEPA optimizer."""
        mock_optimized = Mock(spec=MatchingModule)
        mock_gepa.return_value = mock_optimized

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_dataset(training_examples, optimizer_type="gepa")

        assert result == mock_optimized
        mock_gepa.assert_called_once()

    @patch("dspy.teleprompt.BootstrapFewShotWithRandomSearch")
    @patch("dspy.evaluate.Evaluate")
    def test_optimize_with_dataset_bootstrap_random_search(
        self,
        mock_evaluate,
        mock_bootstrap_rs,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test dataset optimization with BootstrapFewShotWithRandomSearch."""
        mock_optimizer_instance = Mock()
        mock_optimized_module = Mock(spec=MatchingModule)
        mock_optimizer_instance.compile.return_value = mock_optimized_module
        mock_bootstrap_rs.return_value = mock_optimizer_instance

        mock_evaluator = Mock()
        mock_evaluator.return_value = 0.80
        mock_evaluate.return_value = mock_evaluator

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_dataset(
            training_examples, optimizer_type="bootstrap-random-search"
        )

        assert result == mock_optimized_module
        mock_bootstrap_rs.assert_called_once()
        call_kwargs = mock_bootstrap_rs.call_args.kwargs
        assert call_kwargs["num_candidate_programs"] == mock_dspy_optimizer.settings.dspy_num_trials

    @patch("dspy.teleprompt.BootstrapFewShotWithRandomSearch")
    @patch("dspy.evaluate.Evaluate")
    def test_bootstrap_random_search_respects_num_trials_override(
        self,
        mock_evaluate,
        mock_bootstrap_rs,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """The num_trials kwarg should override default candidate count."""
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.compile.return_value = Mock(spec=MatchingModule)
        mock_bootstrap_rs.return_value = mock_optimizer_instance

        mock_evaluator = Mock()
        mock_evaluator.return_value = 0.75
        mock_evaluate.return_value = mock_evaluator

        training_examples = [sample_dspy_example] * 12

        mock_dspy_optimizer.optimize_with_dataset(
            training_examples, optimizer_type="bootstrap-random-search", num_trials=7
        )

        call_kwargs = mock_bootstrap_rs.call_args.kwargs
        assert call_kwargs["num_candidate_programs"] == 7

    @patch("src.optimization.dspy_optimizer.DSPyOptimizer.optimize")
    def test_optimize_with_dataset_bootstrap(
        self,
        mock_optimize,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test dataset optimization with BootstrapFewShot optimizer."""
        mock_optimized = Mock(spec=MatchingModule)
        mock_optimize.return_value = mock_optimized

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_dataset(
            training_examples, optimizer_type="bootstrap"
        )

        assert result == mock_optimized
        mock_optimize.assert_called_once()

    @patch("dspy.teleprompt.BootstrapFewShotWithRandomSearch")
    @patch("dspy.evaluate.Evaluate")
    def test_optimize_with_dataset_handles_exception(
        self,
        mock_evaluate,
        mock_bootstrap_rs,
        mock_dspy_optimizer,
        sample_dspy_example,
    ) -> None:
        """Test dataset optimization handles exceptions gracefully."""
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.compile.side_effect = Exception("Optimization failed")
        mock_bootstrap_rs.return_value = mock_optimizer_instance

        training_examples = [sample_dspy_example] * 10

        result = mock_dspy_optimizer.optimize_with_dataset(
            training_examples, optimizer_type="bootstrap-random-search"
        )

        # Should return unoptimized model on failure
        assert result == mock_dspy_optimizer.matcher
