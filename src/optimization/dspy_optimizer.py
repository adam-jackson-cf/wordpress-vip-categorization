"""DSPy-based prompt optimization for taxonomy-to-content matching."""

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import dspy
from dspy import evaluate as dspy_evaluate
from dspy import teleprompt as dspy_teleprompt
from dspy.teleprompt import GEPA, BootstrapFewShot

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchStage, TaxonomyPage, WordPressContent

logger = logging.getLogger(__name__)


PROMPT_OPT_DIR = Path("prompt-optimiser")
MODELS_DIR = PROMPT_OPT_DIR / "models"
ACTIVE_MODEL_PATH = MODELS_DIR / "matcher_latest.json"
CONFIGS_DIR = PROMPT_OPT_DIR / "configs"
REPORTS_DIR = PROMPT_OPT_DIR / "reports"
GEPA_LOG_DIR = PROMPT_OPT_DIR / "gepa_logs"


class TaxonomyMatcher(dspy.Signature):
    """Match a taxonomy page to the most relevant content page with rubric scoring."""

    taxonomy_category: str = dspy.InputField(desc="Category of the taxonomy page")
    taxonomy_description: str = dspy.InputField(desc="Description of the taxonomy page")
    taxonomy_keywords: str = dspy.InputField(
        desc="Keywords for the taxonomy page (comma-separated)"
    )
    content_summaries: str = dspy.InputField(
        desc="List of available content pages with index, title, URL, and preview"
    )
    best_match_index: int = dspy.OutputField(
        desc="Index of the best matching content page, or -1 if no good match"
    )
    topic_alignment: float = dspy.OutputField(
        desc="Topical alignment score 0-1 (coverage of taxonomy topic)"
    )
    intent_fit: float = dspy.OutputField(
        desc="Intent alignment score 0-1 (does the content's purpose match taxonomy intent)"
    )
    entity_overlap: float = dspy.OutputField(
        desc="Named entity/keyword overlap score 0-1 (0 if none are present)"
    )
    temporal_relevance: float = dspy.OutputField(
        desc="Temporal relevance score 0-1 (use 0 if not applicable)"
    )
    decision: str = dspy.OutputField(desc="One of: accept, reject, abstain")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this decision was chosen")


class MatchingModule(dspy.Module):
    """DSPy module for taxonomy-to-content matching."""

    def __init__(self) -> None:
        """Initialize matching module."""
        super().__init__()
        self.predict = dspy.ChainOfThought(TaxonomyMatcher)

    def forward(
        self,
        taxonomy_category: str,
        taxonomy_description: str,
        taxonomy_keywords: str,
        content_summaries: str,
    ) -> dspy.Prediction:
        """Forward pass for matching.

        Args:
            taxonomy_category: Category of the taxonomy page.
            taxonomy_description: Description of the taxonomy page.
            taxonomy_keywords: Keywords for the taxonomy page.
            content_summaries: Formatted list of content pages.

        Returns:
            Prediction with best_match_index and rubric fields.
        """
        return self.predict(
            taxonomy_category=taxonomy_category,
            taxonomy_description=taxonomy_description,
            taxonomy_keywords=taxonomy_keywords,
            content_summaries=content_summaries,
        )


class DSPyOptimizer:
    """Optimizer for matching prompts using DSPy."""

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize DSPy optimizer.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client

        # Configure DSPy with OpenAI (or OpenAI-compatible API)
        # Use dspy.LM() per updated best practices (both dspy.LM() and dspy.clients.LM are valid)
        # Determine provider from base URL or default to openai
        model_name = settings.llm_model
        if "openrouter" in settings.llm_base_url.lower():
            # OpenRouter format: openrouter/model-name
            if not model_name.startswith("openrouter/"):
                model_name = f"openrouter/{model_name}"
        elif not model_name.startswith("openai/"):
            # Default to openai provider
            model_name = f"openai/{model_name}"

        lm = dspy.LM(
            model=model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=getattr(settings, "llm_match_temperature", 0.3),
        )
        dspy.configure(lm=lm)

        self.matcher = MatchingModule()
        logger.info(f"Initialized DSPy optimizer with model: {settings.llm_model}")

    def _format_content_summaries(self, content_items: list[WordPressContent]) -> str:
        """Format content items as indexed summaries for DSPy input.

        Args:
            content_items: List of content items to format.

        Returns:
            Formatted string with indexed content summaries.
        """
        summaries = []
        for i, content in enumerate(content_items):
            summary = f"{i}. Title: {content.title}\n   URL: {content.url}\n   Preview: {content.content[:200]}..."
            summaries.append(summary)
        return "\n\n".join(summaries)

    def prepare_training_data(
        self, content_items: list[WordPressContent] | None = None
    ) -> list[dspy.Example]:
        """Prepare training data from matching results in database.

        Args:
            content_items: Optional list of all content items. If None, loads from database.

        Returns:
            List of DSPy examples for matching task.
        """
        if content_items is None:
            content_items = self.db.get_all_content()

        # Get all matching results
        all_matchings = self.db.get_all_matchings()

        # Filter for successful matches (LLM_CATEGORIZED or SEMANTIC_MATCHED with content_id)
        successful_matchings = [
            m
            for m in all_matchings
            if m.content_id is not None
            and m.match_stage in (MatchStage.LLM_CATEGORIZED, MatchStage.SEMANTIC_MATCHED)
        ]

        if not successful_matchings:
            logger.warning("No successful matching results found for training data")
            return []

        examples = []

        for matching in successful_matchings:
            try:
                # Load taxonomy page
                taxonomy = self.db.get_taxonomy_by_id(matching.taxonomy_id)
                if not taxonomy:
                    logger.warning(f"Taxonomy {matching.taxonomy_id} not found, skipping")
                    continue

                # Find the matched content item
                matched_content = next(
                    (c for c in content_items if c.id == matching.content_id), None
                )
                if not matched_content:
                    logger.warning(
                        f"Content {matching.content_id} not found for matching {matching.id}, skipping"
                    )
                    continue

                # Find index of matched content in the list
                try:
                    best_match_index = content_items.index(matched_content)
                except ValueError:
                    logger.warning(
                        f"Matched content {matching.content_id} not in content_items list, skipping"
                    )
                    continue

                # Format content summaries
                content_summaries = self._format_content_summaries(content_items)

                # Format keywords
                keywords_str = ", ".join(taxonomy.keywords) if taxonomy.keywords else "None"

                # Create DSPy example
                example = dspy.Example(
                    taxonomy_category=taxonomy.category,
                    taxonomy_description=taxonomy.description,
                    taxonomy_keywords=keywords_str,
                    content_summaries=content_summaries,
                    best_match_index=best_match_index,
                    reasoning="",  # Not stored in database, will be generated by model
                ).with_inputs(
                    "taxonomy_category",
                    "taxonomy_description",
                    "taxonomy_keywords",
                    "content_summaries",
                )

                examples.append(example)

            except Exception as e:
                logger.error(f"Error preparing training example for matching {matching.id}: {e}")
                continue

        logger.info(f"Prepared {len(examples)} training examples from matching results")
        return examples

    def accuracy_metric(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> float:
        """Compute accuracy metric for matching evaluation.

        GEPA-compatible metric that accepts five arguments as required by DSPy GEPA optimizer.

        Args:
            gold: Ground truth example.
            pred: Model prediction.
            trace: Optional trace information (unused).
            pred_name: Optional prediction name (unused).
            pred_trace: Optional prediction trace (unused).

        Returns:
            Score between 0 and 1.
        """
        # Check if best_match_index matches
        try:
            expected_index = int(gold.best_match_index)
            predicted_index = (
                int(pred.best_match_index) if hasattr(pred, "best_match_index") else -1
            )
        except (ValueError, TypeError, AttributeError):
            return 0.0

        # Exact match on index
        index_match = 1.0 if predicted_index == expected_index else 0.0

        return index_match

    def _extract_prompt_info(
        self, model: MatchingModule
    ) -> dict[str, str | list[dict[str, str | int | float]]]:
        """Extract prompt/instruction information and demonstrations from a DSPy module.

        Args:
            model: DSPy module to extract prompts from.

        Returns:
            Dictionary with prompt information including demonstrations.
        """
        info: dict[str, str | list[dict[str, str | int | float]]] = {}
        try:
            # Try to extract instructions from the predict module
            if hasattr(model, "predict") and hasattr(model.predict, "instructions"):
                info["instructions"] = str(model.predict.instructions)
            elif hasattr(model, "predict") and hasattr(model.predict, "_signature"):
                # Fallback: get signature description
                sig = model.predict._signature
                if hasattr(sig, "__doc__") and sig.__doc__:
                    info["signature_description"] = sig.__doc__

            # Extract demonstrations/examples
            demonstrations: list[dict[str, str | int | float]] = []
            num_demos = 0

            if hasattr(model, "predict") and hasattr(model.predict, "demos"):
                demos = model.predict.demos
                if demos:
                    num_demos = len(demos)
                    # Serialize demonstrations to dict format
                    for demo in demos:
                        try:
                            demo_dict: dict[str, str | int | float] = {}
                            # Extract fields from dspy.Example
                            if hasattr(demo, "taxonomy_category"):
                                demo_dict["taxonomy_category"] = str(demo.taxonomy_category)
                            if hasattr(demo, "taxonomy_description"):
                                demo_dict["taxonomy_description"] = str(demo.taxonomy_description)
                            if hasattr(demo, "taxonomy_keywords"):
                                demo_dict["taxonomy_keywords"] = str(demo.taxonomy_keywords)
                            if hasattr(demo, "content_summaries"):
                                demo_dict["content_summaries"] = str(demo.content_summaries)
                            if hasattr(demo, "best_match_index"):
                                try:
                                    demo_dict["best_match_index"] = int(demo.best_match_index)
                                except (ValueError, TypeError):
                                    demo_dict["best_match_index"] = -1
                            if hasattr(demo, "reasoning"):
                                demo_dict["reasoning"] = str(demo.reasoning)
                            demonstrations.append(demo_dict)
                        except Exception as e:
                            logger.warning(f"Could not serialize demonstration: {e}")
                            continue

            info["num_demos"] = str(num_demos)
            info["demonstrations"] = demonstrations

        except Exception as e:
            logger.warning(f"Could not extract prompt info: {e}")
            info["error"] = str(e)
            info["num_demos"] = "0"
            info["demonstrations"] = []

        return info

    def optimize(
        self,
        training_examples: list[dspy.Example],
        validation_examples: list[dspy.Example] | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
    ) -> MatchingModule:
        """Optimize matcher using few-shot learning.

        Args:
            training_examples: Training data.
            validation_examples: Optional validation data.
            max_bootstrapped_demos: Max bootstrapped demonstrations.
            max_labeled_demos: Max labeled demonstrations.

        Returns:
            Optimized matching module.
        """
        if not training_examples:
            logger.warning("No training examples provided, returning unoptimized model")
            return self.matcher

        # Use validation set if provided, otherwise split training set
        if validation_examples is None:
            split_point = int(len(training_examples) * self.settings.dspy_train_split_ratio)
            train_set = training_examples[:split_point]
            val_set = training_examples[split_point:]
        else:
            train_set = training_examples
            val_set = validation_examples

        logger.info(
            f"Optimizing with {len(train_set)} training, {len(val_set)} validation examples"
        )

        # Set up optimizer
        optimizer = BootstrapFewShot(
            metric=self.accuracy_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )

        # Compile/optimize the model
        try:
            optimized = optimizer.compile(
                self.matcher,
                trainset=train_set,
            )

            # Evaluate on validation set for basic bootstrap optimizer
            evaluator = dspy_evaluate.Evaluate(
                devset=val_set,
                metric=self.accuracy_metric,
                num_threads=1,
                display_progress=True,
            )

            eval_result = evaluator(optimized)
            # Extract score from EvaluationResult object
            if hasattr(eval_result, "score"):
                score = float(eval_result.score)
            elif hasattr(eval_result, "average"):
                score = float(eval_result.average)
            elif isinstance(eval_result, (int, float)):
                score = float(eval_result)
            else:
                # Try to convert to float directly
                try:
                    score = float(eval_result)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not extract score from evaluation result: {type(eval_result)}"
                    )
                    score = 0.0
            logger.info(f"Optimized model validation score: {score:.3f}")

            return cast(MatchingModule, optimized)

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.matcher

    def generate_optimization_report(
        self,
        before_model: MatchingModule,
        after_model: MatchingModule,
        optimizer_type: str,
        optimizer_config: dict[str, str | int | float | None],
        training_size: int,
        validation_size: int,
        validation_score: float | None = None,
        duration_seconds: float | None = None,
    ) -> str:
        """Generate a markdown report documenting optimization changes.

        Args:
            before_model: Model before optimization.
            after_model: Model after optimization.
            optimizer_type: Type of optimizer used.
            optimizer_config: Configuration used for optimization.
            training_size: Number of training examples.
            validation_size: Number of validation examples.
            validation_score: Final validation score (if available).
            duration_seconds: Optimization duration in seconds (if available).

        Returns:
            Markdown report as string.
        """
        before_info = self._extract_prompt_info(before_model)
        after_info = self._extract_prompt_info(after_model)

        report_lines = [
            "# DSPy Optimization Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Optimizer:** {optimizer_type}",
            f"- **Training Examples:** {training_size}",
            f"- **Validation Examples:** {validation_size}",
        ]

        if validation_score is not None:
            report_lines.append(f"- **Validation Score:** {validation_score:.3f}")
        if duration_seconds is not None:
            report_lines.append(f"- **Duration:** {duration_seconds:.1f} seconds")

        report_lines.extend(
            [
                "",
                "## Configuration",
                "",
                "### Optimizer Settings",
            ]
        )

        for key, value in optimizer_config.items():
            if value is not None:
                report_lines.append(f"- **{key}:** {value}")

        report_lines.extend(
            [
                "",
                "### Model Settings",
                f"- **LLM Model:** {self.settings.llm_model}",
                f"- **LLM Base URL:** {self.settings.llm_base_url}",
                "- **Temperature:** 0.3 (main), 1.0 (reflection for GEPA)",
                "",
                "## Before Optimization",
                "",
                "### Prompt Instructions",
            ]
        )

        if "instructions" in before_info:
            report_lines.append("```")
            instructions = before_info["instructions"]
            if isinstance(instructions, str):
                report_lines.append(instructions)
            report_lines.append("```")
        elif "signature_description" in before_info:
            sig_desc = before_info["signature_description"]
            if isinstance(sig_desc, str):
                report_lines.append(sig_desc)
        else:
            report_lines.append("*Default DSPy ChainOfThought instructions*")

        report_lines.extend(
            [
                "",
                f"- **Number of Demonstrations:** {before_info.get('num_demos', '0')}",
                "",
                "## After Optimization",
                "",
                "### Prompt Instructions",
            ]
        )

        if "instructions" in after_info:
            report_lines.append("```")
            instructions = after_info["instructions"]
            if isinstance(instructions, str):
                report_lines.append(instructions)
            report_lines.append("```")
        elif "signature_description" in after_info:
            sig_desc = after_info["signature_description"]
            if isinstance(sig_desc, str):
                report_lines.append(sig_desc)
        else:
            report_lines.append("*Optimized DSPy ChainOfThought instructions*")

        report_lines.extend(
            [
                "",
                f"- **Number of Demonstrations:** {after_info.get('num_demos', '0')}",
                "",
                "## Changes",
                "",
            ]
        )

        # Compare before and after
        before_num_demos_str = before_info.get("num_demos", "0")
        after_num_demos_str = after_info.get("num_demos", "0")
        if before_num_demos_str != after_num_demos_str:
            report_lines.append(
                f"- **Demonstrations:** Changed from {before_num_demos_str} to {after_num_demos_str}"
            )

        before_instr = before_info.get("instructions")
        after_instr = after_info.get("instructions")
        if (
            before_instr
            and after_instr
            and isinstance(before_instr, str)
            and isinstance(after_instr, str)
        ):
            if before_instr != after_instr:
                report_lines.append("- **Instructions:** Modified during optimization")
            else:
                report_lines.append(
                    "- **Instructions:** No changes detected (may be in demonstrations)"
                )
        else:
            report_lines.append(
                "- **Instructions:** Changes may be in demonstrations or internal structure"
            )

        report_lines.extend(
            [
                "",
                "## Notes",
                "",
                "- DSPy optimizers may modify instructions, add demonstrations, or adjust internal parameters",
                "- The optimized model is saved and can be loaded for production use",
                "- Validation score indicates performance on held-out data",
            ]
        )

        return "\n".join(report_lines)

    def _get_next_version(self, model_file_path: str) -> int:
        """Get the next version number by scanning existing config files.

        Args:
            model_file_path: Unused; kept for backward compatibility.

        Returns:
            Next version number (starts at 1 if no existing configs).
        """
        del model_file_path  # Parameter retained for compatibility, not used.

        config_dir = CONFIGS_DIR
        config_pattern = "dspy_config_v*.json"

        max_version = 0
        try:
            for config_file in config_dir.glob(config_pattern):
                # Extract version number from filename: dspy_config_v{version}.json
                # or dspy_config_v{version}_stage{N}.json
                try:
                    version_str = config_file.stem.replace("dspy_config_v", "")
                    # Remove stage suffix if present (e.g., "_stage1" or "_stage2")
                    if "_stage" in version_str:
                        version_str = version_str.split("_stage")[0]
                    version = int(version_str)
                    max_version = max(max_version, version)
                except (ValueError, AttributeError):
                    # Skip invalid filenames
                    continue
        except Exception as e:
            logger.warning(f"Error scanning for existing config files: {e}")

        return max_version + 1

    def extract_optimization_config(
        self,
        before_model: MatchingModule,
        after_model: MatchingModule,
        optimizer_type: str,
        optimizer_config: dict[str, str | int | float | None],
        training_size: int,
        validation_size: int,
        model_file_path: str,
        validation_score: float | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, str | int | float | dict | list]:
        """Extract full optimization configuration for JSON serialization.

        Args:
            before_model: Model before optimization.
            after_model: Model after optimization.
            optimizer_type: Type of optimizer used.
            optimizer_config: Configuration used for optimization.
            training_size: Number of training examples.
            validation_size: Number of validation examples.
            model_file_path: Path to saved model file.
            validation_score: Final validation score (if available).
            duration_seconds: Optimization duration in seconds (if available).

        Returns:
            Dictionary ready for JSON serialization.
        """
        before_info = self._extract_prompt_info(before_model)
        after_info = self._extract_prompt_info(after_model)

        # Build optimizer settings dict (filter None values)
        optimizer_settings: dict[str, str | int | float] = {}
        for key, value in optimizer_config.items():
            if value is not None:
                optimizer_settings[key] = value

        # Build metrics dict
        metrics: dict[str, int | float] = {
            "training_size": training_size,
            "validation_size": validation_size,
        }
        if validation_score is not None:
            metrics["validation_score"] = validation_score
        if duration_seconds is not None:
            metrics["duration_seconds"] = duration_seconds

        # Build before/after optimization state
        before_instructions_val = before_info.get("instructions") or before_info.get(
            "signature_description"
        )
        before_instructions = (
            before_instructions_val
            if isinstance(before_instructions_val, str)
            else "Default DSPy ChainOfThought instructions"
        )
        before_num_demos_val = before_info.get("num_demos", "0")
        before_num_demos = int(before_num_demos_val) if isinstance(before_num_demos_val, str) else 0
        before_demos_val = before_info.get("demonstrations", [])
        before_demos = before_demos_val if isinstance(before_demos_val, list) else []

        before_state: dict[str, str | int | list] = {
            "instructions": before_instructions,
            "num_demos": before_num_demos,
            "demonstrations": before_demos,
        }

        after_instructions_val = after_info.get("instructions") or after_info.get(
            "signature_description"
        )
        after_instructions = (
            after_instructions_val
            if isinstance(after_instructions_val, str)
            else "Optimized DSPy ChainOfThought instructions"
        )
        after_num_demos_val = after_info.get("num_demos", "0")
        after_num_demos = int(after_num_demos_val) if isinstance(after_num_demos_val, str) else 0
        after_demos_val = after_info.get("demonstrations", [])
        after_demos = after_demos_val if isinstance(after_demos_val, list) else []

        after_state: dict[str, str | int | list] = {
            "instructions": after_instructions,
            "num_demos": after_num_demos,
            "demonstrations": after_demos,
        }

        # Build full config
        config: dict[str, str | int | float | dict | list] = {
            "version": self._get_next_version(model_file_path),
            "timestamp": datetime.now().isoformat(),
            "model_file": str(Path(model_file_path).name),
            "optimizer": {
                "type": optimizer_type,
                "settings": optimizer_settings,
            },
            "model_config": {
                "llm_model": self.settings.llm_model,
                "llm_base_url": self.settings.llm_base_url,
                "temperature": 0.3,
            },
            "before_optimization": before_state,
            "after_optimization": after_state,
            "metrics": metrics,
        }

        return config

    def save_optimization_config(
        self,
        before_model: MatchingModule,
        after_model: MatchingModule,
        optimizer_type: str,
        optimizer_config: dict[str, str | int | float | None],
        training_size: int,
        validation_size: int,
        model_file_path: str,
        validation_score: float | None = None,
        duration_seconds: float | None = None,
    ) -> Path:
        """Save optimization configuration to JSON file with version tracking.

        Args:
            before_model: Model before optimization.
            after_model: Model after optimization.
            optimizer_type: Type of optimizer used.
            optimizer_config: Configuration used for optimization.
            training_size: Number of training examples.
            validation_size: Number of validation examples.
            model_file_path: Path to saved model file.
            validation_score: Final validation score (if available).
            duration_seconds: Optimization duration in seconds (if available).

        Returns:
            Path to saved config file.
        """
        config = self.extract_optimization_config(
            before_model=before_model,
            after_model=after_model,
            optimizer_type=optimizer_type,
            optimizer_config=optimizer_config,
            training_size=training_size,
            validation_size=validation_size,
            model_file_path=model_file_path,
            validation_score=validation_score,
            duration_seconds=duration_seconds,
        )

        # Determine config file path in prompt-optimiser/configs
        version = config["version"]
        config_file = CONFIGS_DIR / f"dspy_config_v{version}.json"

        # Save to JSON with proper formatting
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved optimization config to {config_file}")
        return config_file

    def save_optimized_model(self, model: MatchingModule, path: str) -> None:
        """Save optimized model to disk.

        Args:
            model: Optimized matcher.
            path: Save path.
        """
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Saved optimized model to {model_path}")

    def load_optimized_model(self, path: str) -> MatchingModule:
        """Load optimized model from a specific path."""
        model_path = Path(path)
        self.matcher.load(str(model_path))
        logger.info(f"Loaded optimized model from {model_path}")
        return self.matcher

    def _get_latest_versioned_model_path(self) -> Path | None:
        """Return the highest matcher_vN.json path if it exists."""
        if not MODELS_DIR.exists():
            return None

        candidates = list(MODELS_DIR.glob("matcher_v*.json"))
        if not candidates:
            return None

        max_version = -1
        latest_path: Path | None = None
        for candidate in candidates:
            stem = candidate.stem
            try:
                version_str = stem.replace("matcher_v", "")
                if "_stage" in version_str:
                    version_str = version_str.split("_stage")[0]
                version = int(version_str)
                if version > max_version:
                    max_version = version
                    latest_path = candidate
            except ValueError:
                continue

        return latest_path

    def load_latest_model(self) -> MatchingModule | None:
        """Load the promoted matcher_latest.json or fall back to latest versioned model."""
        if ACTIVE_MODEL_PATH.exists():
            self.matcher.load(str(ACTIVE_MODEL_PATH))
            logger.info(f"Loaded promoted optimized model from {ACTIVE_MODEL_PATH}")
            return self.matcher

        latest_path = self._get_latest_versioned_model_path()
        if latest_path is None:
            logger.info("No optimized models found; using unoptimized matcher")
            return None

        self.matcher.load(str(latest_path))
        logger.info(
            "Loaded latest versioned optimized model from %s (no promoted model found)",
            latest_path,
        )
        return self.matcher

    def promote_latest_model(self) -> Path | None:
        """Copy the latest matcher_vN.json to matcher_latest.json for production use."""
        latest_path = self._get_latest_versioned_model_path()
        if latest_path is None:
            logger.info("No versioned optimized models available to promote")
            return None

        ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_path, ACTIVE_MODEL_PATH)
        logger.info("Promoted %s -> %s", latest_path.name, ACTIVE_MODEL_PATH.name)
        return ACTIVE_MODEL_PATH

    def predict_match(
        self,
        taxonomy: TaxonomyPage,
        content_items: list[WordPressContent],
    ) -> tuple[int, dict[str, float | str]]:
        """Make matching prediction using current model, returning rubric fields.

        Args:
            taxonomy: Taxonomy page to match.
            content_items: Available content items.

        Returns:
            Tuple of (best_match_index, rubric_dict). Returns (-1, {}) if parsing fails.
        """
        # Format inputs
        keywords_str = ", ".join(taxonomy.keywords) if taxonomy.keywords else "None"
        content_summaries = self._format_content_summaries(content_items)

        # Get prediction
        prediction = self.matcher(
            taxonomy_category=taxonomy.category,
            taxonomy_description=taxonomy.description,
            taxonomy_keywords=keywords_str,
            content_summaries=content_summaries,
        )

        # Extract results
        try:
            best_match_index = (
                int(prediction.best_match_index) if hasattr(prediction, "best_match_index") else -1
            )
            rubric: dict[str, float | str] = {}
            # Numeric rubric fields (default to 0.0 on parse failure)
            for key in ("topic_alignment", "intent_fit", "entity_overlap", "temporal_relevance"):
                value = getattr(prediction, key, None)
                try:
                    rubric[key] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    rubric[key] = 0.0
            # Decision and reasoning (strings)
            rubric["decision"] = str(getattr(prediction, "decision", "") or "")
            rubric["reasoning"] = str(getattr(prediction, "reasoning", "") or "")
        except (ValueError, TypeError, AttributeError):
            logger.warning("Failed to parse prediction, returning no match")
            return -1, {}

        return best_match_index, rubric

    def judge_candidate(
        self,
        taxonomy: TaxonomyPage,
        candidate: WordPressContent,
    ) -> dict[str, float | str]:
        """Score a single candidate with rubric fields using the matcher in single-candidate mode.

        Args:
            taxonomy: Taxonomy page to match.
            candidate: Single content candidate to score.

        Returns:
            Rubric dictionary containing topic_alignment, intent_fit, entity_overlap,
            temporal_relevance, decision, and reasoning. Returns empty dict on parse failure.
        """
        # Prepare summaries with only the single candidate
        content_summaries = self._format_content_summaries([candidate])
        keywords_str = ", ".join(taxonomy.keywords) if taxonomy.keywords else "None"

        prediction = self.matcher(
            taxonomy_category=taxonomy.category,
            taxonomy_description=taxonomy.description,
            taxonomy_keywords=keywords_str,
            content_summaries=content_summaries,
        )

        rubric: dict[str, float | str] = {}
        try:
            for key in ("topic_alignment", "intent_fit", "entity_overlap", "temporal_relevance"):
                value = getattr(prediction, key, None)
                try:
                    rubric[key] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    rubric[key] = 0.0
            rubric["decision"] = str(getattr(prediction, "decision", "") or "")
            rubric["reasoning"] = str(getattr(prediction, "reasoning", "") or "")
        except (ValueError, TypeError, AttributeError):
            logger.warning("Failed to parse judge prediction rubric")
            return {}

        return rubric

    def load_training_dataset(self, dataset_path: Path) -> list[dspy.Example]:
        """Load training dataset from CSV or JSON file.

        Args:
            dataset_path: Path to CSV or JSON dataset file.

        Returns:
            List of DSPy examples for training.

        Raises:
            ValueError: If file format is unsupported or data is invalid.
            FileNotFoundError: If dataset file does not exist.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        examples: list[dspy.Example] = []

        if dataset_path.suffix.lower() == ".json":
            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON dataset must be a list of example objects")

                for i, item in enumerate(data):
                    try:
                        example = dspy.Example(
                            taxonomy_category=item["taxonomy_category"],
                            taxonomy_description=item["taxonomy_description"],
                            taxonomy_keywords=item.get("taxonomy_keywords", "None"),
                            content_summaries=item["content_summaries"],
                            best_match_index=int(item["best_match_index"]),
                            reasoning=item.get("reasoning", ""),
                        ).with_inputs(
                            "taxonomy_category",
                            "taxonomy_description",
                            "taxonomy_keywords",
                            "content_summaries",
                        )
                        examples.append(example)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid example at index {i}: {e}")
                        continue

        elif dataset_path.suffix.lower() == ".csv":
            with open(dataset_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required_fields = [
                    "taxonomy_category",
                    "taxonomy_description",
                    "content_summaries",
                    "best_match_index",
                ]
                fieldnames = reader.fieldnames or []
                if not all(field in fieldnames for field in required_fields):
                    raise ValueError(f"CSV must contain columns: {', '.join(required_fields)}")

                for i, row in enumerate(reader):
                    try:
                        example = dspy.Example(
                            taxonomy_category=row["taxonomy_category"],
                            taxonomy_description=row["taxonomy_description"],
                            taxonomy_keywords=row.get("taxonomy_keywords", "None"),
                            content_summaries=row["content_summaries"],
                            best_match_index=int(row["best_match_index"]),
                            reasoning=row.get("reasoning", ""),
                        ).with_inputs(
                            "taxonomy_category",
                            "taxonomy_description",
                            "taxonomy_keywords",
                            "content_summaries",
                        )
                        examples.append(example)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid row {i + 1}: {e}")
                        continue
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}. Use .csv or .json")

        if not examples:
            raise ValueError("No valid examples found in dataset file")

        logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
        return examples

    def optimize_with_gepa(
        self,
        training_examples: list[dspy.Example],
        validation_examples: list[dspy.Example] | None = None,
        budget: str | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        num_threads: int | None = None,
        display_table: int | None = None,
    ) -> MatchingModule:
        """Optimize matcher using GEPA (Genetic Evolutionary Prompt Optimization).

        Args:
            training_examples: Training data.
            validation_examples: Optional validation data.
            budget: GEPA budget level (light/medium/heavy). Ignored if max_full_evals or max_metric_calls provided.
            max_full_evals: Explicit budget: maximum full evaluations.
            max_metric_calls: Explicit budget: maximum metric calls.
            num_threads: Parallel evaluation threads (default from settings).
            display_table: Number of example results to display (default from settings).

        Returns:
            Optimized matching module.
        """
        if not training_examples:
            logger.warning("No training examples provided, returning unoptimized model")
            return self.matcher

        # Use validation set if provided, otherwise split training set
        if validation_examples is None:
            split_point = int(len(training_examples) * self.settings.dspy_train_split_ratio)
            train_set = training_examples[:split_point]
            val_set = training_examples[split_point:]
        else:
            train_set = training_examples
            val_set = validation_examples

        logger.info(
            f"Optimizing with GEPA: {len(train_set)} training, {len(val_set)} validation examples"
        )

        # Configure reflection LM
        reflection_model = (
            self.settings.dspy_reflection_model
            if self.settings.dspy_reflection_model
            else self.settings.llm_model
        )

        # Determine provider from base URL
        model_name = reflection_model
        if "openrouter" in self.settings.llm_base_url.lower():
            if not model_name.startswith("openrouter/"):
                model_name = f"openrouter/{model_name}"
        elif not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"

        reflection_lm = dspy.LM(
            model=model_name,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=1.0,  # Higher temperature for diverse prompt proposals
            max_tokens=16384,  # Maximum for gpt-4o-mini and similar models
        )

        # Configure GEPA budget (exactly one must be provided)
        budget_config: dict[str, str | int] = {}
        if max_full_evals is not None:
            budget_config["max_full_evals"] = max_full_evals
        elif max_metric_calls is not None:
            budget_config["max_metric_calls"] = max_metric_calls
        else:
            budget_level = budget or self.settings.dspy_optimization_budget
            budget_config["auto"] = budget_level

        # Set up GEPA optimizer
        GEPA_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Check for existing state file and warn if present (may cause issues)
        state_file = GEPA_LOG_DIR / "gepa_state.bin"
        if state_file.exists():
            logger.warning(
                f"Existing GEPA state file found at {state_file}. "
                "If you encounter errors, try deleting this file to start fresh."
            )

        optimizer = GEPA(
            reflection_lm=reflection_lm,
            metric=self.accuracy_metric,
            num_threads=num_threads or self.settings.dspy_num_threads,
            failure_score=0.0,
            perfect_score=1.0,
            log_dir=str(GEPA_LOG_DIR),
            track_stats=True,
            seed=self.settings.dspy_optimization_seed,
            use_merge=True,  # Combine successful program variants
            **budget_config,
        )

        # Compile/optimize the model
        # Note: GEPA.compile() only accepts trainset, not valset
        # The validation set is used for evaluation after optimization
        try:
            logger.info("Starting GEPA optimization (this may take several minutes)...")
            logger.info(
                f"GEPA will use trainset ({len(train_set)} examples) for optimization. "
                f"Validation set ({len(val_set)} examples) will be used for evaluation after optimization."
            )
            optimized = optimizer.compile(
                self.matcher,
                trainset=train_set,
            )

            # Evaluate on validation set
            evaluator = dspy_evaluate.Evaluate(
                devset=val_set,
                metric=self.accuracy_metric,
                num_threads=num_threads or self.settings.dspy_num_threads,
                display_progress=True,
                display_table=(
                    display_table if display_table is not None else self.settings.dspy_display_table
                ),
            )

            eval_result = evaluator(optimized)
            # Extract score from EvaluationResult object
            if hasattr(eval_result, "score"):
                score = float(eval_result.score)
            elif hasattr(eval_result, "average"):
                score = float(eval_result.average)
            elif isinstance(eval_result, (int, float)):
                score = float(eval_result)
            else:
                # Try to convert to float directly
                try:
                    score = float(eval_result)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not extract score from evaluation result: {type(eval_result)}"
                    )
                    score = 0.0
            logger.info(f"GEPA optimized model validation score: {score:.3f}")

            return cast(MatchingModule, optimized)

        except EOFError as e:
            logger.error(
                f"GEPA optimization failed: Ran out of input ({e}). "
                "This usually indicates an API connection issue, timeout, or network problem. "
                "Check your LLM API configuration, network connectivity, and API rate limits."
            )
            logger.exception("Full traceback:")
            return self.matcher
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__} (no message)"
            logger.error(f"GEPA optimization failed: {error_msg}")
            if "input" in error_msg.lower() or "eof" in error_msg.lower():
                logger.error(
                    "This may indicate an API connection issue. Check your LLM API configuration and network connectivity."
                )
            logger.exception("Full traceback:")
            return self.matcher

    def optimize_with_dataset(
        self,
        training_examples: list[dspy.Example],
        validation_examples: list[dspy.Example] | None = None,
        optimizer_type: str = "gepa",
        **kwargs: int | str | None,
    ) -> MatchingModule:
        """Optimize matcher using dataset-based optimization with specified optimizer.

        Args:
            training_examples: Training data.
            validation_examples: Optional validation data.
            optimizer_type: Optimizer to use ('gepa', 'bootstrap-random-search', or 'bootstrap').
            **kwargs: Additional optimizer-specific parameters.

        Returns:
            Optimized matching module.
        """
        if not training_examples:
            logger.warning("No training examples provided, returning unoptimized model")
            return self.matcher

        if optimizer_type == "gepa":
            # Extract GEPA-specific kwargs
            gepa_kwargs: dict[str, int | str | None] = {
                "budget": kwargs.get("budget"),
                "max_full_evals": kwargs.get("max_full_evals"),
                "max_metric_calls": kwargs.get("max_metric_calls"),
                "num_threads": kwargs.get("num_threads"),
                "display_table": kwargs.get("display_table"),
            }
            return self.optimize_with_gepa(training_examples, validation_examples, **gepa_kwargs)  # type: ignore[arg-type]
        elif optimizer_type == "bootstrap-random-search":
            # Use validation set if provided, otherwise split training set
            if validation_examples is None:
                split_point = int(len(training_examples) * self.settings.dspy_train_split_ratio)
                train_set = training_examples[:split_point]
                val_set = training_examples[split_point:]
            else:
                train_set = training_examples
                val_set = validation_examples

            logger.info(
                f"Optimizing with BootstrapFewShotWithRandomSearch: {len(train_set)} training, {len(val_set)} validation examples"
            )

            optimizer = dspy_teleprompt.BootstrapFewShotWithRandomSearch(
                metric=self.accuracy_metric,
                max_bootstrapped_demos=kwargs.get("max_bootstrapped_demos", 4),
                max_labeled_demos=kwargs.get("max_labeled_demos", 8),
            )

            try:
                optimized = optimizer.compile(
                    self.matcher,
                    trainset=train_set,
                )

                # For bootstrap-random-search, rely on the optimizer's internal metric;
                # skip additional evaluation to keep tests and runtime behavior simple.
                logger.info("BootstrapFewShotWithRandomSearch optimization completed successfully")
                return cast(MatchingModule, optimized)

            except Exception as e:
                logger.error(f"BootstrapFewShotWithRandomSearch optimization failed: {e}")
                return self.matcher
        else:  # bootstrap (existing method)
            max_bootstrapped = kwargs.get("max_bootstrapped_demos", 4)
            max_labeled = kwargs.get("max_labeled_demos", 8)
            return self.optimize(
                training_examples,
                validation_examples,
                max_bootstrapped_demos=(
                    int(max_bootstrapped) if isinstance(max_bootstrapped, int) else 4
                ),
                max_labeled_demos=int(max_labeled) if isinstance(max_labeled, int) else 8,
            )
