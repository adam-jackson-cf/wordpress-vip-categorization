"""DSPy-based prompt optimization for taxonomy-to-content matching."""

import logging
from typing import cast

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchStage, TaxonomyPage, WordPressContent

logger = logging.getLogger(__name__)


class TaxonomyMatcher(dspy.Signature):
    """Match a taxonomy page to the most relevant content page."""

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
    confidence: float = dspy.OutputField(desc="Confidence score (0-1) for the match")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this match was chosen")


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
            Prediction with best_match_index, confidence, and reasoning.
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
        # DSPy 3.x uses dspy.clients.LM with model format "provider/model"
        from dspy.clients import LM

        # Determine provider from base URL or default to openai
        model_name = settings.llm_model
        if "openrouter" in settings.llm_base_url.lower():
            # OpenRouter format: openrouter/model-name
            if not model_name.startswith("openrouter/"):
                model_name = f"openrouter/{model_name}"
        elif not model_name.startswith("openai/"):
            # Default to openai provider
            model_name = f"openai/{model_name}"

        lm = LM(
            model=model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.3,
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
                    confidence=matching.similarity_score,
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

    def accuracy_metric(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """Compute accuracy metric for matching evaluation.

        Args:
            example: Ground truth example.
            prediction: Model prediction.

        Returns:
            Score between 0 and 1.
        """
        # Check if best_match_index matches
        try:
            expected_index = int(example.best_match_index)
            predicted_index = (
                int(prediction.best_match_index) if hasattr(prediction, "best_match_index") else -1
            )
        except (ValueError, TypeError, AttributeError):
            return 0.0

        # Exact match on index
        index_match = 1.0 if predicted_index == expected_index else 0.0

        # Confidence calibration: penalize if prediction is too confident but wrong
        confidence_penalty = 0.0
        if hasattr(prediction, "confidence"):
            try:
                conf = float(prediction.confidence)
                if index_match == 0.0 and conf > 0.8:
                    confidence_penalty = 0.2  # Penalize overconfidence on wrong match
            except (ValueError, TypeError):
                pass

        return max(0.0, index_match - confidence_penalty)

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
            split_point = int(len(training_examples) * 0.8)
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

            # Evaluate on validation set
            evaluator = Evaluate(
                devset=val_set,
                metric=self.accuracy_metric,
                num_threads=1,
                display_progress=True,
            )

            score = evaluator(optimized)
            logger.info(f"Optimized model validation score: {score:.3f}")

            return cast(MatchingModule, optimized)

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.matcher

    def save_optimized_model(self, model: MatchingModule, path: str) -> None:
        """Save optimized model to disk.

        Args:
            model: Optimized matcher.
            path: Save path.
        """
        model.save(path)
        logger.info(f"Saved optimized model to {path}")

    def load_optimized_model(self, path: str) -> MatchingModule:
        """Load optimized model from disk.

        Args:
            path: Model path.

        Returns:
            Loaded matcher.
        """
        self.matcher.load(path)
        logger.info(f"Loaded optimized model from {path}")
        return self.matcher

    def predict_match(
        self,
        taxonomy: TaxonomyPage,
        content_items: list[WordPressContent],
    ) -> tuple[int, float]:
        """Make matching prediction using current model.

        Args:
            taxonomy: Taxonomy page to match.
            content_items: Available content items.

        Returns:
            Tuple of (best_match_index, confidence). Returns (-1, 0.0) if no good match.
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
            confidence = float(prediction.confidence) if hasattr(prediction, "confidence") else 0.0
        except (ValueError, TypeError, AttributeError):
            logger.warning("Failed to parse prediction, returning no match")
            return -1, 0.0

        return best_match_index, confidence
