"""DSPy-based prompt optimization for categorization."""

import logging
from typing import Any

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import WordPressContent

logger = logging.getLogger(__name__)


class ContentCategorizer(dspy.Signature):
    """Categorize content into predefined categories."""

    title: str = dspy.InputField(desc="Content title")
    content: str = dspy.InputField(desc="Content text")
    categories: str = dspy.InputField(desc="Available categories")
    category: str = dspy.OutputField(desc="Selected category")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1)")


class CategorizerModule(dspy.Module):
    """DSPy module for content categorization."""

    def __init__(self) -> None:
        """Initialize categorizer module."""
        super().__init__()
        self.predict = dspy.ChainOfThought(ContentCategorizer)

    def forward(self, title: str, content: str, categories: str) -> dspy.Prediction:
        """Forward pass for categorization.

        Args:
            title: Content title.
            content: Content text.
            categories: Available categories as comma-separated string.

        Returns:
            Prediction with category and confidence.
        """
        return self.predict(title=title, content=content, categories=categories)


class DSPyOptimizer:
    """Optimizer for categorization prompts using DSPy."""

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize DSPy optimizer.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client

        # Configure DSPy with OpenAI (or OpenAI-compatible API)
        lm = dspy.OpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_base_url,
            temperature=0.3,
        )
        dspy.settings.configure(lm=lm)

        self.categorizer = CategorizerModule()
        logger.info(f"Initialized DSPy optimizer with model: {settings.openai_model}")

    def prepare_training_data(
        self, content_items: list[WordPressContent], categories: list[str]
    ) -> list[dspy.Example]:
        """Prepare training data from labeled content.

        Args:
            content_items: Content items with known categories.
            categories: Available categories.

        Returns:
            List of DSPy examples.
        """
        examples = []
        categories_str = ", ".join(categories)

        for content in content_items:
            # Get known categorization for this content (if exists)
            categorizations = self.db.get_categorizations_by_content(content.id)

            if categorizations:
                # Use the first categorization as ground truth
                cat = categorizations[0]

                example = dspy.Example(
                    title=content.title,
                    content=content.content[:1000],  # First 1000 chars
                    categories=categories_str,
                    category=cat.category,
                    confidence=cat.confidence,
                ).with_inputs("title", "content", "categories")

                examples.append(example)

        logger.info(f"Prepared {len(examples)} training examples")
        return examples

    def accuracy_metric(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """Compute accuracy metric for evaluation.

        Args:
            example: Ground truth example.
            prediction: Model prediction.

        Returns:
            Score between 0 and 1.
        """
        # Exact match on category
        category_match = 1.0 if prediction.category == example.category else 0.0

        # Confidence should be reasonable (penalize if too confident on wrong answer)
        confidence_penalty = 0.0
        if hasattr(prediction, "confidence"):
            try:
                conf = float(prediction.confidence)
                if category_match == 0.0 and conf > 0.8:
                    confidence_penalty = 0.2  # Penalize overconfidence
            except (ValueError, TypeError):
                pass

        return max(0.0, category_match - confidence_penalty)

    def optimize(
        self,
        training_examples: list[dspy.Example],
        validation_examples: list[dspy.Example] | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
    ) -> CategorizerModule:
        """Optimize categorizer using few-shot learning.

        Args:
            training_examples: Training data.
            validation_examples: Optional validation data.
            max_bootstrapped_demos: Max bootstrapped demonstrations.
            max_labeled_demos: Max labeled demonstrations.

        Returns:
            Optimized categorizer module.
        """
        if not training_examples:
            logger.warning("No training examples provided, returning unoptimized model")
            return self.categorizer

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
                self.categorizer,
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

            return optimized

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.categorizer

    def save_optimized_model(self, model: CategorizerModule, path: str) -> None:
        """Save optimized model to disk.

        Args:
            model: Optimized categorizer.
            path: Save path.
        """
        model.save(path)
        logger.info(f"Saved optimized model to {path}")

    def load_optimized_model(self, path: str) -> CategorizerModule:
        """Load optimized model from disk.

        Args:
            path: Model path.

        Returns:
            Loaded categorizer.
        """
        self.categorizer.load(path)
        logger.info(f"Loaded optimized model from {path}")
        return self.categorizer

    def predict(self, content: WordPressContent, categories: list[str]) -> dict[str, Any]:
        """Make prediction using current model.

        Args:
            content: Content to categorize.
            categories: Available categories.

        Returns:
            Prediction dictionary with category and confidence.
        """
        categories_str = ", ".join(categories)
        prediction = self.categorizer(
            title=content.title,
            content=content.content[:1000],
            categories=categories_str,
        )

        return {
            "category": prediction.category,
            "confidence": (
                float(prediction.confidence) if hasattr(prediction, "confidence") else 0.5
            ),
        }
