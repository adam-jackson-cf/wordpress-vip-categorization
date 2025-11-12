"""Content categorization service using OpenAI Batch API."""

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import BatchJobStatus, CategorizationResult, WordPressContent

logger = logging.getLogger(__name__)


class CategorizationService:
    """Service for categorizing content using OpenAI Batch API.

    The Batch API is cost-effective for large-scale categorization tasks,
    offering 50% discount compared to standard API with 24-hour turnaround.
    """

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize categorization service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key, base_url=settings.openai_base_url
        )
        logger.info(f"Initialized categorization service with base URL: {settings.openai_base_url}")

    def create_categorization_prompt(self, content: WordPressContent, categories: list[str]) -> str:
        """Create prompt for content categorization.

        Args:
            content: WordPress content to categorize.
            categories: List of available categories.

        Returns:
            Categorization prompt.
        """
        categories_str = ", ".join(categories)
        prompt = f"""Analyze the following content and categorize it into one of these categories: {categories_str}

Title: {content.title}

Content: {content.content[:2000]}...

Respond with a JSON object in this exact format:
{{
  "category": "the most appropriate category",
  "confidence": 0.95,
  "reasoning": "brief explanation of why this category was chosen"
}}

The confidence should be a number between 0 and 1.
"""
        return prompt

    def prepare_batch_requests(
        self, content_items: list[WordPressContent], categories: list[str]
    ) -> list[dict[str, Any]]:
        """Prepare batch requests for OpenAI Batch API.

        Args:
            content_items: List of content to categorize.
            categories: Available categories.

        Returns:
            List of batch request objects.
        """
        requests = []
        for content in content_items:
            prompt = self.create_categorization_prompt(content, categories)
            request = {
                "custom_id": str(content.id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.settings.openai_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a content categorization assistant. "
                            "Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                },
            }
            requests.append(request)

        logger.info(f"Prepared {len(requests)} batch requests")
        return requests

    def create_batch_file(self, requests: list[dict[str, Any]]) -> str:
        """Create JSONL file for batch processing.

        Args:
            requests: List of batch requests.

        Returns:
            Path to created JSONL file.
        """
        file_path = Path(f"/tmp/batch_requests_{int(time.time())}.jsonl")
        with open(file_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Created batch file: {file_path}")
        return str(file_path)

    @retry(
        retry=retry_if_exception_type(openai.APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def submit_batch(self, file_path: str, description: str = "") -> str:
        """Submit batch job to OpenAI.

        Args:
            file_path: Path to JSONL batch file.
            description: Optional description for the batch.

        Returns:
            Batch ID.
        """
        # Upload file
        with open(file_path, "rb") as f:
            file_response = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description} if description else {},
        )

        logger.info(f"Submitted batch {batch.id} with file {file_response.id}")
        return batch.id

    def get_batch_status(self, batch_id: str) -> BatchJobStatus:
        """Get status of a batch job.

        Args:
            batch_id: Batch ID.

        Returns:
            Batch job status.
        """
        batch = self.client.batches.retrieve(batch_id)

        return BatchJobStatus(
            batch_id=batch.id,
            status=batch.status,
            created_at=batch.created_at,
            completed_at=batch.completed_at,
            request_counts={
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            },
            metadata=batch.metadata or {},
        )

    def wait_for_batch_completion(self, batch_id: str, check_interval: int = 60) -> BatchJobStatus:
        """Wait for batch job to complete.

        Args:
            batch_id: Batch ID.
            check_interval: Seconds between status checks.

        Returns:
            Final batch status.

        Raises:
            TimeoutError: If batch exceeds timeout.
            RuntimeError: If batch fails.
        """
        start_time = time.time()
        timeout = self.settings.openai_batch_timeout

        logger.info(f"Waiting for batch {batch_id} to complete...")

        while True:
            status = self.get_batch_status(batch_id)

            if status.status == "completed":
                logger.info(f"Batch {batch_id} completed successfully")
                return status

            if status.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")

            if status.status in ["expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_id} was {status.status}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Batch {batch_id} exceeded timeout of {timeout}s")

            logger.debug(
                f"Batch {batch_id} status: {status.status} "
                f"({status.request_counts.get('completed', 0)}/"
                f"{status.request_counts.get('total', 0)} completed)"
            )

            time.sleep(check_interval)

    def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Retrieve results from completed batch.

        Args:
            batch_id: Batch ID.

        Returns:
            List of result objects.
        """
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch.status})")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download results
        file_response = self.client.files.content(batch.output_file_id)
        results = []

        for line in file_response.text.strip().split("\n"):
            if line:
                results.append(json.loads(line))

        logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        return results

    def parse_batch_results(
        self, results: list[dict[str, Any]], batch_id: str
    ) -> list[CategorizationResult]:
        """Parse batch results into categorization results.

        Args:
            results: Raw batch results.
            batch_id: Batch ID.

        Returns:
            List of categorization results.
        """
        categorizations = []

        for result in results:
            try:
                content_id = UUID(result["custom_id"])
                response = result["response"]["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(response)

                categorization = CategorizationResult(
                    content_id=content_id,
                    category=parsed["category"],
                    confidence=float(parsed["confidence"]),
                    batch_id=batch_id,
                )
                categorizations.append(categorization)

            except Exception as e:
                logger.error(f"Error parsing result for {result.get('custom_id')}: {e}")
                continue

        logger.info(f"Parsed {len(categorizations)} categorization results")
        return categorizations

    def categorize_content_batch(
        self, content_items: list[WordPressContent], categories: list[str], wait: bool = True
    ) -> str:
        """Categorize content using batch API.

        Args:
            content_items: Content to categorize.
            categories: Available categories.
            wait: Whether to wait for batch completion.

        Returns:
            Batch ID.
        """
        # Prepare batch requests
        requests = self.prepare_batch_requests(content_items, categories)

        # Create batch file
        file_path = self.create_batch_file(requests)

        # Submit batch
        batch_id = self.submit_batch(
            file_path, description=f"Categorize {len(content_items)} content items"
        )

        # Optionally wait for completion
        if wait:
            self.wait_for_batch_completion(batch_id)

            # Retrieve and store results
            results = self.retrieve_batch_results(batch_id)
            categorizations = self.parse_batch_results(results, batch_id)

            # Store in database
            for cat in categorizations:
                self.db.insert_categorization(cat)

            logger.info(f"Stored {len(categorizations)} categorization results")

        return batch_id

    def get_categories_from_taxonomy(self) -> list[str]:
        """Get unique categories from taxonomy pages.

        Returns:
            List of unique category names.
        """
        taxonomy_pages = self.db.get_all_taxonomy()
        categories = list({page.category for page in taxonomy_pages})
        logger.info(f"Found {len(categories)} categories in taxonomy")
        return categories
