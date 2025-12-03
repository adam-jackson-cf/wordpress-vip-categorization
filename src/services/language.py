"""Language detection and translation helpers."""

from __future__ import annotations

import logging
from typing import Any

import openai
from langdetect import DetectorFactory, LangDetectException, detect
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings

logger = logging.getLogger(__name__)

# langdetect uses a global PRNG; seed it for deterministic results in tests.
DetectorFactory.seed = 42


def detect_language_code(text: str) -> str:
    """Best-effort detection of ISO-639-1 language code.

    Args:
        text: Input text to inspect.

    Returns:
        Lowercase ISO-639-1 code when determinable, otherwise ``"unknown"``.
    """

    candidate = (text or "").strip()
    if not candidate:
        return "unknown"

    try:
        code = detect(candidate)
    except LangDetectException:
        return "unknown"
    return code.lower()


OPENAI_RETRY_EXCEPTIONS = (
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIStatusError,
)


class TranslationService:
    """Lightweight translator built on the LLM provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model
        self._cache: dict[tuple[str, str], str] = {}
        logger.info("Initialized translation service via %s", settings.llm_base_url)

    def _cache_key(self, text: str, target_language: str) -> tuple[str, str]:
        return (text.strip(), target_language.lower())

    @retry(  # pragma: no cover - network I/O
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _translate_via_api(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> str:
        system_prompt = (
            "You are a professional translator. Translate faithfully, keep proper nouns, "
            "and respond with the translation only."
        )
        source_label = source_language or "auto-detected language"
        user_prompt = (
            f"Translate the following text from {source_label} to {target_language}. "
            "Return only the translated text without commentary.\n\n"
            f"Text:\n{text.strip()}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = response.choices[0].message
        content: Any = getattr(message, "content", "")
        if isinstance(content, list):
            # OpenAI SDK may return a list of content parts; join their text values.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            content = str(content)
        return content.strip()

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> str:
        """Translate text into the target language, reusing cached results when possible."""

        snapshot = (text or "").strip()
        if not snapshot:
            return ""

        key = self._cache_key(snapshot, target_language)
        if key in self._cache:
            return self._cache[key]

        translated = self._translate_via_api(snapshot, target_language, source_language)
        self._cache[key] = translated
        return translated
