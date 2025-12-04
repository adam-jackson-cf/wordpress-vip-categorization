"""Language detection helpers."""

from __future__ import annotations

from typing import Callable, cast

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect uses a global PRNG; seed it for deterministic results in tests.
DetectorFactory.seed = 42


def detect_language_code(text: str) -> str:
    """Best-effort detection of ISO-639-1 language code."""

    candidate = (text or "").strip()
    if not candidate:
        return "unknown"

    detect_func: Callable[[str], str] = cast(Callable[[str], str], detect)

    try:
        code = detect_func(candidate)
    except LangDetectException:
        return "unknown"
    return code.lower()
