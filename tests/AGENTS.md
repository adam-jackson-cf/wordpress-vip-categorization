# Tests Contract

Applies to everything under `tests/`.

## Layout Expectations
- `tests/unit/` mocks external services and is the default target for fast feedback.
- `tests/integration/` exercises Supabase/WordPress/OpenRouter and must be marked with `@pytest.mark.integration`.
- Shared fixtures live in `tests/conftest.py`; extend there before duplicating setup code.

## Behavioral Rules
- Coverage must stay ≥80% when running `pytest --cov=src --cov-fail-under=80` (run via `make quality-check`).
- Slow paths belong behind `@pytest.mark.slow`; keep them out of the default gate.
- Follow Arrange–Act–Assert and pick descriptive test names that explain behavior, not implementation.
- Patch module-level helper wrappers (e.g., `_post_schema_sql_with_retry`) rather than third-party clients so refactors don’t break tests.
- Deterministic orchestration logic (rubric gating, demotion flows, evaluator summaries, etc.) must have explicit unit tests; don’t rely on integration coverage.

```python
def test_matching_respects_similarity_threshold(mock_settings, mock_supabase):
    service = MatchingService(mock_settings, mock_supabase)

    result = service.match_taxonomy("tax-123")

    assert result.similarity >= mock_settings.similarity_threshold
```

## Command Reference
- All tests: `pytest`
- Unit only: `pytest tests/unit -v`
- Integration only: `pytest tests/integration -m integration`
- Focused run without coverage: `pytest --no-cov tests/unit/test_matching.py`

## Checklist (`tests/`)
- Coverage ≥80% on the default suite; integration/slow tests are marker-gated.
- External I/O is mocked in unit tests; only integration tests hit live services.
- Tests follow the AAA pattern with descriptive names and assertions tied to behavior.
