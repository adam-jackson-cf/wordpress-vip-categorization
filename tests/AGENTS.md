# Testing Guide

## Organization

`tests/unit/` - Mocked dependencies
`tests/integration/` - Real dependencies (mark `@pytest.mark.integration`)
`tests/conftest.py` - Shared fixtures

## Test Requirements

Minimum 80% coverage (enforced by pytest)
Use descriptive test names that explain what is being tested
Mock external dependencies in unit tests (use `pytest-mock`)
Use fixtures for shared test data (see `tests/conftest.py`)
Follow Arrange-Act-Assert (AAA) pattern
Integration tests must be marked with `@pytest.mark.integration`

## Running Tests

All tests: `pytest`
Unit only: `pytest tests/unit/ -v`
Integration only: `pytest tests/integration/ -v -m integration`
Specific file: `pytest tests/unit/test_name.py`
With coverage: `pytest --cov=src --cov-report=html`

Coverage gate: 80% (enforced by pytest)
Excluded: CLI (`src/cli.py`), DSPy optimization modules
Focus on business logic in unit tests
For focused test runs: `pytest --no-cov tests/unit/test_x.py` (80% gate runs on full suite)

## Writing Tests

### AAA Pattern

```python
def test_matching_returns_high_similarity(mock_supabase):
    # Arrange
    settings = Settings(similarity_threshold=0.85)
    service = MatchingService(settings, mock_supabase)

    # Act
    results = service.match_taxonomy("test-123")

    # Assert
    assert all(r.similarity_score >= 0.85 for r in results)
```

### Using Fixtures

See `tests/conftest.py` for available fixtures:
- `mock_settings` - Pydantic Settings with test values
- `mock_supabase` - Mocked SupabaseClient
- `sample_taxonomy` - Sample TaxonomyPage instances

```python
def test_with_fixture(mock_settings, mock_supabase):
    service = MyService(mock_settings, mock_supabase)
    # Test implementation
```

### Mocking Patterns

Mock Supabase calls:
```python
from unittest.mock import Mock

mock_client = Mock()
mock_client.get_taxonomy_by_id.return_value = TaxonomyPage(...)
```

Mock external APIs:
```python
@patch('src.services.matching.requests.post')
def test_api_call(mock_post):
    mock_post.return_value.json.return_value = {...}
```

Mark integration tests:
```python
@pytest.mark.integration
def test_full_pipeline():
    # Uses real Supabase, OpenRouter
```

## Common Patterns

Test async functions:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

Test exceptions:
```python
with pytest.raises(ValueError, match="Expected error message"):
    function_that_raises()
```

Back to main guide: [../AGENTS.md](../AGENTS.md)

## Testing Checklist (tests)

- [ ] Overall coverage ≥ 80% with `pytest` (excluding CLI/DSPy where appropriate)
- [ ] Tests follow the Arrange–Act–Assert (AAA) structure
- [ ] Test names are descriptive and explain intent/behavior
- [ ] External services and I/O are mocked in unit tests
- [ ] Integration tests are marked with `@pytest.mark.integration`
- [ ] Core business logic paths are covered by unit tests
