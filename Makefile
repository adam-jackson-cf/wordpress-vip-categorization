.PHONY: help install test lint format type-check quality-check clean

help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run tests with coverage"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint           - Run linter (ruff)"
	@echo "  format         - Format code with black"
	@echo "  format-check   - Check code formatting"
	@echo "  type-check     - Run type checker (mypy)"
	@echo "  quality-check  - Run all quality checks"
	@echo "  clean          - Clean up generated files"

install:
	pip install -e ".[dev]"

test:
	pytest

test-unit:
	pytest --no-cov -m "not integration and not slow" tests/unit

test-integration:
	pytest --no-cov tests/integration -m integration

lint:
	ruff check src tests

format:
	black src tests

format-check:
	black --check src tests

type-check:
	mypy src

quality-check: format-check lint type-check
	pytest -n 4 --cov=src --cov-report=term-missing --cov-fail-under=80 -m "not integration and not slow"
	@echo "✓ All quality checks passed!"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
