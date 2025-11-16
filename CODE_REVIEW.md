# Code Review Checklist

This checklist captures the baseline standards reviewers (human or automated) must enforce before merging changes to `wordpress-vip-categorization`. Pair it with `AGENTS.md` for directory-specific guidance and `docs/CODE_REVIEW_GUIDE.md` for workflow tips.

## 1. Quality Gate & Tooling
- `make quality-check` must be green (`black --check`, `ruff`, `mypy src`, `pytest --cov=src --cov-fail-under=80`).
- Slow/integration tests belong behind `@pytest.mark.slow` / `@pytest.mark.integration`; default gate should stay fast.
- No new module may be excluded from coverage without updating `[tool.coverage.run].omit` plus a written justification.

## 2. Python & Typing Standards
- Imports ordered stdlib → third-party → local, alphabetical within each block.
- Every function and method signature uses modern unions (e.g., `str | None`).
- Prefer `collections.abc` for abstract types (`Sequence`, `Mapping`, etc.).
- No direct `os.environ` lookups—configuration flows through `pydantic_settings`.

## 3. Configuration & Models
- Settings classes use `pydantic` v2 semantics (`Field`, `HttpUrl`, `SettingsConfigDict`).
- Models avoid mutable defaults; use `Field(default_factory=list)`.
- Secrets (API keys, tokens) never appear in logs, docs, or committed files.

## 4. Services, CLI, and Dependency Injection
- CLI commands instantiate `Settings` → `SupabaseClient` → service classes explicitly; no singletons or hidden globals.
- Services accept dependencies via constructors to keep tests patchable.
- Long-running orchestration logic (workflow gating, demotion, metrics) requires deterministic unit tests.

## 5. Database & External I/O
- All Supabase access goes through `SupabaseClient` with parameterized queries; the schema source of truth is `src/data/schema.sql`.
- Database migrations happen via `python -m src.cli init-db`; no ad-hoc SQL embedded in other modules.
- Outbound network/storage calls are wrapped with `tenacity` retries, logged with `logging.getLogger(__name__)`, and re-raised on failure.

## 6. DSPy & Categorization
- DSPy optimizers persist artifacts under `prompt-optimiser/`; promotion flows through `scripts/promote_optimized_model.py` to `matcher_latest.json`.
- `optimize-dataset` usage documents dataset size, budget, and promotion plan in PR descriptions.
- Helper wrappers for network/storage calls are exposed at module scope so tests can patch them directly.

## 7. Testing Expectations
- New helpers/services ship with unit tests unless explicitly documented as integration-only.
- Integration-only flows (`src/cli`, connectors, Supabase client) describe manual test coverage in PRs before being excluded from unit coverage.
- Tests mock external services; deterministic orchestration logic gets direct unit coverage even when surrounding module is skipped.

## 8. Documentation & Safety
- READMEs and docs must only reference files/commands that exist; update links when structure changes.
- `.cursor/plans/**` are human-authored—never auto-format or rewrite unless content actually changes.
- No secrets, placeholders, or mock/test data belong in production code or committed configs.

Reviewers should block merges when any checklist item fails or when business logic appears incorrect, even if automated checks pass.
