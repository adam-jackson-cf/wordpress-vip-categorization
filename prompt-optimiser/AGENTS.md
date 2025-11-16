# Prompt Optimiser Layout

Defines how DSPy optimization artifacts are organized under `prompt-optimiser/`.

## Directory Structure
- `models/` stores versioned optimized matchers (`matcher_vN.json`).
- `configs/` stores JSON specs for each optimization run (`dspy_config_vN.json`).
- `reports/` optionally captures human-readable run summaries (`report_vN.md`).

```text
prompt-optimiser/
├── models/matcher_v3.json
├── configs/dspy_config_v3.json
└── reports/report_v3.md
```

## Versioning Rules
- Each optimization run increments `N` by scanning existing configs; reuse that number across models/configs/reports.
- Never overwrite or delete prior versions; add new files instead so historical runs remain reproducible.
- Runtime services load the highest `matcher_vN.json`; keep files well-formed JSON to avoid startup failures.

## Tooling & Testing
- CLI scripts and automation helpers (e.g., promotion, dataset generation) must have at least one smoke test or doctest that exercises the happy path with fixtures.
- Serialization helpers must be covered by unit tests to prove that the emitted config/report/meta JSON is valid (no `None` leakage or unserializable objects).
