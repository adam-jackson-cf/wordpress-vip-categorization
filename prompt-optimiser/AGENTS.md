## Prompt Optimizer Artifacts

This directory holds all DSPy / GEPA prompt-optimization outputs. It is an operational area for models and reports, not the primary source of documentation (see `docs/DSPY_IMPLEMENTATION.md` for full design details).

### Layout

- `models/`
  - `matcher_vN.json` – Versioned optimized matcher artifacts (N starts at 1 and increments per optimization run).
- `configs/`
  - `dspy_config_vN.json` – JSON configs describing each optimization run (optimizer settings, metrics, before/after prompts).
- `reports/`
  - `report_vN.md` – Optional human-readable reports for a given optimization run.

### How Versions Work

- Each optimization run computes the next version number by scanning `configs/` for `dspy_config_v*.json` and incrementing the highest `vN`.
- That same `N` is used for:
  - `models/matcher_vN.json`
  - `configs/dspy_config_vN.json`
  - `reports/report_vN.md` (when generated).

### How the Matcher Is Loaded

- `CategorizationService` scans `models/` for `matcher_v*.json` at startup and loads the highest version number, if present.
- If no versioned model exists, the unoptimized DSPy matcher is used.

For how to run optimizations, choose optimizers, and interpret configs/reports, see:

- `docs/DSPY_IMPLEMENTATION.md`
- `docs/OPTIMIZATION_QUICKSTART.md`
- `docs/DSPY_GEPA_BEST_PRACTICES.md`
