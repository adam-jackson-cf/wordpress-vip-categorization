<!-- e62292f1-c974-4ebe-990d-ef67c2494670 74cf6b34-0262-4da9-ae11-b8b126da6231 -->
# Two-Stage Optimization Workflow Plan

## Overview

Create a guided workflow that runs a cheap bootstrap optimizer test, waits for user confirmation, then runs a thorough GEPA medium optimization. The critical feedback stop ensures users review initial results before committing to expensive optimization.

## Implementation Steps

### 1. Create Optimization Workflow Script

**File:** `scripts/run_optimization_workflow.py`

Create a new script that:

- Runs Stage 1 (bootstrap optimizer) with minimal cost settings
- Displays results summary and prompts for user confirmation
- Only proceeds to Stage 2 if user confirms
- Runs Stage 2 (GEPA medium) after confirmation
- Saves separate output files for each stage with clear naming

**Key features:**

- Uses `bootstrap` optimizer for Stage 1 (cheap test)
- Uses `gepa` with `medium` budget for Stage 2 (thorough)
- Generates reports and config files for both stages
- Uses the versioned layout under `prompt-optimiser/`:
  - Stage 1: `prompt-optimiser/models/matcher_vN_stage1.json`, `prompt-optimiser/configs/dspy_config_vN_stage1.json`, `prompt-optimiser/reports/report_vN_stage1.md`
  - Stage 2: `prompt-optimiser/models/matcher_vM_stage2.json`, `prompt-optimiser/configs/dspy_config_vM_stage2.json`, `prompt-optimiser/reports/report_vM_stage2.md`
- Interactive prompt: "Review Stage 1 results. Proceed to Stage 2 (GEPA medium)? [y/N]"

### 2. Update OPTIMIZATION_QUICKSTART.md

**File:** `docs/OPTIMIZATION_QUICKSTART.md`

Add a new section documenting the two-stage workflow:

- Command to run the workflow script
- Explanation of Stage 1 (bootstrap) vs Stage 2 (GEPA medium)
- What to review during the feedback stop
- Expected outputs and file locations

### 3. Add Workflow Command to CLI (Optional Enhancement)

**File:** `src/cli.py`

Consider adding a convenience command:

```python
@cli.command(name="optimize-workflow")
@click.option("--dataset", ...)
@click.option("--auto-proceed", is_flag=True, help="Skip confirmation prompt")
def optimize_workflow(...):
    """Run two-stage optimization workflow (bootstrap test + GEPA medium)."""
```

This would wrap the script functionality in the CLI for better integration.

## Stage 1 Configuration (Cheap Test)

- Optimizer: `bootstrap`
- Train split: `0.2` (20 examples training, 80 validation)
- Output model: `prompt-optimiser/models/matcher_vN_stage1.json`
- Report: `prompt-optimiser/reports/report_vN_stage1.md`
- Config: `prompt-optimiser/configs/dspy_config_vN_stage1.json`
- Seed: `42` (for reproducibility)

## Stage 2 Configuration (Thorough)

- Optimizer: `gepa`
- Budget: `medium`
- Train split: `0.2` (same split as Stage 1)
- Output model: `prompt-optimiser/models/matcher_vM_stage2.json`
- Report: `prompt-optimiser/reports/report_vM_stage2.md`
- Config: `prompt-optimiser/configs/dspy_config_vM_stage2.json`
- Seed: `42` (same seed for consistency)

## Critical Feedback Stop Requirements

- Display Stage 1 summary: validation score, duration, cost estimate
- Show file locations: report, config, model
- Clear prompt: "Review Stage 1 results. Proceed to Stage 2 (GEPA medium)? [y/N]"
- Default to "No" (user must explicitly confirm)
- Exit gracefully if user declines
- Provide instructions for manual Stage 2 run if needed

## File Structure

```
prompt-optimiser/
  models/
    matcher_vN_stage1.json             # Stage 1 optimized model
    matcher_vM_stage2.json             # Stage 2 optimized model (if confirmed)
  reports/
    report_vN_stage1.md                # Stage 1 report
    report_vM_stage2.md                # Stage 2 report (if confirmed)
  configs/
    dspy_config_vN_stage1.json         # Stage 1 config
    dspy_config_vM_stage2.json         # Stage 2 config (if confirmed)
```

## Error Handling

- If Stage 1 fails, stop workflow and report error
- If user declines Stage 2, provide command to run it manually later
- Preserve Stage 1 outputs even if Stage 2 is skipped
- Log all actions for audit trail

### To-dos

- [ ] Create scripts/run_optimization_workflow.py with two-stage optimization logic, user confirmation prompt, and proper file naming
- [ ] Implement critical feedback stop with clear summary display, file locations, and confirmation prompt (defaults to No)
- [ ] Add two-stage workflow section to OPTIMIZATION_QUICKSTART.md with usage instructions and what to review
- [ ] Test the workflow script with the dataset to ensure both stages work correctly and feedback stop functions properly