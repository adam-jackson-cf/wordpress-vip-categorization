#!/bin/bash
# DSPy/GEPA Optimization Workflow Orchestrator
#
# This script sequences the DSPy optimization workflow:
# 1. Build/generate training dataset
# 2. Run quick bootstrap test (optional but recommended)
# 3. Run full GEPA optimization
# 4. Promote optimized model to production
#
# Usage:
#   ./orchestrate.sh [--skip-bootstrap] [--dataset PATH] [--budget light|medium|heavy]
#
# Environment:
#   - Ensure .env is loaded (source .env) or export required vars
#   - For corporate networks, use scripts/corp_ca_exec.sh wrapper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Defaults
SKIP_BOOTSTRAP=false
DATASET="${PROJECT_ROOT}/data/dspy_training_dataset.csv"
BUDGET="medium"
TAXONOMY_FILE="${PROJECT_ROOT}/data/Spain_New.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-bootstrap)
            SKIP_BOOTSTRAP=true
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --taxonomy-file)
            TAXONOMY_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-bootstrap] [--dataset PATH] [--budget light|medium|heavy] [--taxonomy-file PATH]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "DSPy/GEPA Optimization Workflow"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Budget: ${BUDGET}"
echo "Taxonomy: ${TAXONOMY_FILE}"
echo ""

# Step 1: Generate dataset if it doesn't exist
if [[ ! -f "${DATASET}" ]]; then
    echo "Step 1: Generating DSPy training dataset..."
    uv run python prompt-optimiser/scripts/generate_dspy_dataset.py \
        --taxonomy "${TAXONOMY_FILE}" \
        --output "${DATASET}"
    echo "✓ Dataset generated"
else
    echo "Step 1: Using existing dataset: ${DATASET}"
fi

# Step 2: Bootstrap smoke test (optional but recommended)
if [[ "${SKIP_BOOTSTRAP}" == "false" ]]; then
    echo ""
    echo "Step 2: Running bootstrap smoke test..."
    uv run python prompt-optimiser/scripts/run_quick_optimization_test.py \
        --dataset "${DATASET}" \
        --train-split 0.2 \
        --seed 42
    echo "✓ Bootstrap test completed"
    echo ""
    read -p "Continue with full GEPA optimization? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted by user"
        exit 0
    fi
else
    echo "Step 2: Skipping bootstrap test (--skip-bootstrap)"
fi

# Step 3: Full GEPA optimization
echo ""
echo "Step 3: Running full GEPA optimization..."
uv run python prompt-optimiser/scripts/run_optimization_workflow.py \
    --dataset "${DATASET}" \
    --budget "${BUDGET}" \
    --train-split 0.2 \
    --seed 42
echo "✓ GEPA optimization completed"

# Step 4: Prompt for promotion
echo ""
echo "Step 4: Model promotion"
echo "Review the optimization report in prompt-optimiser/reports/"
read -p "Promote the latest optimized model to production? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python prompt-optimiser/scripts/promote_optimized_model.py
    echo "✓ Model promoted to matcher_latest.json"
else
    echo "Skipping promotion. Run manually:"
    echo "  uv run python prompt-optimiser/scripts/promote_optimized_model.py [--version N]"
fi

echo ""
echo "=========================================="
echo "Workflow completed successfully!"
echo "=========================================="
