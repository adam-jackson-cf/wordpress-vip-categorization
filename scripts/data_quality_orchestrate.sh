#!/bin/bash
# Data Quality Analysis Workflow Orchestrator
#
# This script sequences data quality analysis tasks:
# 1. Evaluate semantic thresholds
# 2. Analyze missing signals
# 3. Extract sample matches
# 4. Generate comprehensive report
#
# Usage:
#   ./data_quality_orchestrate.sh [--skip-thresholds] [--skip-signals] [--skip-samples]
#
# Environment:
#   - Ensure .env is loaded (source .env) or export required vars
#   - For corporate networks, use scripts/corp_ca_exec.sh wrapper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Defaults
SKIP_THRESHOLDS=false
SKIP_SIGNALS=false
SKIP_SAMPLES=false
LIMIT=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-thresholds)
            SKIP_THRESHOLDS=true
            shift
            ;;
        --skip-signals)
            SKIP_SIGNALS=true
            shift
            ;;
        --skip-samples)
            SKIP_SAMPLES=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-thresholds] [--skip-signals] [--skip-samples] [--limit N]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Data Quality Analysis Workflow"
echo "=========================================="
echo ""

# Step 1: Evaluate semantic thresholds
if [[ "${SKIP_THRESHOLDS}" == "false" ]]; then
    echo "Step 1: Evaluating semantic similarity thresholds..."
    uv run python scripts/helpers/evaluate_semantic_thresholds.py
    echo "✓ Threshold evaluation completed"
else
    echo "Step 1: Skipping threshold evaluation (--skip-thresholds)"
fi

# Step 2: Analyze missing signals
if [[ "${SKIP_SIGNALS}" == "false" ]]; then
    echo ""
    echo "Step 2: Analyzing missing detection signals..."
    uv run python scripts/helpers/analyze_missing_signals.py \
        --limit "${LIMIT}" \
        --output data/diagnostics/missing_signal_analysis.json
    echo "✓ Missing signals analysis completed"
else
    echo "Step 2: Skipping missing signals analysis (--skip-signals)"
fi

# Step 3: Extract sample matches
if [[ "${SKIP_SAMPLES}" == "false" ]]; then
    echo ""
    echo "Step 3: Extracting sample matches..."
    # Find the most recent match snapshot
    LATEST_SNAPSHOT=$(ls -t results/match_snapshot_*.csv 2>/dev/null | head -1)
    if [[ -n "${LATEST_SNAPSHOT}" ]]; then
        uv run python scripts/helpers/extract_sample_matches.py \
            --input "${LATEST_SNAPSHOT}" \
            --output data/examples/sample_matches.csv \
            --min-score 0.5 \
            --max-score 0.75 \
            --sample-size 50
        echo "✓ Sample matches extracted"
    else
        echo "⚠ No match snapshot found in results/, skipping sample extraction"
    fi
else
    echo "Step 3: Skipping sample extraction (--skip-samples)"
fi

# Step 4: Generate comprehensive report
echo ""
echo "Step 4: Generating comprehensive analysis report..."
uv run python scripts/helpers/generate_report.py \
    --output results/semantic_match_analysis_$(date +%Y%m%d).md
echo "✓ Report generated"

echo ""
echo "=========================================="
echo "Data quality analysis completed!"
echo "=========================================="
echo ""
echo "Review outputs:"
echo "  - Thresholds: console output above"
echo "  - Missing signals: data/diagnostics/missing_signal_analysis.json"
echo "  - Sample matches: data/examples/sample_matches.csv"
echo "  - Full report: results/semantic_match_analysis_*.md"
