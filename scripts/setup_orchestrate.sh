#!/bin/bash
# Setup and Infrastructure Workflow Orchestrator
#
# This script sequences setup and infrastructure tasks:
# 1. Test setup and configuration
# 2. Initialize/reset Supabase schema
# 3. Bootstrap Supabase with data (optional)
#
# Usage:
#   ./setup_orchestrate.sh [--skip-test] [--skip-init] [--bootstrap] [--taxonomy-file PATH] [--sites SITES]
#
# Environment:
#   - Ensure .env is loaded (source .env) or export required vars
#   - For corporate networks, use scripts/corp_ca_exec.sh wrapper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Defaults
SKIP_TEST=false
SKIP_INIT=false
BOOTSTRAP=false
TAXONOMY_FILE=""
SITES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-init)
            SKIP_INIT=true
            shift
            ;;
        --bootstrap)
            BOOTSTRAP=true
            shift
            ;;
        --taxonomy-file)
            TAXONOMY_FILE="$2"
            shift 2
            ;;
        --sites)
            SITES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-test] [--skip-init] [--bootstrap] [--taxonomy-file PATH] [--sites SITES]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Setup and Infrastructure Workflow"
echo "=========================================="
echo ""

# Step 1: Test setup
if [[ "${SKIP_TEST}" == "false" ]]; then
    echo "Step 1: Testing setup and configuration..."
    uv run python scripts/helpers/test_setup.py
    echo "✓ Setup test completed"
else
    echo "Step 1: Skipping setup test (--skip-test)"
fi

# Step 2: Initialize/reset Supabase
if [[ "${SKIP_INIT}" == "false" ]]; then
    echo ""
    echo "Step 2: Initializing/resetting Supabase schema..."
    uv run python scripts/helpers/reset_supabase_local.py
    echo "✓ Supabase schema initialized"
else
    echo "Step 2: Skipping schema init (--skip-init)"
fi

# Step 3: Bootstrap Supabase (optional)
if [[ "${BOOTSTRAP}" == "true" ]]; then
    echo ""
    echo "Step 3: Bootstrapping Supabase with data..."
    BOOTSTRAP_ARGS=()
    if [[ -n "${TAXONOMY_FILE}" ]]; then
        BOOTSTRAP_ARGS+=(--taxonomy-file "${TAXONOMY_FILE}")
    fi
    if [[ -n "${SITES}" ]]; then
        BOOTSTRAP_ARGS+=(--sites "${SITES}")
    fi
    uv run python scripts/helpers/bootstrap_supabase.py "${BOOTSTRAP_ARGS[@]}"
    echo "✓ Bootstrap completed"
else
    echo "Step 3: Skipping bootstrap (use --bootstrap to enable)"
fi

echo ""
echo "=========================================="
echo "Setup workflow completed!"
echo "=========================================="
