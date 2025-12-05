#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "$project_root"

if [[ -f .venv/bin/activate ]]; then
  # Ensure project tools (black, ruff, mypy, pytest) are on PATH for hooks
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

make quality-check
