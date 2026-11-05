#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rl4ppt-final"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required." >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  ENV_TOOL="mamba"
  YES_FLAG="--yes"
else
  ENV_TOOL="conda"
  YES_FLAG=""
fi

cd "$ROOT"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$ENV_TOOL" env update $YES_FLAG --name "$ENV_NAME" --file environment.yml --prune
else
  "$ENV_TOOL" env create --yes --file environment.yml
fi

ENV_PREFIX="$(conda env list | awk -v name="$ENV_NAME" '$1 == name {print $NF}')"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "Cannot locate the $ENV_NAME environment after installation." >&2
  exit 1
fi
install -D scripts/conda/activate.d/rl4ppt-final.sh \
  "$ENV_PREFIX/etc/conda/activate.d/rl4ppt-final.sh"
install -D scripts/conda/deactivate.d/rl4ppt-final.sh \
  "$ENV_PREFIX/etc/conda/deactivate.d/rl4ppt-final.sh"

PATH="$ENV_PREFIX/bin:$PATH" conda run --name "$ENV_NAME" npm ci
PATH="$ENV_PREFIX/bin:$PATH" conda run --name "$ENV_NAME" \
  python -m playwright install chromium
PATH="$ENV_PREFIX/bin:$PATH" conda run --name "$ENV_NAME" \
  python scripts/check_environment.py

echo "Environment ready. Activate it with: conda activate $ENV_NAME"
