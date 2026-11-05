#!/usr/bin/env bash
set -euo pipefail
SLIDESWEAVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Open a shell where conda is available." >&2
  exit 1
fi
cd "$SLIDESWEAVER_ROOT"
if conda env list | awk '{print $1}' | grep -Fxq 'rl4ppt-final'; then
  conda env update --name rl4ppt-final --file environment.yml
else
  conda env create --yes --file environment.yml
fi
SLIDESWEAVER_PREFIX="$(conda run -n rl4ppt-final python -c 'import sys; print(sys.prefix)' | tail -n 1)"
install -D paths/text_first/scripts/conda/activate.d/rl4ppt-final.sh "$SLIDESWEAVER_PREFIX/etc/conda/activate.d/rl4ppt-final.sh"
install -D paths/text_first/scripts/conda/deactivate.d/rl4ppt-final.sh "$SLIDESWEAVER_PREFIX/etc/conda/deactivate.d/rl4ppt-final.sh"
PATH="$SLIDESWEAVER_PREFIX/bin:$PATH" conda run --no-capture-output -n rl4ppt-final npm ci
PATH="$SLIDESWEAVER_PREFIX/bin:$PATH" conda run --no-capture-output -n rl4ppt-final python -m playwright install chromium
PATH="$SLIDESWEAVER_PREFIX/bin:$PATH" conda run --no-capture-output -n rl4ppt-final python scripts/check_environment.py --offline
echo "Ready for offline validation. Activate rl4ppt-final and run check_environment.py --model-path /path/to/Z-Image-Turbo before live generation."
