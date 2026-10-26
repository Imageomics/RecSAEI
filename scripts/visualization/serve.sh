#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
uv run --project "$repo_root" python -m visualization.api \
  --export-dir "${VIEWER_EXPORT:-artifacts/viewer}" \
  --config "${VIEWER_CONFIG:-scripts/visualization/viewer.yaml}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
