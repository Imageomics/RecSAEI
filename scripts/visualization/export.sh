#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
uv run --project "$repo_root" python -m visualization.export --config "${VIEWER_CONFIG:-scripts/visualization/viewer.yaml}"
