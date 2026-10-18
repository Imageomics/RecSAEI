#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Edit the settings below before running; use matching values across stages.
dataset="${DATASET:-nabirds}"
token_mode="${TOKEN_MODE:-content}" # content or cls
shards_root="${SHARDS_ROOT:-$repo_root/artifacts/saev/shards}"
runs_root="${RUNS_ROOT:-$repo_root/artifacts/saev/runs}"
batch_size="${INFERENCE_BATCH_SIZE:-16384}"

case "$token_mode" in
  content)
    saev_tokens=content
    layer=-2
    d_sae=16384
    ;;
  cls)
    saev_tokens=special
    layer=-1
    d_sae=4096
    ;;
  *) echo "Unknown token_mode: $token_mode" >&2; exit 1 ;;
esac

layer="${LAYER:-$layer}"
prefix="$dataset-raev2-dinov3-l-i256"
train_tag="$prefix-train-l$layer-$token_mode-t256"
val_tag="$prefix-val-l$layer-$token_mode-t256"
activation="${ACTIVATION:-batch-top-k}" # batch-top-k or top-k
case "$activation" in
  batch-top-k|top-k) ;;
  *) echo "Unknown activation: $activation" >&2; exit 1 ;;
esac
d_sae="${D_SAE:-$d_sae}"
top_k="${TOP_K:-32}"
run_id="${RUN_ID:-$prefix-l$layer-$token_mode-${activation//-/}-d$d_sae-k$top_k}"

cd "$repo_root"

for tag in "$train_tag" "$val_tag"; do
  cmd=(
    uv run --project "$repo_root" python "$repo_root/saev/launch.py" inference
    --run "$runs_root/$run_id"
    --log-to "$repo_root/artifacts/logs/$run_id"
    --device cuda
    --data.shards "$shards_root/$tag"
    --data.layer "$layer"
    --data.tokens "$saev_tokens"
    --data.batch-size "$batch_size"
  )
  "${cmd[@]}"
done
