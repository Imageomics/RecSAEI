#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Edit the settings below before running; use matching values across stages.
dataset="${DATASET:-nabirds}"
token_mode="${TOKEN_MODE:-content}" # content or cls
shards_root="${SHARDS_ROOT:-$repo_root/artifacts/saev/shards}"
runs_root="${RUNS_ROOT:-$repo_root/artifacts/saev/runs}"

case "$token_mode" in
  content)
    saev_tokens=content
    layer=-2
    d_sae=16384
    n_train=100000000
    if [[ "$dataset" == cub || "$dataset" == cars ]]; then n_train=25000000; fi
    train_batch_size=8192
    val_batch_size=16384
    dead_threshold_tokens=10000000
    ;;
  cls)
    saev_tokens=special
    layer=-1
    d_sae=4096
    n_train=2000000
    train_batch_size=512
    val_batch_size=512
    dead_threshold_tokens=100000
    ;;
  *) echo "Unknown token_mode: $token_mode" >&2; exit 1 ;;
esac

layer="${LAYER:-$layer}"
prefix="$dataset-raev2-dinov3-l-i256"
train_tag="$prefix-train-l$layer-$token_mode-t256"
val_tag="$prefix-val-l$layer-$token_mode-t256"

# Budgets count patch tokens for content, image tokens for CLS.
d_model=1024
lr="${LR:-0.0004}"
n_train="${N_TRAIN:-$n_train}"
n_val="${N_VAL:-10000000}"
train_batch_size="${TRAIN_BATCH_SIZE:-$train_batch_size}"
val_batch_size="${VAL_BATCH_SIZE:-$val_batch_size}"
n_prefixes="${N_PREFIXES:-1}" # 1 disables Matryoshka prefixes.
dead_threshold_tokens="${DEAD_THRESHOLD_TOKENS:-$dead_threshold_tokens}"
activation="${ACTIVATION:-batch-top-k}" # batch-top-k or top-k
case "$activation" in
  batch-top-k|top-k) ;;
  *) echo "Unknown activation: $activation" >&2; exit 1 ;;
esac
d_sae="${D_SAE:-$d_sae}"
top_k="${TOP_K:-32}"
run_id="${RUN_ID:-$prefix-l$layer-$token_mode-${activation//-/}-d$d_sae-k$top_k}"

cd "$repo_root"
mkdir -p "$runs_root"
if [[ -e "$runs_root/$run_id" ]]; then
  echo "Run already exists: $runs_root/$run_id" >&2
  exit 1
fi

cmd=(
  uv run --project "$repo_root" python "$repo_root/saev/launch.py" train
  --runs-root "$runs_root"
  --run-id "$run_id"
  --log-to "$repo_root/artifacts/logs/$run_id"
  --lr "$lr"
  --n-train "$n_train"
  --n-val "$n_val"
  --device cuda --no-track
  --train-data.shards "$shards_root/$train_tag"
  --train-data.layer "$layer"
  --train-data.tokens "$saev_tokens"
  --train-data.batch-size "$train_batch_size"
  --val-data.shards "$shards_root/$val_tag"
  --val-data.layer "$layer"
  --val-data.tokens "$saev_tokens"
  --val-data.batch-size "$val_batch_size"
  --sae.d-model "$d_model"
  --sae.d-sae "$d_sae"
  --objective.n-prefixes "$n_prefixes"
  --objective.dead-threshold-tokens "$dead_threshold_tokens"
  "sae.activation:$activation"
  --sae.activation.top-k "$top_k"
)
"${cmd[@]}"

if [[ ! -s "$runs_root/$run_id/checkpoint/sae.pt" ]]; then
  echo "Training did not produce a checkpoint" >&2
  exit 1
fi
