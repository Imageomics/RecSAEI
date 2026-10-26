# RecSAEI

## Prerequisites

From the repository root, install the environment with `uv sync`.

Prepare the required encoder and RAE decoder weights from
`nyu-visionx/RAEv2-models`. For example, download DINOv3-L and its k1 decoder:

```bash
uv run hf download nyu-visionx/RAEv2-models \
  encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  stage1/imagenet/dinov3l-k1/decoder.pt \
  stage1/imagenet/dinov3l-k1/stats.pt \
  --local-dir RAEv2/pretrained_models
```

Adjust the model paths for other encoders or decoders.

## Data staging and SAE training

```bash
bash scripts/data/stage.sh
bash scripts/sae/extract.sh
bash scripts/sae/shuffle.sh
bash scripts/sae/train.sh
bash scripts/sae/infer.sh
```

Edit settings in the scripts or override them through environment variables.
Outputs default to `data/staged/<dataset>`, `artifacts/saev/shards`, and
`artifacts/saev/runs`; override `STAGED_ROOT`, `SHARDS_ROOT`, and `RUNS_ROOT`
as needed. Keep settings consistent across stages. Patch training uses shuffled
shards; validation and inference use image-ordered shards. CLS skips shuffling.

## Feature viewer

Edit `scripts/visualization/viewer.yaml`, then run:

```bash
bash scripts/visualization/export.sh
bash scripts/visualization/serve.sh
```

The viewer binds to `0.0.0.0:8000`; open `http://<server-hostname>:8000`.
Add patch/CLS inference sources as needed;
class selection uses the optional training `fit` source. Omit `runtime` for
browsing only. Exports reference the staged images, which must remain accessible.
