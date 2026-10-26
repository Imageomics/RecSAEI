# RecSAEI

RecSAEI is a research toolkit for interpreting vision representations with sparse
autoencoders (SAEs) and visualizing their effects through image reconstruction.
Researchers can inspect a feature, change its activation, and compare the
resulting image with the original and an unedited reconstruction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `core/data/` | Dataset staging and token shuffling. |
| `core/selection/` | Dimension selection for feature interventions. |
| `core/intervention/` | RAEv2 backend and patch/CLS feature interventions. |
| `visualization/` | Viewer export, API, live runtime, and browser interface. |
| `scripts/data/`, `scripts/sae/`, `scripts/visualization/` | Workflow entry points and viewer configuration. |
| `tests/` | Checks for the local workflows and intervention behavior. |
| `saev/`, `RAEv2/` | Bundled dependencies; see their respective licenses. |

## Prerequisites

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

## Citation

If you use RecSAEI in your research, please cite our forthcoming arXiv preprint.
The entry below is a template: replace the provisional title, `YEAR`, and
`ARXIV_ID` once the preprint is available. Author order is shown as intended for
the paper.

```bibtex
@article{gu2026recsaei,
  title         = {From Activation to Intervention: Interpreting Sparse Visual Features through Reconstruction},
  author        = {Gu, Jianyang and Beattie, Jacob and Khurana, Mridul and
                   Chae, Julia and Xiao, Xi and Zhang, Cheng and
                   Karpatne, Anuj and Chao, Wei-Lun and Beery, Sara and
                   Berger-Wolf, Tanya},
  journal       = {arXiv preprint arXiv:ARXIV_ID},
  year          = {YEAR},
  eprint        = {ARXIV_ID},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/ARXIV_ID}
}
```

## Acknowledgments

This repository builds on [saev](https://github.com/Imageomics/saev) and
[RAEv2](https://github.com/nanovisionx/RAEv2). Please also credit these projects
when using their software or methods. Their upstream citation sources are
[saev's citation metadata](https://github.com/Imageomics/saev/blob/main/CITATION.cff)
and the [RAEv2 paper](https://arxiv.org/abs/2605.18324).
