# IMC2025-ML-Project
Repo for our groups project work. The challenge tackled is the [Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025)

## Setup

```
conda env create -f envReq.yml
conda activate IMC2025
```

The competition data goes in `image-matching-challenge-2025/` at the repo root (accept the competition rules on Kaggle first):

```
kaggle competitions download -c image-matching-challenge-2025
unzip image-matching-challenge-2025.zip -d image-matching-challenge-2025
```

To read the data from somewhere else (the GPU cluster, or `/kaggle/input/image-matching-challenge-2025` on Kaggle), set the `IMC_DATA` environment variable to that folder.

## Layout

| Path | Contents |
|---|---|
| `imc/data.py` | image lists and scene labels |
| `imc/retrieval.py` | DINOv2 global descriptors and the top-k pair shortlist |
| `imc/evaluate.py` | train-set evaluation: clustering ceiling, shortlist report, official score |
| `imc/metric.py` | official IMC2025 metric (from the `imc25-utils` Kaggle dataset) |
| `evaluation.ipynb` | runs those checks on the train set |
| `imc/pipeline.py` | shared inference pipeline and command-line entry point |
| `test.ipynb` | runs the shared pipeline and scores train predictions |
| `KaggleNotebook.ipynb`, `copying work.ipynb` | historical, incomplete experiments; not runnable entry points |

## Evaluation

`evaluation.ipynb` scores each stage against `train_labels.csv`:

- **Clustering ceiling**: the best score a scene clustering can reach if every clustered image gets a perfect pose. Each scene is scored only on the cluster holding most of its images, so splitting a scene costs mAA, merging scenes costs clusterness, and images left in `outliers` never count towards mAA.
- **Pair shortlist**: how much of each scene the candidate pairs can still reconstruct, and how many pairs need matching.
- **Submission**: the official metric on a submission csv. From the command line:

```
python -m imc.evaluate outputs/train_submission.csv
```

The metric aligns cameras over every triplet of images, so scoring all of train takes over 10 minutes; score a few datasets while iterating.

Tests: `pytest`

## Run the integrated baseline

Run commands from the repository root in the `IMC2025` conda environment.
The pipeline uses pretrained DINOv2 ViT-S/14, SuperPoint and LightGlue. It does
**not train neural network weights**. COLMAP estimates camera poses and 3D points
separately for each dataset; ground-truth labels are used only afterwards for scoring.

1. Activate your existing GPU PyTorch environment, or create a new environment using the setup commands above. Download/extract the data. Do not recreate an existing working GPU environment.
2. Check dependencies and GPU visibility:

```sh
python -c "import torch, torchvision, lightglue, pycolmap; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('XPU:', hasattr(torch, 'xpu') and torch.xpu.is_available()); print(pycolmap.__version__)"
python -m pytest -q
```

The CLI and main notebook auto-select Intel XPU, then NVIDIA CUDA, then CPU.
Pass `--device xpu` or `--device cuda` to require a particular GPU; an unavailable
backend raises an error showing the Python interpreter instead of silently falling
back. Select the corresponding existing PyTorch environment as the notebook kernel.
CUDA users keep the normal LightGlue settings. On XPU only, we disable LightGlue's
CUDA FlashAttention/mixed-precision path and point pruning, add its missing XPU
threshold entry, and disable DINOv2's optional CUDA-oriented xFormers path before
loading DINOv2. Restart the kernel if DINOv2 was already imported with xFormers.
On the tested Intel Arc / PyTorch 2.15.0+xpu environment, SuperPoint and LightGlue
indexing/stack operations fail with Level Zero OUT_OF_RESOURCES. XPU runs therefore
use CPU for both local feature extraction and matching, while DINOv2 uses XPU.
CUDA runs continue to use CUDA for all three models. The selected stage devices
are printed and recorded in the run JSON.
COLMAP is a separate native component: selecting XPU for PyTorch does not enable
Intel GPU acceleration for COLMAP; expect CPU SfM work on the Intel machine.

For existing Intel environments, keep your working XPU PyTorch installation.
`envReq.yml` retains the original team's PyTorch pins; it is not an Intel XPU
installation recipe. If creating a separate Intel environment, follow the
[official PyTorch XPU instructions](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
and install the remaining project dependencies without replacing that XPU build.
Do not install CUDA wheels on the Intel-only machine.

The first normal run needs internet to download pretrained weights and DINOv2 source.

3. Start with one dataset (use a folder name present under your `train/`):

```sh
python -m imc.pipeline --split train --datasets ETs --output outputs/ets_baseline.csv --evaluate
```

Use `--root /path/to/image-matching-challenge-2025` to override the data location
without environment variables. Alternatively, in PowerShell set
`$env:IMC_DATA = 'D:\data\image-matching-challenge-2025'` before starting Python.
`test.ipynb` exposes the same pipeline as editable notebook configuration.

Outputs are the CSV, a `.run.json` with parameters/registration counts/timing,
and (with `--evaluate`) a `.scores.csv`. Each run preserves its own COLMAP database
and sparse reconstructions under `outputs/colmap/run-*`; these can consume disk space.
Reusing an output filename replaces that CSV and its reports.

## Validation and tuning (instead of training)

Choose fixed train datasets for parameter tuning and keep other complete datasets
held out for the final report. Do not randomly split images of the same scene across
tuning and final evaluation. Do not tune against test/leaderboard results.

Use `evaluation.ipynb` to compare CLS/MAC retrieval and shortlist coverage, then
compare full reconstruction scores, registered-image counts and runtime. Change one
parameter at a time, using a distinct output filename for each experiment:

```sh
python -m imc.pipeline --datasets ETs --k 30 --max-keypoints 2048 --resize 1024 --output outputs/ets_larger.csv --evaluate
python -m imc.pipeline --datasets ETs --descriptor mac --output outputs/ets_mac.csv --evaluate
```

The default is 512 keypoints at resize 512, preserving the old coarse baseline.
There is no second fine-matching pass. `--exhaustive-below 40` matches small datasets
exhaustively, so changing k does not affect those datasets. `--max-error` is COLMAP's
RANSAC error threshold in original-image pixels (default 4), and
`--min-inlier-ratio` is the RANSAC sampling assumption, not a post-filter on the
measured inlier fraction. `--min-inliers` controls the minimum verified support.
Geometric verification now uses COLMAP's own estimator, so numerical results are
not expected to exactly match the old OpenCV RANSAC notebook.

For a final held-out evaluation, supply those dataset names with `--datasets` and
the selected fixed parameters. Omit `--datasets` to process all datasets.
Scoring train predicts on labelled data; it is not evidence of unseen-data quality
if those same datasets were repeatedly used for tuning. The official scorer can be slow.

## Test inference and Kaggle

```sh
python -m imc.pipeline --split test --output outputs/submission.csv
```

Test inference does not read labels and cannot calculate a local ground-truth score.
Every input image receives exactly one row; unregistered images are `outliers`
with NaN poses. Images appearing in multiple models are assigned to the largest model.
The CSV schema is `dataset,scene,image,rotation_matrix,translation_vector`.

The original [2025 competition requirements](https://www.kaggle.com/competitions/image-matching-challenge-2025/overview)
specified notebook submissions, internet disabled, and a 9-hour CPU/GPU runtime limit.
Its original deadline was June 2, 2025. There is also a separate
[ongoing competition](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing);
check which competition/data version your course expects before submitting.

For an offline Kaggle notebook:

- Attach this repository as an input dataset and add its folder to `sys.path`.
- Prepare compatible dependency wheels for the Kaggle Python version, install with
  `pip --no-index --find-links /kaggle/input/YOUR-WHEELS ...`, or use preinstalled dependencies.
  A Windows environment is not a portable Kaggle environment.
- In a preparation environment with internet, instantiate DINOv2, SuperPoint and
  LightGlue once. Save DINOv2's source repository and ViT-S/14 checkpoint, and the
  Torch Hub cache containing the SuperPoint and LightGlue checkpoints as Kaggle inputs.
- Copy the prepared Torch Hub cache to a writable directory and set `TORCH_HOME`
  to that cache's parent structure **before constructing models**. A complete cache
  contains `TORCH_HOME/hub/checkpoints/`. Missing local-matcher weights otherwise
  trigger downloads even when DINOv2 is supplied locally.
- Pass `dino_repo` and `dino_weights` to `run_pipeline`, set `root` to the competition
  input directory, `split='test'`, `datasets=None`, and
  `output='/kaggle/working/submission.csv'`.
- Run the whole notebook with internet disabled before committing a submission.
  Hidden test data can be larger than the visible sample: measure time and memory.

See the upstream [LightGlue setup](https://github.com/cvg/LightGlue) and
[pycolmap API](https://colmap.github.io/pycolmap/pycolmap.html) for model and SfM dependencies.

## Verification status and limitations

Unit tests cover retrieval/evaluation, duplicate reconstructions, missing images,
and a small end-to-end CSV run without downloaded models. A real DINOv2/LightGlue/
COLMAP run requires the dependencies, weights and competition data. A real 20-image
Brandenburg Gate smoke test completed on Intel Arc with DINOv2 on XPU and local
features/matching on CPU: 19 registered images in one reconstruction, about 74 seconds.
The CSV contains 20 unique rows with finite registered poses. This subset has no
ground-truth labels, so there is no measured competition score here.
Model/library versions and GPU kernels may change results despite the fixed seed.
The legacy `imc25_utils/metric.py` is retained for old notebooks; the integrated
pipeline exclusively uses `imc/metric.py` through `imc.evaluate`.
