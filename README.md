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
| `test.ipynb` | DINOv2 / HDBSCAN exploration |

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
