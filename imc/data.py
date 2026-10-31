"""Image lists and scene labels of the competition data."""
import csv
import os
from collections import defaultdict
from pathlib import Path

# Folder holding train/, test/, train_labels.csv, ... Set IMC_DATA to use a copy elsewhere,
# e.g. /kaggle/input/image-matching-challenge-2025 on Kaggle.
DATA_ROOT = Path(os.environ.get('IMC_DATA', 'image-matching-challenge-2025'))

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def list_images(split='train', root=DATA_ROOT):
    """{dataset: sorted image paths} for every dataset folder of a split."""
    split_dir = Path(root, split)
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f'Competition image folder not found: {split_dir.resolve()}\n'
            'Download and extract Image Matching Challenge 2025 data first. '
            'Pass --root to the directory containing train/ and test/ '
            '(not to train/ itself), or set IMC_DATA. '
            'The path is relative to the current working directory when not absolute.')
    return {
        d.name: sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for d in sorted(split_dir.iterdir())
        if d.is_dir()
    }


def read_scenes(labels_csv):
    """{dataset: {image name: scene}} from train_labels.csv; outlier images have scene 'outliers'."""
    scenes = defaultdict(dict)
    with open(labels_csv, newline='') as f:
        for row in csv.DictReader(f):
            scenes[row['dataset']][row['image']] = row['scene']
    return dict(scenes)
