"""Evaluation against the training labels.

clustering_ceiling  best score a scene clustering can reach (every clustered image posed perfectly)
shortlist_report    how much of each scene a set of candidate pairs can still reconstruct
score_submission    official IMC2025 metric on a submission csv

Scores are in %, like the leaderboard.
"""
import contextlib
import csv
import io
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import metric

TO_DEC = 3  # the metric discounts 3 cameras per scene: any 3 camera centres align exactly


def clustering_ceiling(clusters, scene_of):
    """Best score, mAA and clusterness one dataset can reach with this clustering.

    Follows the official metric with perfect poses: each scene is scored on the cluster that
    holds most of its images (the first one on ties), mAA counts that cluster's images of the
    scene minus 3 over the scene size minus 3, clusterness is the share of those clusters'
    images that belong to their scene, and the score is the harmonic mean of the two.

    clusters: {cluster name: [image names]}; images in 'outliers' or in no cluster are unclustered
    scene_of: {image name: scene} for the whole dataset
    """
    counts = {c: Counter(scene_of[Path(i).name] for i in images)
              for c, images in clusters.items() if c != 'outliers'}
    registered = possible = correct = clustered = 0
    for scene, size in Counter(scene_of.values()).items():
        if scene == 'outliers':
            continue
        best = max(counts, key=lambda c: counts[c][scene], default=None)
        n = counts[best][scene] if best is not None else 0
        registered += max(0, n - TO_DEC)
        possible += size - TO_DEC
        correct += n
        clustered += sum(counts[best].values()) if best is not None else 0
    mAA = registered / possible if possible > 0 else 0.0
    clusterness = correct / clustered if clustered else 0.0
    score = 2 * mAA * clusterness / (mAA + clusterness) if mAA + clusterness else 0.0
    return {'score': 100 * score, 'mAA': 100 * mAA, 'clusterness': 100 * clusterness}


def shortlist_report(pairs, images, scene_of):
    """How well candidate pairs (i, j) into `images` cover the scenes of one dataset.

    pairs     number of candidate pairs (matching cost)
    of_all    share of all possible pairs
    cross     share of candidates joining different scenes or outliers (wasted, or wrong merges)
    coverage  share of a scene in its largest group of linked same-scene candidates, averaged
              over scenes: a reconstruction cannot correctly register more of the scene
    worst     lowest coverage of any scene
    isolated  scene images without a single same-scene candidate (can never be registered)
    """
    scenes = [scene_of[Path(i).name] for i in images]
    n = len(scenes)
    same = [(i, j) for i, j in pairs if scenes[i] == scenes[j] != 'outliers']
    rows, cols = np.array(same, dtype=int).reshape(-1, 2).T
    _, group = connected_components(coo_matrix((np.ones(len(same)), (rows, cols)), shape=(n, n)), directed=False)
    coverage = []
    for scene in sorted(set(scenes) - {'outliers'}):
        members = [i for i in range(n) if scenes[i] == scene]
        coverage.append(Counter(group[members]).most_common(1)[0][1] / len(members))
    linked = Counter(i for pair in same for i in pair)
    return {
        'pairs': len(pairs),
        'of_all': len(pairs) / max(1, n * (n - 1) // 2),
        'cross': 1 - len(same) / len(pairs) if pairs else 0.0,
        'coverage': float(np.mean(coverage)),
        'worst': min(coverage),
        'isolated': sum(1 for i in range(n) if scenes[i] != 'outliers' and not linked[i]),
    }


def write_submission(path, rows):
    """Write (dataset, cluster, image, R, t) rows as a submission csv.

    R (3x3) and t (3) are the cam_from_world pose as COLMAP gives it, None for unregistered
    images. Rows of the 'outliers' cluster go last: the metric breaks ties by row order, and
    this way a scene left entirely in 'outliers' counts as missed, as in clustering_ceiling.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector'])
        for dataset, cluster, image, R, t in sorted(rows, key=lambda row: row[1] == 'outliers'):
            writer.writerow([dataset, cluster, image, _join(R, 9), _join(t, 3)])


def _join(values, n):
    return ';'.join(['nan'] * n if values is None else (f'{v:.9f}' for v in np.ravel(values)))


def score_submission(submission_csv, labels_csv, thresholds_csv):
    """Official score, mAA and clusterness per dataset and on average.
    Only the datasets that appear in the submission are scored."""
    with open(submission_csv, newline='') as f:
        datasets = {row['dataset'] for row in csv.DictReader(f)}
    with tempfile.TemporaryDirectory() as tmp:
        gt_csv = Path(tmp, 'labels.csv')
        with open(labels_csv, newline='') as src, open(gt_csv, 'w', newline='') as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, reader.fieldnames)
            writer.writeheader()
            writer.writerows(row for row in reader if row['dataset'] in datasets)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, (per_dataset, _, _) = metric.score(
                gt_csv=gt_csv, user_csv=submission_csv, thresholds_csv=thresholds_csv, verbose=True)
    lines = re.findall(r'^(\S+): score=\S+% \(mAA=([\d.]+)%, clusterness=([\d.]+)%\)', log.getvalue(), re.M)
    table = pd.DataFrame(lines, columns=['dataset', 'mAA', 'clusterness']).set_index('dataset').astype(float)
    table.insert(0, 'score', pd.Series(per_dataset))
    table.loc['mean'] = table.mean()
    return table


if __name__ == '__main__':
    import argparse

    from .data import DATA_ROOT

    parser = argparse.ArgumentParser(description='Score a submission csv against the training labels.')
    parser.add_argument('submission')
    parser.add_argument('--labels', default=DATA_ROOT / 'train_labels.csv')
    parser.add_argument('--thresholds', default=DATA_ROOT / 'train_thresholds.csv')
    args = parser.parse_args()
    print(score_submission(args.submission, args.labels, args.thresholds).round(2).to_string())
