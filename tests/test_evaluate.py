import csv

import numpy as np
import pytest

from imc.evaluate import clustering_ceiling, score_submission, shortlist_report, write_submission

LAYOUT = {
    'alpha': {'castle': 12, 'bridge': 9, 'outliers': 4},
    'beta': {'stairs_a': 8, 'stairs_b': 7},
}


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


@pytest.fixture
def toy(tmp_path):
    """Ground-truth labels, poses and thresholds for two small datasets."""
    rng = np.random.default_rng(0)
    scene_of, poses, rows = {}, {}, []
    for dataset, scenes in LAYOUT.items():
        scene_of[dataset] = {}
        for scene, n in scenes.items():
            for k in range(n):
                image = f'{scene}_{k:03d}.png'
                scene_of[dataset][image] = scene
                poses[image] = (None, None) if scene == 'outliers' else (random_rotation(rng), 5 * rng.normal(size=3))
                rows.append((dataset, scene, image, *poses[image]))
    labels = tmp_path / 'train_labels.csv'
    write_submission(labels, rows)
    thresholds = tmp_path / 'train_thresholds.csv'
    with open(thresholds, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'scene', 'thresholds'])
        for dataset, scenes in LAYOUT.items():
            writer.writerows([dataset, scene, '0.025;0.05;0.1;0.2;0.5;1.0;2.0'] for scene in scenes if scene != 'outliers')
    return scene_of, poses, labels, thresholds


def by_scene(scene_of):
    groups = {}
    for image, scene in scene_of.items():
        groups.setdefault(scene, []).append(image)
    return groups


def test_ceiling_matches_official_metric(toy, tmp_path):
    scene_of, poses, labels, thresholds = toy
    alpha, beta = by_scene(scene_of['alpha']), by_scene(scene_of['beta'])
    clusterings = {
        'alpha': {  # castle split in two, outliers leaking into a cluster, part of bridge unclustered
            'c0': alpha['castle'][:7] + alpha['outliers'][:2],
            'c1': alpha['castle'][7:],
            'c2': alpha['bridge'][:8],
            'outliers': alpha['bridge'][8:] + alpha['outliers'][2:],
        },
        'beta': {  # two look-alike scenes merged
            'c0': beta['stairs_a'] + beta['stairs_b'][:4],
            'c1': beta['stairs_b'][4:],
        },
    }
    rows = [(dataset, cluster, image, *poses[image])
            for dataset, clusters in clusterings.items()
            for cluster, images in clusters.items()
            for image in images]
    submission = tmp_path / 'submission.csv'
    write_submission(submission, rows)
    official = score_submission(submission, labels, thresholds)

    for dataset, clusters in clusterings.items():
        ceiling = clustering_ceiling(clusters, scene_of[dataset])
        for key in ('score', 'mAA', 'clusterness'):
            assert ceiling[key] == pytest.approx(official.loc[dataset, key], abs=0.01)
    assert clustering_ceiling(clusterings['beta'], scene_of['beta'])['clusterness'] == pytest.approx(50)


def test_ceiling_reference_clusterings(toy):
    scene_of = toy[0]['alpha']
    assert clustering_ceiling(by_scene(scene_of), scene_of)['score'] == pytest.approx(100)
    assert clustering_ceiling({'outliers': list(scene_of)}, scene_of)['score'] == 0
    one = clustering_ceiling({'c0': list(scene_of)}, scene_of)
    assert one['mAA'] == pytest.approx(100)
    assert one['clusterness'] == pytest.approx(100 * 21 / 50)  # both scenes claim the whole cluster


def test_ground_truth_scores_100(toy, tmp_path):
    scene_of, poses, labels, thresholds = toy
    table = score_submission(labels, labels, thresholds)
    assert list(table.index) == ['alpha', 'beta', 'mean']
    assert np.allclose(table.to_numpy(), 100)


def test_only_submitted_datasets_are_scored(toy, tmp_path):
    scene_of, poses, labels, thresholds = toy
    rows = [('beta', scene, image, *poses[image]) for image, scene in scene_of['beta'].items()]
    submission = tmp_path / 'beta.csv'
    write_submission(submission, rows)
    assert list(score_submission(submission, labels, thresholds).index) == ['beta', 'mean']


def test_shortlist_report():
    images = ['a0.png', 'a1.png', 'a2.png', 'a3.png', 'b0.png', 'b1.png', 'x0.png']
    scene_of = {'a0.png': 'a', 'a1.png': 'a', 'a2.png': 'a', 'a3.png': 'a',
                'b0.png': 'b', 'b1.png': 'b', 'x0.png': 'outliers'}
    report = shortlist_report([(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)], images, scene_of)
    assert report['pairs'] == 5
    assert report['of_all'] == pytest.approx(5 / 21)
    assert report['cross'] == pytest.approx(2 / 5)  # a3-b0 and b1-x0
    assert report['coverage'] == pytest.approx((3 / 4 + 1) / 2)  # a3 is cut off from a0-a2
    assert report['worst'] == pytest.approx(3 / 4)
    assert report['isolated'] == 1
