"""Pretrained retrieval -> local matching -> COLMAP -> validated submission.

Run `python -m imc.pipeline --help`. No training labels are read during inference.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch

from .data import DATA_ROOT, list_images
from .evaluate import score_submission, write_submission
from .retrieval import embed_images, load_dinov2, pick_device, shortlist_pairs


@dataclass
class Config:
    k: int = 20
    descriptor: str = 'cls'
    global_size: int = 448
    exhaustive_below: int = 40
    max_keypoints: int = 512
    resize: int = 512
    min_inliers: int = 30
    min_inlier_ratio: float = 0.2
    max_error: float = 4.0  # original image pixels, not resized pixels
    seed: int = 0

    def __post_init__(self):
        for field in ('k', 'global_size', 'max_keypoints', 'resize'):
            if getattr(self, field) < 1:
                raise ValueError(f'{field} must be positive')
        if self.descriptor not in ('cls', 'mac'):
            raise ValueError('descriptor must be cls or mac')
        if self.min_inliers < 8 or not 0 <= self.min_inlier_ratio <= 1 or self.max_error <= 0:
            raise ValueError('Invalid geometric verification thresholds')


def resolve_device(device=None):
    selected = torch.device(device) if device else pick_device()
    if selected.type == 'xpu' and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise RuntimeError(
            f'Intel XPU requested but unavailable in this Python: {sys.executable}\n'
            f'PyTorch: {torch.__version__}\n'
            'Activate your existing Intel XPU PyTorch environment, or select its Jupyter kernel. '
            'Check torch.xpu.is_available() and the Intel graphics driver. '
            'CUDA packages are not appropriate for Intel GPUs.')
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA requested but unavailable in this Python: {sys.executable}\n'
            f'PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}\n'
            'Activate your existing GPU PyTorch environment, or select its Jupyter kernel. '
            'Use --device cpu only if you intend to run on CPU.')
    return selected


def create_matcher(device):
    from lightglue import LightGlue
    if device.type == 'xpu':
        matcher = LightGlue(features='superpoint', flash=False, mp=False, width_confidence=-1)
        # Upstream looks up this table even when pruning is disabled.
        # Copy per instance so other CPU/CUDA matchers remain unchanged.
        matcher.pruning_keypoint_thresholds = {
            **matcher.pruning_keypoint_thresholds, 'xpu': -1}
    else:
        matcher = LightGlue(features='superpoint')
    # The tested Intel stack also fails on LightGlue's output indexing/stack.
    # CPU local matching is the validated compatibility path; DINOv2 stays on XPU.
    return matcher.eval().to('cpu' if device.type == 'xpu' else device)


def reconstruction_rows(dataset, paths, reconstructions):
    """Largest model wins overlaps; every input image appears exactly once."""
    remaining = {Path(p).name for p in paths}
    rows = []
    for cluster, (_, rec) in enumerate(sorted(
            reconstructions.items(), key=lambda item: (-item[1].num_reg_images(), item[0]))):
        for image in rec.images.values():
            if image.name not in remaining or not image.has_pose:
                continue
            pose = image.cam_from_world()
            rows.append((dataset, f'cluster{cluster}', image.name,
                         pose.rotation.matrix(), pose.translation))
            remaining.remove(image.name)
    rows.extend((dataset, 'outliers', name, None, None) for name in sorted(remaining))
    return rows


def validate_rows(rows, images):
    expected = {(dataset, p.name) for dataset, paths in images.items() for p in paths}
    seen = set()
    for dataset, cluster, image, rotation, translation in rows:
        key = (dataset, image)
        if key in seen:
            raise ValueError(f'Duplicate submission image: {key}')
        seen.add(key)
        if cluster != 'outliers':
            if (np.shape(rotation) != (3, 3) or np.shape(translation) != (3,)
                    or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
                raise ValueError(f'Invalid pose for {key}')
    if seen != expected:
        raise ValueError(f'Submission mismatch: missing={expected - seen}, extra={seen - expected}')


@torch.inference_mode()
def reconstruct(paths, pairs, extractor, matcher, device, work, config):
    import pycolmap
    from lightglue.utils import load_image, rbd

    # CPU cache avoids retaining all feature tensors in GPU memory.
    features = {}
    extraction_device = next(extractor.parameters()).device
    matching_device = next(matcher.parameters()).device
    print(f'Extracting local features on {extraction_device}; matching on {matching_device}', flush=True)
    for i in sorted({i for pair in pairs for i in pair}):
        feature = extractor.extract(load_image(paths[i]).to(extraction_device), resize=config.resize)
        features[i] = {key: value.cpu() for key, value in feature.items()}
    db_path = work / 'database.db'
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, paths[0].parent, pycolmap.CameraMode.AUTO,
                           image_names=[p.name for p in paths])
    db = pycolmap.Database.open(db_path)
    kept = []
    try:
        ids = {image.name: image.image_id for image in db.read_all_images()}
        for i, feature in features.items():
            # LightGlue returns coordinates in the original image frame.
            keypoints = feature['keypoints'][0].numpy().astype(np.float32) + 0.5
            db.write_keypoints(ids[paths[i].name], keypoints)
        for i, j in pairs:
            a = {key: value.to(matching_device) for key, value in features[i].items()}
            b = {key: value.to(matching_device) for key, value in features[j].items()}
            matches = rbd(matcher({'image0': a, 'image1': b}))['matches'].cpu().numpy()
            if len(matches) < config.min_inliers:
                continue
            db.write_matches(ids[paths[i].name], ids[paths[j].name], matches.astype(np.uint32))
            kept.append((i, j))
    finally:
        db.close()
    if not kept:
        return {}
    pair_file = work / 'pairs.txt'
    pair_file.write_text(''.join(f'{paths[i].name} {paths[j].name}\n' for i, j in kept), encoding='utf-8')
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = config.min_inliers
    verification.ransac.min_inlier_ratio = config.min_inlier_ratio
    verification.ransac.max_error = config.max_error
    pycolmap.verify_matches(db_path, pair_file, verification)
    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = 3
    options.mapper.init_min_num_inliers = config.min_inliers
    options.mapper.abs_pose_min_num_inliers = 15
    return pycolmap.incremental_mapping(db_path, paths[0].parent, work / 'sparse', options)


def run_pipeline(root=DATA_ROOT, split='train', datasets=None, output='outputs/train_submission.csv',
                 work_dir='outputs/colmap', config=None, device=None,
                 dino_repo=None, dino_weights=None, evaluate=False):
    config = config or Config()
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if evaluate and split != 'train':
        raise ValueError('Ground-truth scoring is only available for train')
    images = list_images(split, root)
    if datasets is not None:
        unknown = set(datasets) - images.keys()
        if unknown:
            raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        images = {name: images[name] for name in datasets}
    if not images or not any(images.values()):
        raise ValueError('No input images found')
    for paths in images.values():
        if any(any(c.isspace() for c in p.name) for p in paths):
            raise ValueError('COLMAP pair files require image names without whitespace')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(device)
    print(f'Python: {sys.executable}; PyTorch: {torch.__version__}; device: {device}', flush=True)
    model = extractor = matcher = None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    # Keep each run separate; never delete or reuse a previous COLMAP database.
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work_dir))
    rows, stats = [], []
    for dataset, paths in images.items():
        start = time.perf_counter()
        print(f'{dataset}: {len(paths)} images', flush=True)
        pairs, recs = [], {}
        if len(paths) >= 3:
            if model is None:
                from lightglue import SuperPoint
                import pycolmap
                pycolmap.set_random_seed(config.seed)
                model = load_dinov2(device=device, repo=dino_repo, weights=dino_weights)
                # SuperPoint keypoint indexing exhausts Level Zero resources on the
                # tested Intel Arc/PyTorch build. Keep extraction on CPU for XPU runs;
                # retrieval still runs on the selected Intel GPU.
                extraction_device = torch.device('cpu') if device.type == 'xpu' else device
                extractor = SuperPoint(max_num_keypoints=config.max_keypoints).eval().to(extraction_device)
                matcher = create_matcher(device)
            descriptors = embed_images(paths, model, max_side=config.global_size)
            pairs = shortlist_pairs(descriptors[config.descriptor], k=config.k,
                                    exhaustive_below=config.exhaustive_below)
            work = run_dir / dataset
            work.mkdir()
            recs = reconstruct(paths, pairs, extractor, matcher, device, work, config)
        dataset_rows = reconstruction_rows(dataset, paths, recs)
        rows.extend(dataset_rows)
        stats.append({'dataset': dataset, 'images': len(paths), 'pairs': len(pairs),
                      'models': len(recs), 'registered': sum(r[1] != 'outliers' for r in dataset_rows),
                      'seconds': time.perf_counter() - start})
        print(stats[-1], flush=True)
    validate_rows(rows, images)
    write_submission(output, rows)
    metadata = {'config': asdict(config), 'root': str(root), 'split': split,
                'device': str(device),
                'local_feature_device': 'cpu' if device.type == 'xpu' else str(device),
                'matching_device': 'cpu' if device.type == 'xpu' else str(device),
                'work_dir': str(run_dir), 'datasets': stats}
    output.with_suffix('.run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if evaluate:
        table = score_submission(output, Path(root) / 'train_labels.csv', Path(root) / 'train_thresholds.csv')
        table.to_csv(output.with_suffix('.scores.csv'))
        print(table.round(2).to_string())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--datasets', nargs='+', help='Default: all datasets in the split')
    parser.add_argument('--output', type=Path, default=Path('outputs/submission.csv'))
    parser.add_argument('--work-dir', type=Path, default=Path('outputs/colmap'))
    parser.add_argument('--device', help='xpu, xpu:0, cpu, cuda, cuda:0, or mps')
    parser.add_argument('--dino-repo', type=Path, help='Local DINOv2 source for offline runs')
    parser.add_argument('--dino-weights', type=Path, help='Local DINOv2 ViT-S/14 checkpoint')
    parser.add_argument('--evaluate', action='store_true')
    for name, default in asdict(Config()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = vars(parser.parse_args())
    config = Config(**{key: args.pop(key) for key in asdict(Config())})
    print(run_pipeline(config=config, **args))


if __name__ == '__main__':
    main()
