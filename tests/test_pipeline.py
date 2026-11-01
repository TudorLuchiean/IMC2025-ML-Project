import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imc.pipeline import Config, reconstruction_rows, resolve_device, run_pipeline, validate_rows
from imc.evaluate import shortlist_report
from imc.retrieval import pick_device


def fake_image(name, value):
    pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: np.eye(3)),
                           translation=np.full(3, value))
    return SimpleNamespace(name=name, has_pose=True, cam_from_world=lambda: pose)


def test_reconstruction_overlap_and_missing_images():
    paths = [Path(name) for name in ['a.png', 'b.png', 'c.png', 'lonely.png']]
    large = SimpleNamespace(num_reg_images=lambda: 3,
                            images={i: fake_image(p.name, 1) for i, p in enumerate(paths[:3])})
    small = SimpleNamespace(num_reg_images=lambda: 1, images={0: fake_image('a.png', 2)})
    rows = reconstruction_rows('d', paths, {0: small, 1: large})
    validate_rows(rows, {'d': paths})
    assert len(rows) == 4
    assert np.array_equal(next(r[4] for r in rows if r[2] == 'a.png'), np.ones(3))
    assert rows[-1] == ('d', 'outliers', 'lonely.png', None, None)


def test_validate_rejects_duplicate_and_missing():
    row = ('d', 'outliers', 'a.png', None, None)
    images = {'d': [Path('a.png')]}
    with pytest.raises(ValueError, match='Duplicate'):
        validate_rows([row, row], images)
    with pytest.raises(ValueError, match='missing'):
        validate_rows([], images)


def test_small_dataset_pipeline_without_models(tmp_path):
    root = tmp_path / 'data'
    folder = root / 'test' / 'tiny'
    folder.mkdir(parents=True)
    Image.new('RGB', (16, 16)).save(folder / 'single.png')
    output = tmp_path / 'outputs' / 'submission.csv'
    run_pipeline(root=root, split='test', output=output, work_dir=tmp_path / 'work')
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['scene'] == 'outliers'
    assert rows[0]['image'] == 'single.png'
    assert json.loads(output.with_suffix('.run.json').read_text())['datasets'][0]['images'] == 1
    with pytest.raises(ValueError, match='only available for train'):
        run_pipeline(root=root, split='test', evaluate=True)
    with pytest.raises(ValueError, match='Unknown datasets'):
        run_pipeline(root=root, split='test', datasets=['missing'])


def test_outlier_only_shortlist():
    result = shortlist_report([], ['a.png'], {'a.png': 'outliers'})
    assert result['coverage'] == result['worst'] == 0


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Config(k=0)


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='existing GPU PyTorch environment'):
        resolve_device('cuda')
    assert resolve_device('cpu').type == 'cpu'


@pytest.mark.parametrize('xpu,cuda,expected', [
    (True, False, 'xpu'), (False, True, 'cuda'),
    (False, False, 'cpu'), (True, True, 'xpu'),
])
def test_auto_device_selection(monkeypatch, xpu, cuda, expected):
    import torch
    monkeypatch.setattr(torch, 'xpu', SimpleNamespace(is_available=lambda: xpu))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    assert pick_device().type == expected
    if cuda:
        assert resolve_device('cuda').type == 'cuda'


def test_explicit_xpu_requires_available_backend(monkeypatch):
    import torch
    monkeypatch.setattr(torch, 'xpu', SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match='Intel XPU requested'):
        resolve_device('xpu')


def test_xpu_matcher_settings_do_not_change_cuda(monkeypatch):
    import torch
    from imc.pipeline import create_matcher
    class Matcher:
        pruning_keypoint_thresholds = {'cpu': -1, 'cuda': 1024}
        def __init__(self, **kwargs): self.options = kwargs
        def eval(self): return self
        def to(self, device): return self
    monkeypatch.setitem(sys.modules, 'lightglue', SimpleNamespace(LightGlue=Matcher))
    xpu = create_matcher(torch.device('xpu'))
    cuda = create_matcher(torch.device('cuda'))
    assert xpu.options['flash'] is False and xpu.options['width_confidence'] == -1
    assert xpu.pruning_keypoint_thresholds['xpu'] == -1
    assert cuda.options == {'features': 'superpoint'}
    assert 'xpu' not in cuda.pruning_keypoint_thresholds


def test_pipeline_wires_retrieval_reconstruction_and_scoring(tmp_path, monkeypatch):
    """Exercise orchestration with model/SfM boundaries stubbed, not real GPU inference."""
    import imc.pipeline as pipeline
    root = tmp_path / 'data'
    folder = root / 'train' / 'scene'
    folder.mkdir(parents=True)
    for i in range(3):
        Image.new('RGB', (16, 16)).save(folder / f'{i}.png')
    class Model:
        pruning_keypoint_thresholds = {'cpu': -1, 'cuda': 1024}
        def __init__(self, **kwargs): pass
        def eval(self): return self
        def to(self, device): return self
    monkeypatch.setitem(sys.modules, 'lightglue', SimpleNamespace(SuperPoint=Model, LightGlue=Model))
    monkeypatch.setitem(sys.modules, 'pycolmap', SimpleNamespace(set_random_seed=lambda seed: None))
    monkeypatch.setattr(pipeline, 'load_dinov2', lambda **kwargs: Model())
    monkeypatch.setattr(pipeline, 'embed_images', lambda *args, **kwargs: {'cls': np.eye(3)})
    def reconstruct(paths, pairs, extractor, matcher, device, work, config):
        assert pairs == [(0, 1), (0, 2), (1, 2)]
        assert work.is_dir()
        return {0: SimpleNamespace(num_reg_images=lambda: 3,
                                  images={i: fake_image(p.name, 1) for i, p in enumerate(paths)})}
    monkeypatch.setattr(pipeline, 'reconstruct', reconstruct)
    output = tmp_path / 'submission.csv'
    run_pipeline(root=root, output=output, work_dir=tmp_path / 'work')
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and all(row['scene'] == 'cluster0' for row in rows)
    # No labels exist: successful inference proves it does not require ground truth.
    assert not (root / 'train_labels.csv').exists()
