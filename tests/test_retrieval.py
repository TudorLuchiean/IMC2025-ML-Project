import numpy as np
import torch
from PIL import Image

from imc.retrieval import embed_images, load_image, shortlist_pairs


class TinyViT(torch.nn.Module):
    """Stand-in for DINOv2 with the same forward_features outputs."""

    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 8, kernel_size=14, stride=14)

    def forward_features(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return {'x_norm_clstoken': tokens.mean(dim=1), 'x_norm_patchtokens': tokens}


def test_shortlist_pairs_keeps_k_nearest():
    desc = np.random.default_rng(0).normal(size=(60, 16))
    pairs = shortlist_pairs(desc, k=5)
    assert pairs == sorted(set(pairs)) and all(i < j for i, j in pairs)
    assert np.bincount(np.ravel(pairs), minlength=60).min() >= 5
    unit = desc / np.linalg.norm(desc, axis=1, keepdims=True)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    assert all((min(i, j), max(i, j)) in pairs for i, j in enumerate(sim.argmax(axis=1)))


def test_shortlist_pairs_exhaustive_cases():
    desc = np.random.default_rng(1).normal(size=(50, 8))
    everything = 50 * 49 // 2
    assert len(shortlist_pairs(desc[:10])) == 45  # small sets: all pairs
    assert len(shortlist_pairs(desc, k=100)) == everything  # k >= n: all pairs, no self-pairs
    assert len(shortlist_pairs(desc, k=1, sim_th=-1.0)) == everything  # threshold adds every pair


def test_load_image_keeps_aspect_ratio(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (640, 480)).save(path)
    assert load_image(path, max_side=448).shape == (3, 336, 448)
    Image.new('RGB', (300, 1000)).save(path)
    _, h, w = load_image(path, max_side=448).shape
    assert h == 448 and w % 14 == 0 and abs(w / h - 0.3) < 0.02


def test_embed_images(tmp_path):
    rng = np.random.default_rng(2)
    paths = []
    for k, (w, h) in enumerate([(640, 480), (480, 640), (500, 500)]):
        paths.append(tmp_path / f'{k}.png')
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(paths[-1])
    desc = embed_images(paths, TinyViT().eval())
    assert set(desc) == {'cls', 'mac'}
    for d in desc.values():
        assert d.shape == (3, 8)
        assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)
