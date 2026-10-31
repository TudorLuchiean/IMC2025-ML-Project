"""Global image descriptors (DINOv2) and the candidate pairs chosen from them."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pick_device():
    """Prefer Intel XPU, then CUDA, otherwise CPU. MPS can be selected explicitly."""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dinov2(name='dinov2_vits14', device=None, repo=None, weights=None):
    """DINOv2 backbone via torch.hub.

    Kaggle submissions run without internet: there, pass a local copy of the
    facebookresearch/dinov2 repo as `repo` and the checkpoint file as `weights`.
    """
    device = torch.device(device) if device is not None else pick_device()
    if device.type == 'xpu':
        # DINOv2's optional xFormers implementation targets CUDA; use PyTorch attention.
        os.environ['XFORMERS_DISABLED'] = '1'
    model = torch.hub.load(repo or 'facebookresearch/dinov2', name,
                           source='local' if repo else 'github', pretrained=weights is None)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location='cpu'))
    return model.eval().to(device)


def load_image(path, max_side=448, patch=14):
    """Normalised (3, H, W) tensor, resized keeping the aspect ratio so that the longer side
    is about `max_side` and both sides are multiples of the ViT patch size."""
    img = Image.open(path).convert('RGB')
    scale = max_side / max(img.size)
    size = tuple(max(patch, round(s * scale / patch) * patch) for s in img.size)
    x = torch.from_numpy(np.array(img.resize(size, Image.BICUBIC))).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, max_side):
        self.paths, self.max_side = list(paths), max_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], self.max_side)


@torch.inference_mode()
def embed_images(paths, model, max_side=448, num_workers=0):
    """L2-normalised global descriptors, one row per image:
    {'cls': class token, 'mac': max-pooled patch tokens}, each an (n, dim) array."""
    paths = list(paths)
    if not paths:
        raise ValueError('embed_images requires at least one image')
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(_Images(paths, max_side), batch_size=None, num_workers=num_workers)
    out = {'cls': [], 'mac': []}
    for x in loader:
        f = model.forward_features(x[None].to(device))
        out['cls'].append(f['x_norm_clstoken'].cpu())
        out['mac'].append(f['x_norm_patchtokens'].amax(dim=1).cpu())
    return {k: F.normalize(torch.cat(v), dim=1).cpu().numpy() for k, v in out.items()}


def shortlist_pairs(desc, k=20, sim_th=None, exhaustive_below=40):
    """Image pairs (i, j), i < j, worth matching.

    Each image is paired with its k most similar images (cosine similarity of the global
    descriptors), plus every pair with similarity >= sim_th; sets smaller than
    `exhaustive_below` are matched exhaustively. No image is left out, so a wrong guess only
    costs matching time: geometric verification and SfM decide which images form a scene.
    """
    if k < 1:
        raise ValueError('k must be positive')
    n = len(desc)
    if n < exhaustive_below:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
    sim = desc @ desc.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :min(k, n - 1)]
    pairs = {(min(i, j), max(i, j)) for i, row in enumerate(nearest.tolist()) for j in row}
    if sim_th is not None:
        rows, cols = np.nonzero(np.triu(sim >= sim_th, 1))
        pairs.update(zip(rows.tolist(), cols.tolist()))
    return sorted(pairs)
