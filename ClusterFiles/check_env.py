"""Environment check for the IMC2025 pipeline. Run it on a compute node:

    sbatch run_python.sh check_env.py

Prints one PASS/FAIL line per check. Run it from the project root so the relative
paths (imc25_utils/, image-matching-challenge-2025/) resolve.
"""
import socket
import sys
import traceback
from pathlib import Path

results = []


def check(name):
    def deco(fn):
        try:
            info = fn()
            results.append((name, True, info or ''))
        except Exception as e:                       # noqa: BLE001
            results.append((name, False, f'{type(e).__name__}: {e}'))
            traceback.print_exc()
        print(f'[{"PASS" if results[-1][1] else "FAIL"}] {name}: {results[-1][2]}', flush=True)
        return fn
    return deco


@check('python')
def _():
    return sys.version.split()[0]


@check('internet (github.com:443)')
def _():
    socket.create_connection(('github.com', 443), timeout=10).close()
    return 'reachable, model weights can be downloaded here'


@check('torch + CUDA')
def _():
    import torch
    assert torch.cuda.is_available(), 'CUDA not available (is this a GPU compute node? --gres=gpu:1 set?)'
    x = torch.randn(256, 256, device='cuda')
    float((x @ x).sum().item())                      # forces a real kernel launch
    return f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}'


@check('packages')
def _():
    import cv2, networkx, numpy, pandas, sklearn, torchvision, tqdm      # noqa: F401,E401
    return (f'numpy {numpy.__version__}, pandas {pandas.__version__}, cv2 {cv2.__version__}, '
            f'networkx {networkx.__version__}, sklearn {sklearn.__version__}, torchvision {torchvision.__version__}')


@check('pycolmap API used by the notebook')
def _():
    import pycolmap
    needed = ['Database', 'import_images', 'incremental_mapping', 'IncrementalPipelineOptions',
              'TwoViewGeometry', 'TwoViewGeometryConfiguration', 'CameraMode']
    missing = [n for n in needed if not hasattr(pycolmap, n)]
    assert not missing, f'missing in pycolmap {pycolmap.__version__}: {missing}'
    assert hasattr(pycolmap.Database, 'open'), f'pycolmap {pycolmap.__version__} has no Database.open (too old?)'
    return f'pycolmap {pycolmap.__version__}'


@check('lightglue: SuperPoint + LightGlue (downloads weights on first use)')
def _():
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd          # noqa: F401
    SuperPoint(max_num_keypoints=512).eval().cuda()
    LightGlue(features='superpoint').eval().cuda()
    return 'loaded'


@check('DINOv2 via torch.hub (downloads weights on first use)')
def _():
    import torch
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()
    return 'loaded'


@check('imc25_utils/metric.py')
def _():
    sys.path.append('imc25_utils')
    import metric                                         # noqa: F401
    return 'imported'


@check('dataset folder')
def _():
    train = Path('image-matching-challenge-2025/train')
    assert train.is_dir(), f'{train.resolve()} not found'
    n = {d.name: len(list(d.iterdir())) for d in sorted(train.iterdir()) if d.is_dir()}
    return f'{len(n)} datasets, {sum(n.values())} files'


print('\n=== SUMMARY ===')
failed = [r for r in results if not r[1]]
print('ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED: ' + ', '.join(r[0] for r in failed))
sys.exit(1 if failed else 0)
