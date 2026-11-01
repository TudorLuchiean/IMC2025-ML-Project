#!/bin/bash
# ---------------------------------------------------------------------
# One-time conda environment setup for the IMC2025 pipeline on TC1.
#
# Run on the HEAD NODE, in your normal SSH session:   bash setup_env.sh
# Installing packages here is fine; running the pipeline here is NOT allowed
# (see the TC1 user guide, "Guideline for Job Submission", rule 1).
# ---------------------------------------------------------------------
set -e

ENV_NAME=imc2025
PY_VERSION=3.12

# cu124 wheel: bundles its own CUDA runtime (no `module load cuda` needed) and still
# supports the Tesla V100 (compute capability 7.0) in TC1.
TORCH_INDEX=https://download.pytorch.org/whl/cu124

# TIP: pin this to the version you use locally (run `pip show pycolmap` there),
#      e.g. PYCOLMAP_SPEC="pycolmap==3.12.6". The notebook uses the newer
#      Database.open(...) API, so the version matters.
PYCOLMAP_SPEC="pycolmap"

if ! type module >/dev/null 2>&1; then
    echo "The 'module' command is not available. Run this from your normal SSH login shell." >&2
    exit 1
fi

module load anaconda

if conda env list | grep -q "^${ENV_NAME} "; then
    echo ">> Environment '${ENV_NAME}' already exists, reusing it."
else
    conda create -y -n ${ENV_NAME} python=${PY_VERSION}
fi
source activate ${ENV_NAME}

# 1) conda packages first (the guide advises conda before pip)
conda install -y -c conda-forge jupyterlab ipykernel numpy pandas scikit-learn networkx tqdm pillow matplotlib

# 2) PyTorch via pip
python -m pip install torch torchvision --index-url ${TORCH_INDEX}

# 3) OpenCV (headless: no GUI libraries needed on a cluster) + COLMAP bindings
python -m pip install opencv-python-headless ${PYCOLMAP_SPEC}

# 4) LightGlue is not on conda. Install it from source as described in its README.
#    --no-deps so pip does not pull in a second, conflicting OpenCV build.
mkdir -p ~/src
cd ~/src
[ -d LightGlue ] || git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install --no-deps -e .

echo
echo ">> Done. Do NOT test on the head node (no GPU here)."
echo ">> Next:  sbatch run_python.sh check_env.py"
