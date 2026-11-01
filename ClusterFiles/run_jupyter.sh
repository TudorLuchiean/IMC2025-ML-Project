#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=240
#SBATCH --job-name=jupyter
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Starts JupyterLab on a compute node (for debugging / checking the environment).
#   sbatch run_jupyter.sh
#   tail error_jupyter_<jobid>.err        # shows the node IP and the URL with the access token
# Then open an SSH tunnel from your PC (while on NTU network/VPN), e.g.:
#   ssh -L 8890:<node-ip>:8890 <your-username>@10.96.189.11
# and open  http://127.0.0.1:8890/lab?token=...   in your browser.
#
# This job holds your ONLY GPU (QoS: gres/gpu=1) until it ends -> scancel it when done.
# Pick a free port first on the head node:  netstat -tp | grep 8890

PORT=8890

module load anaconda
source activate imc2025

cd "$SLURM_SUBMIT_DIR"          # becomes the Jupyter root folder
echo "Node: $(hostname)   IP: $(hostname -i)   port: $PORT"
jupyter-lab --no-browser --ip=$(hostname -i) --port=$PORT
