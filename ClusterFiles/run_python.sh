#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=60
#SBATCH --job-name=imc
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Generic launcher: runs any python script on a compute node.
#
#   sbatch run_python.sh check_env.py                         # first: verify the environment
#   sbatch --time=360 run_python.sh FullPipeline_refactored.py   # then: the real run (360 min = QoS max)
#
# Command-line options of sbatch override the #SBATCH lines above.
# QoS 'normal' limits (see MyTCinfo): cpu<=20, mem<=64G, 1 GPU, 2 jobs, 6 h.

SCRIPT=${1:?usage: sbatch run_python.sh <script.py>}

module load anaconda
source activate imc2025

cd "$SLURM_SUBMIT_DIR"          # the scripts use paths relative to the project root
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi                      # fine here: this runs on a compute node
python -u "$SCRIPT"
echo "Finished $(date)"
