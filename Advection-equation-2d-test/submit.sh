#!/bin/bash
# SLURM job array for the PINN sweep.
# Main sweep: SWEEP="main"  → 42 configs × 5 seeds = 210 jobs → array 0-209
# Follow-up:  SWEEP="followup" → N configs × 5 seeds, adjust --array accordingly

#SBATCH --job-name=pinn_sweep
#SBATCH --array=0-209
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1               # remove this line if running CPU-only
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# ---- Cluster-specific setup (adjust for your HPC) -------------------------
module load python/3.11            # or whichever Python module your cluster has
source "$HOME/venv/bin/activate"   # path to your virtual environment
# ---------------------------------------------------------------------------

mkdir -p logs results

python train.py \
    --job_id  "$SLURM_ARRAY_TASK_ID" \
    --n_seeds 8 \
    --out_dir results

# ---------------------------------------------------------------------------
# To run the finite-difference reference solvers (fast, submit separately):
#   sbatch --wrap="python train.py --fd_config_idx 0 --out_dir results"
#   sbatch --wrap="python train.py --fd_config_idx 1 --out_dir results"
# Or just run them interactively (each takes < 1 second):
#   python train.py --fd_config_idx 0
#   python train.py --fd_config_idx 1
# ---------------------------------------------------------------------------
