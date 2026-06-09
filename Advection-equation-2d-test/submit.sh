#!/bin/sh
# LSF job array for the PINN sweep — DTU HPC
#
# Submit:       bsub < submit.sh
# Check status: bstat
#
# Main sweep: SWEEP="main" in configs.py → 42 configs × 5 seeds = 210 jobs
# Follow-up:  SWEEP="followup"           → adjust [0-N] below accordingly

### Job name and array indices
#BSUB -J "pinn_sweep[0-209]"

### Queue
#BSUB -q hpc

### Cores (single node)
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### Memory: 4 GB per core → 16 GB total
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB

### Walltime (hh:mm) — 8 hours should be safe for 10k epochs
#BSUB -W 08:00

### Output and error files (%J = job id, %I = array index)
#BSUB -o logs/pinn_%J_%I.out
#BSUB -e logs/pinn_%J_%I.err

### Email when job finishes
#BSUB -N

# ---------------------------------------------------------------------------
# Environment — you already have a "pytorch" conda environment
# ---------------------------------------------------------------------------
source activate pytorch

mkdir -p logs results

python train.py \
    --job_id  "$LSB_JOBINDEX" \
    --n_seeds 5 \
    --out_dir results

# ---------------------------------------------------------------------------
# Finite-difference reference (fast — run interactively, not via job array):
#   python train.py --fd_config_idx 0
#   python train.py --fd_config_idx 1
# ---------------------------------------------------------------------------
