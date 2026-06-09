#!/bin/sh
#BSUB -J pinn_sweep
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 24:00
#BSUB -o logs/pinn_sweep_%J.out
#BSUB -e logs/pinn_sweep_%J.err
#BSUB -N
#BSUB -u s245200@dtu.dk

module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs results

START=${1:-0}
END=${2:-209}

for i in $(seq $START $END); do
    echo "======== Job $i / $END  $(date) ========"
    python train.py --job_id $i --n_seeds 5 --out_dir results
done

echo "All done at $(date)"
