#!/bin/bash
#SBATCH --job-name=fin_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- Environment setup ---
mkdir -p logs
source .venv/bin/activate

# --- Hyperparameters ---
MY_LR=1e-4
MY_WINDOW=30
MY_BATCH=64

echo "Job ID: $SLURM_JOB_ID"
echo "Running with: LR=$MY_LR, Window=$MY_WINDOW, Batch=$MY_BATCH"

# --- Run the Script ---
python scripts/train/supervised_baseline.py \
    --train_start 2022-01-03 \
    --train_end 2023-12-31 \
    --test_start 2024-01-02 \
    --test_end 2024-03-31 \
    --lr $MY_LR \
    --window_size $MY_WINDOW \
    --batch_size $MY_BATCH \
    --max_epochs 10 \
    --results_file "financial_results.json"