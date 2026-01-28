#!/bin/bash
#SBATCH --job-name=fin_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-11                 # <--- 12 JOBS TOTAL (0 to 11)

# --- Environment setup ---
mkdir -p logs
source .venv/bin/activate

# --- Hyperparameter Grid Definition ---
# Define the values you want to test for each parameter
LRS=(1e-4)        # 3 values
WINDOWS=(30)             # 2 values
BATCHES=(64)             # 2 values

# Get the lengths of the arrays
NUM_LRS=${#LRS[@]}
NUM_WINDOWS=${#WINDOWS[@]}
NUM_BATCHES=${#BATCHES[@]}

# --- Auto-calculate Combinations ---
# Use modulo logic to map the single SLURM_ARRAY_TASK_ID to 3 indices
# This creates a "Cartesian Product" of all arrays

# 1. Get Batch index (changes fastest)
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_BATCHES ))

# 2. Get Window index (changes medium speed)
WINDOW_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_BATCHES) % NUM_WINDOWS ))

# 3. Get LR index (changes slowest)
LR_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_BATCHES * NUM_WINDOWS) ))

# Extract the actual values
MY_LR=${LRS[$LR_IDX]}
MY_WINDOW=${WINDOWS[$WINDOW_IDX]}
MY_BATCH=${BATCHES[$BATCH_IDX]}

echo "Job ID: $SLURM_ARRAY_TASK_ID"
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