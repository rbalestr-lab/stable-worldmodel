#!/bin/bash

################################################################################
# Ablation  Script for I-JEPA style and V-JEPA style Models on financial data
#
# Training Period: 3 months (2023-01-03 to 2023-03-31)
# Test Period: 1 month (2023-04-03 to 2023-04-28)
#
# Comprehensive ablations: 13 I-JEPA + 20 V-JEPA = 33 total experiments
#
# I-JEPA ablations (13):
#   - Baseline: mask=0.3, hidden=128, momentum=0.996 (1)
#   - Mask ratio: 0.1, 0.2, 0.4, 0.5 (4 new)
#   - Hidden dim: 64, 256, 512 (3 new)
#   - Momentum: 0.99, 0.995, 0.998, 0.999 (4 new)
#   - Optimal combination (1)
#
# V-JEPA ablations (20):
#   - Baseline: context=10, predict=5, hidden=128, momentum=0.996 (1)
#   - Context length: 5, 15, 20, 30 (4 new)
#   - Predict length: 3, 7, 10, 15 (4 new)
#   - Hidden dim: 64, 256, 512 (3 new)
#   - Momentum: 0.99, 0.995, 0.998, 0.999 (4 new)
#   - Temporal ratios: 4:1, 1:2, 6:1 (3 unique)
#   - Optimal combination (1)
#
# By default uses ALL available tickers from NYSE/NASDAQ (~3600 stocks)
# For quick testing, set TICKERS="AAPL MSFT GOOGL" below
################################################################################

set -e  # Exit on error

# Configuration
# Leave TICKERS empty to use all available tickers (default)
# For quick testing, uncomment: TICKERS="AAPL MSFT GOOGL"
TICKERS=""
EPOCHS=50
BATCH_SIZE=64
NUM_ENVS=4
PRED_EPOCHS=50
PRED_EPISODES=100
SEED=42  # Fixed seed for reproducibility across all experiments
EPISODES=100
STEPS_PER_EPISODE=252  # ~1 trading year worth of steps

# Date ranges
TRAIN_START="2023-01-03"
TRAIN_END="2023-03-31"
TEST_START="2023-04-03"
TEST_END="2023-04-28"

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="ablation_results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Log file
LOG_FILE="${RESULTS_DIR}/ablation_log.txt"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to run experiment and save results
run_experiment() {
    local model_name=$1
    local experiment_name=$2
    shift 2
    local args=("$@")

    local exp_dir="${RESULTS_DIR}/${model_name}_${experiment_name}"
    mkdir -p "${exp_dir}"

    # Record start time
    local start_time
    start_time=$(date +%s)

    log "=========================================="
    log "Starting: ${model_name} - ${experiment_name}"
    log "=========================================="

    # Save experiment configuration as JSON
    cat > "${exp_dir}/config.json" << EOF
{
    "model": "${model_name}",
    "experiment": "${experiment_name}",
    "tickers": "${TICKERS:-all}",
    "train_start": "${TRAIN_START}",
    "train_end": "${TRAIN_END}",
    "test_start": "${TEST_START}",
    "test_end": "${TEST_END}",
    "seed": ${SEED},
    "episodes": ${EPISODES},
    "steps_per_episode": ${STEPS_PER_EPISODE},
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "num_envs": ${NUM_ENVS},
    "pred_epochs": ${PRED_EPOCHS},
    "pred_episodes": ${PRED_EPISODES},
    "additional_args": "${args[*]}"
}
EOF

    # Build ticker argument
    local ticker_arg=""
    if [ -n "${TICKERS}" ]; then
        ticker_arg="--tickers ${TICKERS}"
    fi
    # If TICKERS is empty, script will use all available tickers by default

    # Run training
    # shellcheck disable=SC2086
    if python "scripts/train/${model_name}.py" \
    ${ticker_arg} \
    --train-start ${TRAIN_START} \
    --train-end ${TRAIN_END} \
    --test-start ${TEST_START} \
    --test-end ${TEST_END} \
    --seed ${SEED} \
    --steps-per-episode "${STEPS_PER_EPISODE}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-envs ${NUM_ENVS} \
    --pred-epochs ${PRED_EPOCHS} \
    --pred-episodes ${PRED_EPISODES} \
    "${args[@]}" \
    2>&1 | tee "${exp_dir}/training_log.txt"; then

        # Record end time
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log "✓ Completed: ${model_name} - ${experiment_name} (${duration}s)"

        # Move generated files to experiment directory
        mv ijepa_worldmodel_*.pt "${exp_dir}/" 2>/dev/null || true
        mv ijepa_worldmodel_loss_*.png "${exp_dir}/" 2>/dev/null || true
        mv ijepa_pred_head_*.pt "${exp_dir}/" 2>/dev/null || true
        mv vjepa_worldmodel_*.pt "${exp_dir}/" 2>/dev/null || true
        mv vjepa_worldmodel_loss_*.png "${exp_dir}/" 2>/dev/null || true
        mv vjepa_pred_head_*.pt "${exp_dir}/" 2>/dev/null || true

        # Extract metrics from log
        IC=$(grep "Mean Information Coefficient:" "${exp_dir}/training_log.txt" | tail -1 | awk '{print $4}')
        FINAL_LOSS=$(grep "Loss:" "${exp_dir}/training_log.txt" | grep "Epoch ${EPOCHS}/${EPOCHS}" | awk '{print $5}' | tr -d ',')
        PRED_LOSS=$(grep "Prediction Epoch ${PRED_EPOCHS}/${PRED_EPOCHS}" "${exp_dir}/training_log.txt" | awk '{print $5}')

        # Save to CSV (simple format)
        echo "${experiment_name},${IC}" >> "${RESULTS_DIR}/${model_name}_results.csv"

        # Save detailed results as JSON
        cat > "${exp_dir}/results.json" << EOF
{
    "model": "${model_name}",
    "experiment": "${experiment_name}",
    "ic": ${IC:-null},
    "final_world_model_loss": ${FINAL_LOSS:-null},
    "final_prediction_loss": ${PRED_LOSS:-null},
    "training_time_seconds": ${duration},
    "status": "success"
}
EOF

        log "IC Score: ${IC}, Final Loss: ${FINAL_LOSS}, Pred Loss: ${PRED_LOSS}"
    else
        log "✗ Failed: ${model_name} - ${experiment_name}"
        echo "${experiment_name},ERROR" >> "${RESULTS_DIR}/${model_name}_results.csv"

        # Save failure status
        cat > "${exp_dir}/results.json" << EOF
{
    "model": "${model_name}",
    "experiment": "${experiment_name}",
    "status": "failed"
}
EOF
    fi

    log ""
}

################################################################################
# I-JEPA Ablations
################################################################################

log "==============================================================================="
log "Starting I-JEPA Ablations (19 experiments)"
log "==============================================================================="
log ""

# Initialize results CSV
echo "Experiment,IC" > "${RESULTS_DIR}/ijepa_worldmodel_results.csv"

# -----------------------------------
# 1. BASELINE
# -----------------------------------
run_experiment "ijepa_worldmodel" "baseline" \
--mask-ratio 0.3 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 2. MASK RATIO ABLATION (5 experiments)
#    Test effect of masking proportion
# -----------------------------------
log "Mask Ratio Ablation Series"
run_experiment "ijepa_worldmodel" "mask_0.1" \
--mask-ratio 0.1 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "ijepa_worldmodel" "mask_0.2" \
--mask-ratio 0.2 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "ijepa_worldmodel" "mask_0.4" \
--mask-ratio 0.4 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "ijepa_worldmodel" "mask_0.5" \
--mask-ratio 0.5 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 3. MODEL CAPACITY ABLATION (4 experiments)
#    Test effect of model size
# -----------------------------------
log "Model Capacity Ablation Series"
run_experiment "ijepa_worldmodel" "hidden_64" \
--mask-ratio 0.3 \
--hidden-dim 64 \
--momentum 0.996

run_experiment "ijepa_worldmodel" "hidden_256" \
--mask-ratio 0.3 \
--hidden-dim 256 \
--momentum 0.996

run_experiment "ijepa_worldmodel" "hidden_512" \
--mask-ratio 0.3 \
--hidden-dim 512 \
--momentum 0.996

# -----------------------------------
# 4. MOMENTUM ABLATION (5 experiments)
#    Test target encoder update rate
# -----------------------------------
log "Momentum Ablation Series"
run_experiment "ijepa_worldmodel" "momentum_0.99" \
--mask-ratio 0.3 \
--hidden-dim 128 \
--momentum 0.99

run_experiment "ijepa_worldmodel" "momentum_0.995" \
--mask-ratio 0.3 \
--hidden-dim 128 \
--momentum 0.995

run_experiment "ijepa_worldmodel" "momentum_0.998" \
--mask-ratio 0.3 \
--hidden-dim 128 \
--momentum 0.998

run_experiment "ijepa_worldmodel" "momentum_0.999" \
--mask-ratio 0.3 \
--hidden-dim 128 \
--momentum 0.999

# -----------------------------------
# 5. COMBINATION: BEST FROM EACH
# -----------------------------------
log "Testing optimal combination"
run_experiment "ijepa_worldmodel" "optimal_combo" \
--mask-ratio 0.3 \
--hidden-dim 256 \
--momentum 0.996
################################################################################
# V-JEPA Ablations
################################################################################

log "==============================================================================="
log "Starting V-JEPA Ablations (20 experiments)"
log "==============================================================================="
log ""

# Initialize results CSV
echo "Experiment,IC" > "${RESULTS_DIR}/vjepa_worldmodel_results.csv"

# -----------------------------------
# 1. BASELINE
# -----------------------------------
run_experiment "vjepa_worldmodel" "baseline" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 2. CONTEXT LENGTH ABLATION (5 experiments)
#    Test effect of historical window size
# -----------------------------------
log "Context Length Ablation Series"
run_experiment "vjepa_worldmodel" "context_5" \
--context-len 5 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "context_15" \
--context-len 15 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "context_20" \
--context-len 20 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "context_30" \
--context-len 30 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 3. PREDICTION LENGTH ABLATION (5 experiments)
#    Test effect of forecast horizon
# -----------------------------------
log "Prediction Length Ablation Series"
run_experiment "vjepa_worldmodel" "predict_3" \
--context-len 10 \
--predict-len 3 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "predict_7" \
--context-len 10 \
--predict-len 7 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "predict_10" \
--context-len 10 \
--predict-len 10 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "predict_15" \
--context-len 10 \
--predict-len 15 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 4. MODEL CAPACITY ABLATION (4 experiments)
#    Test effect of model size
# -----------------------------------
log "Model Capacity Ablation Series"
run_experiment "vjepa_worldmodel" "hidden_64" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 64 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "hidden_256" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 256 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "hidden_512" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 512 \
--momentum 0.996

# -----------------------------------
# 5. MOMENTUM ABLATION (5 experiments)
#    Test target encoder update rate
# -----------------------------------
log "Momentum Ablation Series"
run_experiment "vjepa_worldmodel" "momentum_0.99" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.99

run_experiment "vjepa_worldmodel" "momentum_0.995" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.995

run_experiment "vjepa_worldmodel" "momentum_0.998" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.998

run_experiment "vjepa_worldmodel" "momentum_0.999" \
--context-len 10 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.999

# -----------------------------------
# 6. TEMPORAL RATIO TESTS (3 experiments)
#    Test different context/predict ratios beyond baseline (2:1)
# -----------------------------------
log "Temporal Ratio Ablation Series"
run_experiment "vjepa_worldmodel" "ratio_4to1" \
--context-len 20 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "ratio_1to2" \
--context-len 5 \
--predict-len 10 \
--hidden-dim 128 \
--momentum 0.996

run_experiment "vjepa_worldmodel" "ratio_6to1" \
--context-len 30 \
--predict-len 5 \
--hidden-dim 128 \
--momentum 0.996

# -----------------------------------
# 7. COMBINATION: BEST FROM EACH
# -----------------------------------
log "Testing optimal combination"
run_experiment "vjepa_worldmodel" "optimal_combo" \
--context-len 15 \
--predict-len 5 \
--hidden-dim 256 \
--momentum 0.996

################################################################################
# Summary
################################################################################

log "==============================================================================="
log "All Ablations Complete!"
log "==============================================================================="
log ""
log "Results saved to: ${RESULTS_DIR}"
log ""
log "I-JEPA Results:"
cat "${RESULTS_DIR}/ijepa_worldmodel_results.csv" | tee -a "${LOG_FILE}"
log ""
log "V-JEPA Results:"
cat "${RESULTS_DIR}/vjepa_worldmodel_results.csv" | tee -a "${LOG_FILE}"
log ""

# Create summary visualization script
cat > "${RESULTS_DIR}/plot_results.py" << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Read results
ijepa_df = pd.read_csv('ijepa_worldmodel_results.csv')
vjepa_df = pd.read_csv('vjepa_worldmodel_results.csv')

# Filter out errors and convert to float
ijepa_df = ijepa_df[ijepa_df['IC'] != 'ERROR']
vjepa_df = vjepa_df[vjepa_df['IC'] != 'ERROR']
ijepa_df['IC'] = ijepa_df['IC'].astype(float)
vjepa_df['IC'] = vjepa_df['IC'].astype(float)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# I-JEPA
colors_ijepa = ['steelblue' if ic >= 0 else 'coral' for ic in ijepa_df['IC']]
ax1.barh(ijepa_df['Experiment'], ijepa_df['IC'], color=colors_ijepa)
ax1.set_xlabel('Information Coefficient (IC)', fontsize=12)
ax1.set_title('I-JEPA Ablation Study Results', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax1.grid(axis='x', alpha=0.3)

# V-JEPA
colors_vjepa = ['steelblue' if ic >= 0 else 'coral' for ic in vjepa_df['IC']]
ax2.barh(vjepa_df['Experiment'], vjepa_df['IC'], color=colors_vjepa)
ax2.set_xlabel('Information Coefficient (IC)', fontsize=12)
ax2.set_title('V-JEPA Ablation Study Results', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ablation_comparison.png")

# Summary statistics
print("\n" + "="*60)
print("I-JEPA Summary Statistics")
print("="*60)
print(f"Mean IC: {ijepa_df['IC'].mean():.4f}")
print(f"Std IC:  {ijepa_df['IC'].std():.4f}")
print(f"Best:    {ijepa_df.loc[ijepa_df['IC'].idxmax(), 'Experiment']} ({ijepa_df['IC'].max():.4f})")

print("\n" + "="*60)
print("V-JEPA Summary Statistics")
print("="*60)
print(f"Mean IC: {vjepa_df['IC'].mean():.4f}")
print(f"Std IC:  {vjepa_df['IC'].std():.4f}")
print(f"Best:    {vjepa_df.loc[vjepa_df['IC'].idxmax(), 'Experiment']} ({vjepa_df['IC'].max():.4f})")

# Create LaTeX table
print("\n" + "="*60)
print("LaTeX Table (copy-paste ready)")
print("="*60)

# Load detailed results if available
try:
    detailed_df = pd.read_csv('all_results_detailed.csv')

    print("\n% I-JEPA Results")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{I-JEPA Ablation Study Results}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Experiment & IC & World Model Loss & Pred. Loss & Time (s) \\\\")
    print("\\midrule")

    ijepa_detailed = detailed_df[detailed_df['model'] == 'ijepa_worldmodel']
    for _, row in ijepa_detailed.iterrows():
        exp = row['experiment'].replace('_', '\\_')
        ic = f"{row['ic']:.4f}" if pd.notna(row['ic']) else "---"
        wm_loss = f"{row['final_world_model_loss']:.4f}" if pd.notna(row['final_world_model_loss']) else "---"
        pred_loss = f"{row['final_prediction_loss']:.4f}" if pd.notna(row['final_prediction_loss']) else "---"
        time = f"{int(row['training_time_seconds'])}" if pd.notna(row['training_time_seconds']) else "---"
        print(f"{exp} & {ic} & {wm_loss} & {pred_loss} & {time} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% V-JEPA Results")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{V-JEPA Ablation Study Results}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Experiment & IC & World Model Loss & Pred. Loss & Time (s) \\\\")
    print("\\midrule")

    vjepa_detailed = detailed_df[detailed_df['model'] == 'vjepa_worldmodel']
    for _, row in vjepa_detailed.iterrows():
        exp = row['experiment'].replace('_', '\\_')
        ic = f"{row['ic']:.4f}" if pd.notna(row['ic']) else "---"
        wm_loss = f"{row['final_world_model_loss']:.4f}" if pd.notna(row['final_world_model_loss']) else "---"
        pred_loss = f"{row['final_prediction_loss']:.4f}" if pd.notna(row['final_prediction_loss']) else "---"
        time = f"{int(row['training_time_seconds'])}" if pd.notna(row['training_time_seconds']) else "---"
        print(f"{exp} & {ic} & {wm_loss} & {pred_loss} & {time} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

except FileNotFoundError:
    print("(Detailed results CSV not found, skipping LaTeX table generation)")

print("\n" + "="*60)
EOF

log "To visualize results, run:"
log "  cd ${RESULTS_DIR} && python plot_results.py"
log ""
log "Total runtime: $SECONDS seconds"

# Create master results file (JSON format for easy analysis)
python3 << 'PYEOF'
import json
import glob
import pandas as pd

# Collect all results.json files
results = []
for result_file in glob.glob('*/results.json'):
    with open(result_file, 'r') as f:
        data = json.load(f)
        # Also load config
        config_file = result_file.replace('results.json', 'config.json')
        try:
            with open(config_file, 'r') as cf:
                config = json.load(cf)
                data['config'] = config
        except:
            pass
        results.append(data)

# Save master JSON
with open('all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create comprehensive CSV
df_data = []
for r in results:
    row = {
        'model': r.get('model'),
        'experiment': r.get('experiment'),
        'ic': r.get('ic'),
        'final_world_model_loss': r.get('final_world_model_loss'),
        'final_prediction_loss': r.get('final_prediction_loss'),
        'training_time_seconds': r.get('training_time_seconds'),
        'status': r.get('status'),
    }
    if 'config' in r:
        row.update({
            'mask_ratio': None,
            'context_len': None,
            'predict_len': None,
            'hidden_dim': None,
            'momentum': None,
        })
        # Parse additional_args
        args = r['config'].get('additional_args', '')
        for arg in args.split():
            if '--mask-ratio' in arg:
                continue
            elif arg.replace('.', '').replace('-', '').isdigit():
                if '--mask-ratio' in args and row['mask_ratio'] is None:
                    row['mask_ratio'] = float(arg)
                elif '--context-len' in args and row['context_len'] is None:
                    row['context_len'] = int(arg)
                elif '--predict-len' in args and row['predict_len'] is None:
                    row['predict_len'] = int(arg)
                elif '--hidden-dim' in args and row['hidden_dim'] is None:
                    row['hidden_dim'] = int(arg)
                elif '--momentum' in args and row['momentum'] is None:
                    row['momentum'] = float(arg)
    df_data.append(row)

df = pd.DataFrame(df_data)
df.to_csv('all_results_detailed.csv', index=False)

print("\n" + "="*80)
print("Master results files created:")
print("  - all_results.json (structured data)")
print("  - all_results_detailed.csv (for analysis/tables)")
print("="*80)
PYEOF
