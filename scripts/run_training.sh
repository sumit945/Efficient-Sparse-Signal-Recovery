#!/usr/bin/env bash
# ==============================================================================
# UDiff Training Script
# ==============================================================================
#
# Launches UDiff training with default configuration and optional overrides.
#
# Usage:
#   bash scripts/run_training.sh                          # Default config
#   bash scripts/run_training.sh --config configs/mri_2d.yaml
#   bash scripts/run_training.sh --epochs 100 --lr 5e-5
#   bash scripts/run_training.sh --distributed --gpus 4
#
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------------------
CONFIG="configs/synthetic_1d.yaml"
SEED=42
DISTRIBUTED=false
NUM_GPUS=1
EXTRA_ARGS=""

# ------------------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            DISTRIBUTED=true
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/run_training.sh [OPTIONS] [-- EXTRA_TRAIN_ARGS]"
            echo ""
            echo "Options:"
            echo "  --config        Path to config file (default: configs/synthetic_1d.yaml)"
            echo "  --seed          Random seed (default: 42)"
            echo "  --distributed   Enable multi-GPU training with torchrun"
            echo "  --gpus          Number of GPUs for distributed training (default: 1)"
            echo "  --help          Show this help message"
            echo ""
            echo "Any additional arguments after '--' are passed directly to train.py."
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_training.sh"
            echo "  bash scripts/run_training.sh --config configs/mri_2d.yaml"
            echo "  bash scripts/run_training.sh --distributed --gpus 4"
            echo "  bash scripts/run_training.sh -- --epochs 100 --batch_size 32"
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        *)
            # Pass unknown args directly to train.py
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "                         UDiff Training Launcher"
echo "=============================================================================="
echo ""
echo "  Config:       ${CONFIG}"
echo "  Seed:         ${SEED}"
echo "  Distributed:  ${DISTRIBUTED}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  Extra args:   ${EXTRA_ARGS:-none}"
echo ""

# Check if config file exists
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config file '${CONFIG}' not found."
    exit 1
fi

# Check CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    echo "Warning: CUDA not available. Training will use CPU (significantly slower)."
    echo ""
}

# Create output directories
mkdir -p outputs/checkpoints
mkdir -p outputs/logs

# ------------------------------------------------------------------------------
# Launch training
# ------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="outputs/logs/train_${TIMESTAMP}.log"

echo "  Log file:     ${LOG_FILE}"
echo ""
echo "=============================================================================="
echo ""

if [[ "${DISTRIBUTED}" == true ]]; then
    echo "Launching distributed training on ${NUM_GPUS} GPUs..."
    echo ""

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29500 \
        train.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --distributed \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"
else
    echo "Launching single-GPU training..."
    echo ""

    python train.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"
fi

# ------------------------------------------------------------------------------
# Post-training summary
# ------------------------------------------------------------------------------
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=============================================================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "  Training completed successfully!"
    echo ""
    echo "  Next steps:"
    echo "    1. Check TensorBoard logs:  tensorboard --logdir outputs/logs"
    echo "    2. Evaluate the model:"
    echo "       python evaluate.py --config ${CONFIG} \\"
    echo "           --checkpoint outputs/checkpoints/best_model.pt"
    echo "    3. Run inference:"
    echo "       python inference.py --checkpoint outputs/checkpoints/best_model.pt \\"
    echo "           --input <input_file>"
else
    echo "  Training failed with exit code ${EXIT_CODE}."
    echo "  Check log file: ${LOG_FILE}"
fi
echo "=============================================================================="
