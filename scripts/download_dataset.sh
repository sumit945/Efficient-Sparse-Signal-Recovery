#!/usr/bin/env bash
# ==============================================================================
# Dataset Download Script for UDiff
# ==============================================================================
#
# This script downloads and prepares datasets for UDiff experiments.
#
# Supported datasets:
#   - fastMRI knee singlecoil
#
# Usage:
#   bash scripts/download_dataset.sh --dataset <dataset_name> --output_dir <path>
#
# Examples:
#   bash scripts/download_dataset.sh --dataset knee_singlecoil --output_dir data/raw/fastmri
#   bash scripts/download_dataset.sh --dataset knee_singlecoil --output_dir data/raw/fastmri --split train
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------------------
DATASET="knee_singlecoil"
OUTPUT_DIR="data/raw/fastmri"
SPLIT="all"  # Options: train, val, test, all

# ------------------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/download_dataset.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset     Dataset to download (default: knee_singlecoil)"
            echo "  --output_dir  Output directory (default: data/raw/fastmri)"
            echo "  --split       Data split: train, val, test, all (default: all)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Data Use Agreement Notice
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "                    DATASET DOWNLOAD - DATA USE AGREEMENT"
echo "=============================================================================="
echo ""
echo "IMPORTANT: Before downloading the fastMRI dataset, you must:"
echo ""
echo "  1. Visit https://fastmri.med.nyu.edu/"
echo "  2. Create an account and log in"
echo "  3. Read and accept the Data Use Agreement (DUA)"
echo "  4. Obtain download credentials"
echo ""
echo "The fastMRI dataset is provided by NYU Langone Health and is subject to"
echo "a Data Use Agreement. By proceeding, you confirm that you have read and"
echo "accepted the DUA."
echo ""
echo "=============================================================================="
echo ""

read -p "Have you accepted the Data Use Agreement? (yes/no): " DUA_ACCEPTED

if [[ "${DUA_ACCEPTED}" != "yes" ]]; then
    echo "You must accept the Data Use Agreement before downloading."
    echo "Visit: https://fastmri.med.nyu.edu/"
    exit 1
fi

# ------------------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------------------
echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# Download functions
# ------------------------------------------------------------------------------
download_fastmri_knee() {
    local split="$1"
    local output_dir="$2"

    # Placeholder URLs - replace with actual fastMRI download URLs after DUA acceptance
    # The actual URLs are provided after registering at https://fastmri.med.nyu.edu/
    local BASE_URL="https://fastmri-dataset.s3.amazonaws.com"

    case "${split}" in
        train)
            echo "Downloading knee singlecoil training data..."
            echo "[PLACEHOLDER] wget -c ${BASE_URL}/knee_singlecoil_train.tar.gz -O ${output_dir}/knee_singlecoil_train.tar.gz"
            echo "[PLACEHOLDER] tar -xzf ${output_dir}/knee_singlecoil_train.tar.gz -C ${output_dir}/"
            echo ""
            echo "NOTE: Replace the placeholder URLs above with actual download links"
            echo "      obtained from https://fastmri.med.nyu.edu/ after DUA acceptance."
            ;;
        val)
            echo "Downloading knee singlecoil validation data..."
            echo "[PLACEHOLDER] wget -c ${BASE_URL}/knee_singlecoil_val.tar.gz -O ${output_dir}/knee_singlecoil_val.tar.gz"
            echo "[PLACEHOLDER] tar -xzf ${output_dir}/knee_singlecoil_val.tar.gz -C ${output_dir}/"
            echo ""
            echo "NOTE: Replace the placeholder URLs above with actual download links"
            echo "      obtained from https://fastmri.med.nyu.edu/ after DUA acceptance."
            ;;
        test)
            echo "Downloading knee singlecoil test data..."
            echo "[PLACEHOLDER] wget -c ${BASE_URL}/knee_singlecoil_test.tar.gz -O ${output_dir}/knee_singlecoil_test.tar.gz"
            echo "[PLACEHOLDER] tar -xzf ${output_dir}/knee_singlecoil_test.tar.gz -C ${output_dir}/"
            echo ""
            echo "NOTE: Replace the placeholder URLs above with actual download links"
            echo "      obtained from https://fastmri.med.nyu.edu/ after DUA acceptance."
            ;;
        all)
            download_fastmri_knee "train" "${output_dir}"
            download_fastmri_knee "val" "${output_dir}"
            download_fastmri_knee "test" "${output_dir}"
            ;;
        *)
            echo "Error: Unknown split '${split}'. Use: train, val, test, or all."
            exit 1
            ;;
    esac
}

# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "  Dataset: ${DATASET}"
echo "  Split:   ${SPLIT}"
echo "  Output:  ${OUTPUT_DIR}"
echo "=============================================================================="
echo ""

case "${DATASET}" in
    knee_singlecoil)
        download_fastmri_knee "${SPLIT}" "${OUTPUT_DIR}"
        ;;
    *)
        echo "Error: Unknown dataset '${DATASET}'."
        echo "Supported datasets: knee_singlecoil"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "  Download script completed."
echo ""
echo "  Next steps:"
echo "    1. Replace placeholder URLs with actual fastMRI download links"
echo "    2. Re-run this script to download the data"
echo "    3. Preprocess the data:"
echo "       python -m src.data.mri --input_dir ${OUTPUT_DIR} \\"
echo "           --output_dir data/processed/fastmri --resolution 320"
echo "=============================================================================="
