#!/bin/bash
# Prepare RULER synthetic data for all tasks and sequence lengths.
# Usage: bash evaluate/ruler/prepare_data.sh <model_path>
#
# This downloads required datasets (Paul Graham essays, SQuAD, HotpotQA)
# and generates synthetic evaluation data at multiple sequence lengths.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

MODEL_PATH=$1

# ── Directories ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RULER_SCRIPTS="${SCRIPT_DIR}/RULER/scripts"
ROOT_DIR="${SCRIPT_DIR}/benchmark_root"
MODEL_NAME=$(basename "$MODEL_PATH")

# ── Configuration ──
NUM_SAMPLES=100
SEQ_LENGTHS="4096 8192 16384 32768 65536 131072"
TASKS="niah_multikey_3 vt qa_2"

# ── Download data if needed ──
echo "Checking/downloading required datasets..."
JSON_DIR="${RULER_SCRIPTS}/data/synthetic/json"

# Paul Graham essays
if [ ! -f "${JSON_DIR}/PaulGrahamEssays.json" ]; then
    echo "Downloading Paul Graham essays..."
    pushd "${JSON_DIR}" > /dev/null
    python download_paulgraham_essay.py
    popd > /dev/null
fi

# QA datasets (SQuAD, HotpotQA)
if [ ! -f "${JSON_DIR}/squad.json" ] || [ ! -f "${JSON_DIR}/hotpotqa.json" ]; then
    echo "Downloading QA datasets..."
    pushd "${JSON_DIR}" > /dev/null
    bash download_qa_dataset.sh
    popd > /dev/null
fi

echo "Datasets ready ✅"

# ── Generate synthetic data ──
if [ $# -ge 2 ]; then
    shift
    TASKS="$*"
fi

for MAX_SEQ_LENGTH in $SEQ_LENGTHS; do
    DATA_DIR="${ROOT_DIR}/${MODEL_NAME}/synthetic/${MAX_SEQ_LENGTH}/data"
    mkdir -p "${DATA_DIR}"

    for TASK in $TASKS; do
        echo "Preparing task=${TASK}, seq_len=${MAX_SEQ_LENGTH}..."
        python "${RULER_SCRIPTS}/data/prepare.py" \
            --save_dir "${DATA_DIR}" \
            --benchmark synthetic \
            --task "${TASK}" \
            --tokenizer_path "${MODEL_PATH}" \
            --tokenizer_type hf \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type base \
            --num_samples "${NUM_SAMPLES}"
    done
done

echo ""
echo "Data preparation complete ✅"
echo "Root dir: ${ROOT_DIR}/${MODEL_NAME}"
