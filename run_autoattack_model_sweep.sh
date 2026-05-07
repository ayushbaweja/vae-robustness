#!/bin/bash
# Run all epsilon combinations for a single model on a specific GPU.
# Usage: bash run_autoattack_model_sweep.sh <model> <gpu_id>
#   model: sd15, flux1, flux2, cogvideox, ltx
#   gpu_id: 0, 1, 2, 3

set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?Usage: $0 <model> <gpu_id>}
GPU=${2:?Usage: $0 <model> <gpu_id>}

export CUDA_VISIBLE_DEVICES=$GPU

EPSILONS=(0.02 0.04 0.06 0.1 0.15 0.2)

get_apgd_steps() {
    case $1 in
        0.02|0.04|0.06|0.1) echo 100 ;;
        0.15) echo 125 ;;
        0.2)  echo 150 ;;
    esac
}

get_square_steps() {
    case $1 in
        0.02|0.04|0.06|0.1) echo 200 ;;
        0.15) echo 250 ;;
        0.2)  echo 300 ;;
    esac
}

OUTDIR=results/autoattack
SKIP=0
RUN=0

echo "[GPU $GPU] Starting autoattack sweep for $MODEL"

for eps in "${EPSILONS[@]}"; do
    result_dir="${OUTDIR}/${MODEL}_eps_${eps}"
    if [ -f "${result_dir}/summary.json" ]; then
        echo "[GPU $GPU] SKIP: $MODEL eps=$eps"
        SKIP=$((SKIP + 1))
        continue
    fi

    apgd_steps=$(get_apgd_steps "$eps")
    square_steps=$(get_square_steps "$eps")

    echo "[GPU $GPU] RUN: $MODEL eps=$eps apgd_steps=$apgd_steps square_steps=$square_steps"
    uv run python autoattack_vae.py \
        --model "$MODEL" \
        --output_dir "$OUTDIR" \
        --epsilon "$eps" \
        --apgd_steps "$apgd_steps" \
        --square_steps "$square_steps"
    RUN=$((RUN + 1))
done

echo "[GPU $GPU] Done: $MODEL — Ran $RUN, Skipped $SKIP"
