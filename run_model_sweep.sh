#!/bin/bash
# Run all epsilon/loss combinations for a single model on a specific GPU.
# Usage: bash run_model_sweep.sh <model> <gpu_id>
#   model: sd15, flux1, flux2, cogvideox, ltx
#   gpu_id: 0, 1, 2, 3

set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?Usage: $0 <model> <gpu_id>}
GPU=${2:?Usage: $0 <model> <gpu_id>}

export CUDA_VISIBLE_DEVICES=$GPU

EPSILONS=(0.02 0.04 0.06 0.1 0.15 0.2)
LOSSES=(pixel latent)

get_alpha() {
    case $1 in
        0.02) echo 0.005 ;; 0.04) echo 0.007 ;; 0.06) echo 0.01 ;;
        0.1) echo 0.015 ;; 0.15) echo 0.02 ;; 0.2) echo 0.02 ;;
    esac
}

get_iters() {
    case $1 in
        0.02|0.04|0.06|0.1) echo 40 ;; 0.15) echo 50 ;; 0.2) echo 60 ;;
    esac
}

case $MODEL in
    sd15)      SCRIPT=pgd_sd15_vae.py;      OUTDIR=results/sd15_pgd ;;
    flux1)     SCRIPT=pgd_flux_vae.py;       OUTDIR=results/flux1_pgd ;;
    flux2)     SCRIPT=pgd_flux2_vae.py;      OUTDIR=results/flux2_pgd ;;
    cogvideox) SCRIPT=pgd_cogvideox_vae.py;  OUTDIR=results/cogvideox_pgd ;;
    ltx)       SCRIPT=pgd_ltx_vae.py;        OUTDIR=results/ltx_pgd ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

SKIP=0
RUN=0

echo "[GPU $GPU] Starting sweep for $MODEL ($SCRIPT)"

for eps in "${EPSILONS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        result_dir="${OUTDIR}/eps_${eps}_${loss}"
        if [ -f "${result_dir}/summary.json" ]; then
            echo "[GPU $GPU] SKIP: $MODEL eps=$eps loss=$loss"
            SKIP=$((SKIP + 1))
            continue
        fi

        alpha=$(get_alpha "$eps")
        iters=$(get_iters "$eps")

        echo "[GPU $GPU] RUN: $MODEL eps=$eps loss=$loss alpha=$alpha iters=$iters"
        uv run python "$SCRIPT" \
            --output_dir "$OUTDIR" \
            --epsilon "$eps" \
            --alpha "$alpha" \
            --num_iter "$iters" \
            --loss "$loss"
        RUN=$((RUN + 1))
    done
done

echo "[GPU $GPU] Done: $MODEL — Ran $RUN, Skipped $SKIP"
