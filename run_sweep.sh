#!/bin/bash
# ── Full Experiment Sweep ────────────────────────────────────────────────────
# Runs the complete matrix: 5 models × 2 loss modes × 6 epsilons
# Skips experiments that already have results (summary.json exists)
#
# Usage:
#   bash run_sweep.sh              # Run everything
#   bash run_sweep.sh sd15         # Run only SD1.5
#   bash run_sweep.sh flux1 flux2  # Run only FLUX models
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")"

EPSILONS=(0.02 0.04 0.06 0.1 0.15 0.2)
LOSSES=(pixel latent)

# Alpha and num_iter scaled to epsilon
get_alpha() {
    case $1 in
        0.02) echo 0.005 ;;
        0.04) echo 0.007 ;;
        0.06) echo 0.01 ;;
        0.1)  echo 0.015 ;;
        0.15) echo 0.02 ;;
        0.2)  echo 0.02 ;;
    esac
}

get_iters() {
    case $1 in
        0.02|0.04|0.06|0.1) echo 40 ;;
        0.15) echo 50 ;;
        0.2)  echo 60 ;;
    esac
}

SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0

run_experiment() {
    local script=$1
    local output_dir=$2
    local eps=$3
    local loss=$4
    local alpha
    local iters
    alpha=$(get_alpha "$eps")
    iters=$(get_iters "$eps")

    local result_dir="${output_dir}/eps_${eps}_${loss}"

    if [ -f "${result_dir}/summary.json" ]; then
        echo "SKIP: ${result_dir} (already exists)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $script | eps=$eps | loss=$loss | alpha=$alpha | iters=$iters"
    echo "  Output: $result_dir"
    echo "════════════════════════════════════════════════════════════════"

    if uv run python "$script" \
        --output_dir "$output_dir" \
        --epsilon "$eps" \
        --alpha "$alpha" \
        --num_iter "$iters" \
        --loss "$loss"; then
        RUN_COUNT=$((RUN_COUNT + 1))
    else
        echo "FAILED: $script eps=$eps loss=$loss"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ── Model definitions ────────────────────────────────────────────────────────
# Each model: (short_name, script, output_dir)

run_sd15() {
    for eps in "${EPSILONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            run_experiment pgd_sd15_vae.py results/sd15_pgd "$eps" "$loss"
        done
    done
}

run_flux1() {
    for eps in "${EPSILONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            run_experiment pgd_flux_vae.py results/flux1_pgd "$eps" "$loss"
        done
    done
}

run_flux2() {
    for eps in "${EPSILONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            run_experiment pgd_flux2_vae.py results/flux2_pgd "$eps" "$loss"
        done
    done
}

run_cogvideox() {
    for eps in "${EPSILONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            run_experiment pgd_cogvideox_vae.py results/cogvideox_pgd "$eps" "$loss"
        done
    done
}

run_ltx() {
    for eps in "${EPSILONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            run_experiment pgd_ltx_vae.py results/ltx_pgd "$eps" "$loss"
        done
    done
}

# ── Main ─────────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════╗"
echo "║     VAE Robustness — Full Experiment Sweep       ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Models:   SD1.5, FLUX.1, FLUX.2, CogVideoX, LTX║"
echo "║  Epsilons: ${EPSILONS[*]}"
echo "║  Losses:   ${LOSSES[*]}"
echo "║  Total:    $(( ${#EPSILONS[@]} * ${#LOSSES[@]} * 5 )) experiments"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ $# -eq 0 ]; then
    # Run all models — grouped by model to maximize HF cache reuse
    run_sd15
    run_flux1
    run_flux2
    run_cogvideox
    run_ltx
else
    # Run only requested models
    for model in "$@"; do
        case $model in
            sd15)       run_sd15 ;;
            flux1)      run_flux1 ;;
            flux2)      run_flux2 ;;
            cogvideox)  run_cogvideox ;;
            ltx)        run_ltx ;;
            *) echo "Unknown model: $model (valid: sd15, flux1, flux2, cogvideox, ltx)"; exit 1 ;;
        esac
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Sweep complete!"
echo "  Ran: $RUN_COUNT | Skipped: $SKIP_COUNT | Failed: $FAIL_COUNT"
echo "════════════════════════════════════════════════════════════════"
