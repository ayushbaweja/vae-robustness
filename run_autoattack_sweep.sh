#!/bin/bash
# ── AutoAttack-Style Full Sweep ──────────────────────────────────────────────
# Runs the complete matrix: 5 models × 6 epsilons
# Skips experiments that already have results (summary.json exists)
#
# Usage:
#   bash run_autoattack_sweep.sh
#   bash run_autoattack_sweep.sh sd15
#   bash run_autoattack_sweep.sh flux1 flux2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")"

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

SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0

run_experiment() {
    local model=$1
    local output_dir=$2
    local eps=$3
    local apgd_steps
    local square_steps
    apgd_steps=$(get_apgd_steps "$eps")
    square_steps=$(get_square_steps "$eps")

    local result_dir="${output_dir}/${model}_eps_${eps}"

    if [ -f "${result_dir}/summary.json" ]; then
        echo "SKIP: ${result_dir} (already exists)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  autoattack_vae.py | model=$model | eps=$eps"
    echo "  apgd_steps=$apgd_steps | square_steps=$square_steps"
    echo "  Output: $result_dir"
    echo "════════════════════════════════════════════════════════════════"

    if uv run python autoattack_vae.py \
        --model "$model" \
        --output_dir "$output_dir" \
        --epsilon "$eps" \
        --apgd_steps "$apgd_steps" \
        --square_steps "$square_steps"; then
        RUN_COUNT=$((RUN_COUNT + 1))
    else
        echo "FAILED: model=$model eps=$eps"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

run_sd15() {
    for eps in "${EPSILONS[@]}"; do
        run_experiment sd15 results/autoattack "$eps"
    done
}

run_flux1() {
    for eps in "${EPSILONS[@]}"; do
        run_experiment flux1 results/autoattack "$eps"
    done
}

run_flux2() {
    for eps in "${EPSILONS[@]}"; do
        run_experiment flux2 results/autoattack "$eps"
    done
}

run_cogvideox() {
    for eps in "${EPSILONS[@]}"; do
        run_experiment cogvideox results/autoattack "$eps"
    done
}

run_ltx() {
    for eps in "${EPSILONS[@]}"; do
        run_experiment ltx results/autoattack "$eps"
    done
}

echo "╔══════════════════════════════════════════════════╗"
echo "║     VAE Robustness — AutoAttack Sweep            ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Models:   SD1.5, FLUX.1, FLUX.2, CogVideoX, LTX║"
echo "║  Epsilons: ${EPSILONS[*]}"
echo "║  Total:    $(( ${#EPSILONS[@]} * 5 )) experiments"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ $# -eq 0 ]; then
    run_sd15
    run_flux1
    run_flux2
    run_cogvideox
    run_ltx
else
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
