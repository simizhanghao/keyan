#!/usr/bin/env bash
# Launch parallel EM full curves on 5 GPUs (run from tmux).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
OUT_DIR="${ROOT}/experiments/em_robustness_openset/results/em_full_$(date +%Y%m%d)"
mkdir -p "${OUT_DIR}/logs"

echo "OUT_DIR=${OUT_DIR}"
echo "Launch in separate tmux panes:"
echo "  GPU_ID=0 PERTURB=awgn OUT_DIR=${OUT_DIR} bash experiments/em_robustness_openset/run_em_full_one.sh"
echo "  GPU_ID=1 PERTURB=cfo OUT_DIR=${OUT_DIR} bash experiments/em_robustness_openset/run_em_full_one.sh"
echo "  GPU_ID=2 PERTURB=narrowband OUT_DIR=${OUT_DIR} bash experiments/em_robustness_openset/run_em_full_one.sh"
echo "  GPU_ID=3 PERTURB=phase_iq OUT_DIR=${OUT_DIR} bash experiments/em_robustness_openset/run_em_full_one.sh"
echo "  GPU_ID=4 PERTURB=filter_mixed OUT_DIR=${OUT_DIR} bash experiments/em_robustness_openset/run_em_full_one.sh"
