#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/run_cpu_evidence.sh
bash scripts/run_gpu_evidence.sh

python seg/tools/build_final_evidence_package.py \
 --outdir final_evidence \
 --compare_dir final_evidence/02_metrics_and_compare_cpu \
 --gpu_compare_dir final_evidence/02_metrics_and_compare_gpu \
 --literace_service_dir seg/runs/literace_service \
 --segformer_infer_dir seg/runs/segformer_b3_infer_after_train
