#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LITERACE_CKPT="seg/runs/literace_boundary_degradation/best.pth"
SEGFORMER_CKPT="seg/transformer_b3/checkpoints/segformer_b3_best.pth"

if [[ ! -f "$LITERACE_CKPT" ]]; then
 echo "[ERROR] LiteRaceSegNet checkpoint not found: $LITERACE_CKPT"
 exit 1
fi
if [[ ! -f "$SEGFORMER_CKPT" ]]; then
 echo "[ERROR] SegFormer-B3 checkpoint not found: $SEGFORMER_CKPT"
 exit 1
fi

python seg/compare/compare_models.py \
 --configs seg/config/pothole_binary_literace_train.yaml seg/config/pothole_binary_segformer_b3_train.yaml \
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer \
 --ckpts "$LITERACE_CKPT" "$SEGFORMER_CKPT" \
 --input_dir datasets/pothole_binary/processed/val/images \
 --mask_dir datasets/pothole_binary/processed/val/masks \
 --device cpu \
 --batch_size 1 \
 --latency_warmup 10 \
 --latency_repeats 50 \
 --outdir final_evidence/02_metrics_and_compare_cpu
