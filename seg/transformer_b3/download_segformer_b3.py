"""Download/cache HuggingFace SegFormer-B3 for the capstone baseline.

Purpose:
- This script prepares a local Transformer-family baseline folder.
- It does not produce a trained pothole detector by itself.
- The ADE/Cityscapes-style pretrained checkpoint can initialize the encoder,
 but the 2-class pothole segmentation head still requires fine-tuning.

Typical Windows usage:
  02_SETUP_SEGFORMER_B3_HF.bat

Manual usage:
  python seg/transformer_b3/download_segformer_b3.py \
    --model-id nvidia/segformer-b3-finetuned-ade-512-512 \
    --outdir seg/transformer_b3/hf_pretrained/segformer_b3_ade \
    --write-config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _resolve(path_text: str) -> Path:
  p = Path(path_text)
  if p.is_absolute():
    return p
  return (_project_root() / p).resolve()


def main() -> int:
  parser = argparse.ArgumentParser(description="Download local SegFormer-B3 HuggingFace baseline")
  parser.add_argument(
    "--model-id",
    default="nvidia/segformer-b3-finetuned-ade-512-512",
    help="HuggingFace model id or local model folder",
  )
  parser.add_argument(
    "--outdir",
    default="seg/transformer_b3/hf_pretrained/segformer_b3_ade",
    help="Local output folder inside the project",
  )
  parser.add_argument(
    "--write-config",
    action="store_true",
    help="Also write seg/config/pothole_binary_segformer_b3_hf.yaml when missing",
  )
  args = parser.parse_args()

  try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
  except Exception as exc:
    print("[ERROR] transformers is not installed.")
    print("    Run: python -m pip install -r requirements_transformer_optional.txt")
    print(f"    Detail: {exc}")
    return 1

  root = _project_root()
  outdir = _resolve(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Project root: {root}")
  print(f"[INFO] Downloading/loading: {args.model_id}")
  print(f"[INFO] Saving local folder: {outdir}")

  # Keep the original pretrained label count when saving. The adapter later loads
  # it with num_labels=2 and ignore_mismatched_sizes=True, which reinitializes
  # the final classifier for binary pothole segmentation.
  config = SegformerConfig.from_pretrained(args.model_id)
  model = SegformerForSemanticSegmentation.from_pretrained(args.model_id, config=config)
  model.save_pretrained(outdir, safe_serialization=True)
  config.save_pretrained(outdir)

  manifest = {
    "role": "Transformer-family comparison baseline, not a replacement for the CNN/lightweight model",
    "source_model_id": args.model_id,
    "local_folder": str(outdir.relative_to(root) if outdir.is_relative_to(root) else outdir),
    "important_note": (
      "This is a general semantic-segmentation pretrained checkpoint. "
      "For pothole_binary, the 2-class segmentation head must be fine-tuned."
    ),
    "recommended_config": "seg/config/pothole_binary_segformer_b3_hf.yaml",
  }
  local_folder_for_yaml = manifest["local_folder"].replace("\\", "/")
  with open(outdir / "CAPSTONE_SEGFORMER_B3_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

  if args.write_config:
    cfg_path = root / "seg" / "config" / "pothole_binary_segformer_b3_hf.yaml"
    if cfg_path.exists():
      print(f"[INFO] Config already exists: {cfg_path.relative_to(root)}")
    else:
      cfg_text = f"""seed: 42
device: cpu
num_workers: 0
save_dir: runs/segformer_b3_hf_pothole

data:
 root: ../datasets/pothole_binary/processed
 ignore_index: 255
 class_names:
  - background
  - pothole

model:
 name: segformer_b3
 num_classes: 2
 use_aux: false
 variant: b3
 pretrained: true
 hf_model_name: {local_folder_for_yaml}

train:
 image_size: [192, 320]
 batch_size: 2
 epochs: 20
 amp: false
 base_lr: 0.00006
 weight_decay: 0.01
 grad_clip: 1.0
 print_freq: 1
 save_every: 1
 early_stopping_patience: 6
 boundary_width: 3
 eval_mode: composite
 lane_class_ids: []
 pothole_class_ids: [1]
 class_weights: [0.35, 0.65]
 loss:
  ce_weight: 1.0
  dice_weight: 0.6
  aux_weight: 0.0
  boundary_weight: 0.0
  focal_gamma: 1.5

val:
 batch_size: 1

optimizer:
 name: adamw

scheduler:
 name: poly
 power: 0.9
 min_lr: 0.000001

infer:
 overlay_alpha: 0.45
 palette:
  - [30, 30, 30]
  - [255, 90, 0]
"""
      cfg_path.write_text(cfg_text, encoding="utf-8")
      print(f"[OK] Wrote config: {cfg_path.relative_to(root)}")

  print("[OK] SegFormer-B3 HuggingFace baseline prepared.")
  print("[NEXT] Use run_COMPARE_MODELS_HF.bat for architecture/cost comparison.")
  print("[NOTE] For real pothole mIoU, fine-tune and save a pothole checkpoint first.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
