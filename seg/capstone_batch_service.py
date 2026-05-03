"""Capstone batch service runner.

Capstone rule:
- CV model creates damage mask when a trained checkpoint exists.
- Post-processing creates overlay, damage percentage, severity and JSON.
- LLM is optional and should only verbalize the produced JSON summary.

Team workflow is preserved:
1) Put images into assets/service_demo/input_batch
2) Run run_batch_infer_service.bat
3) Check seg/runs/capstone_batch_service
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _run(cmd, cwd):
  print("[RUN]", " ".join(str(x) for x in cmd))
  subprocess.check_call([str(x) for x in cmd], cwd=str(cwd))


def _copy_pred_class_masks(raw_dir: Path, mask_dir: Path):
  mask_dir.mkdir(parents=True, exist_ok=True)
  copied = 0
  for pred in raw_dir.glob("*_pred_class.png"):
    stem = pred.name.replace("_pred_class.png", "")
    target = mask_dir / f"{stem}.png"
    shutil.copy2(pred, target)
    copied += 1
  return copied


def main():
  p = argparse.ArgumentParser(description="Capstone road-damage batch service pipeline")
  p.add_argument("--input_dir", default="assets/service_demo/input_batch")
  p.add_argument("--outdir", default="seg/runs/capstone_batch_service")
  p.add_argument("--model_output_dir", default="seg/runs/capstone_model_raw_output")
  p.add_argument("--config", default="seg/config/pothole_binary.yaml")
  p.add_argument("--ckpt", default="seg/runs/literace_boundary_degradation/best.pth")
  p.add_argument("--mode", choices=["auto", "model", "cv_demo"], default="auto")
  p.add_argument("--min_area_pixels", type=int, default=80)
  p.add_argument("--no_card", action="store_true")
  p.add_argument("--no_boundary", action="store_true")
  args = p.parse_args()

  root = Path(__file__).resolve().parents[1]
  seg_dir = Path(__file__).resolve().parent
  input_dir = (root / args.input_dir).resolve()
  outdir = (root / args.outdir).resolve()
  model_output_dir = (root / args.model_output_dir).resolve()
  temp_mask_dir = (root / "seg/runs/capstone_pred_masks_for_service").resolve()
  config = (root / args.config).resolve()
  ckpt = (root / args.ckpt).resolve()

  input_dir.mkdir(parents=True, exist_ok=True)
  outdir.mkdir(parents=True, exist_ok=True)

  images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
  if not images:
    raise FileNotFoundError(f"No input images found. Put road images into: {input_dir}")

  if args.mode == "model" and not ckpt.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

  use_model = (args.mode == "model") or (args.mode == "auto" and ckpt.exists())

  mode_note = {
    "pipeline": "capstone_service",
    "input_dir": str(input_dir),
    "outdir": str(outdir),
    "requested_mode": args.mode,
    "resolved_mode": "model_checkpoint" if use_model else "cv_demo_without_checkpoint",
    "cv_llm_rule": "CV/model produces mask, overlay and percentage; optional LLM only explains JSON/text.",
    "config": str(config),
    "checkpoint": str(ckpt),
    "warning": None,
  }

  if use_model:
    print("[MODE] Real model checkpoint mode")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    _run([
      sys.executable, str(seg_dir / "infer_seg.py"),
      "--config", str(config),
      "--ckpt", str(ckpt),
      "--input_dir", str(input_dir),
      "--output_dir", str(model_output_dir),
    ], cwd=root)
    if temp_mask_dir.exists():
      shutil.rmtree(temp_mask_dir)
    copied = _copy_pred_class_masks(model_output_dir, temp_mask_dir)
    if copied == 0:
      raise RuntimeError(f"No *_pred_class.png masks produced in {model_output_dir}")
    cmd = [
      sys.executable, str(seg_dir / "infer_service_visual.py"),
      "--input_dir", str(input_dir),
      "--mask_dir", str(temp_mask_dir),
      "--outdir", str(outdir),
      "--min_area_pixels", str(args.min_area_pixels),
      "--fallback_to_mock_if_bad_mask",
    ]
    if args.no_card:
      cmd.append("--no_card")
    if args.no_boundary:
      cmd.append("--no_boundary")
    _run(cmd, cwd=root)
  else:
    print("[MODE] CV demo mode without trained checkpoint")
    mode_note["warning"] = (
      "No trained checkpoint was found, so this run uses conservative CV heuristic demo mode. "
      "Use this for capstone UI/flow demonstration only, not for final accuracy claims."
    )
    cmd = [
      sys.executable, str(seg_dir / "infer_service_visual.py"),
      "--input_dir", str(input_dir),
      "--mock",
      "--outdir", str(outdir),
      "--min_area_pixels", str(args.min_area_pixels),
    ]
    if args.no_card:
      cmd.append("--no_card")
    if args.no_boundary:
      cmd.append("--no_boundary")
    _run(cmd, cwd=root)

  with open(outdir / "_CAPSTONE_SERVICE_MODE.json", "w", encoding="utf-8") as f:
    json.dump(mode_note, f, ensure_ascii=False, indent=2)

  print("\n[DONE] Capstone service output saved to:", outdir)
  print("Main files:")
  print("- *_service_overlay.png : colored result")
  print("- *_service_mask.png  : binary damage mask")
  print("- *_service_summary.json/csv : damage percent, severity, explanation")
  print("- service_batch_summary.csv/json : batch summary")


if __name__ == "__main__":
  main()
