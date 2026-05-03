"""Compare CNN/lightweight and Transformer segmentation candidates.

This script is intentionally conservative:
- It can compare architecture cost even when checkpoints are missing.
- It reports mIoU/PixelAcc only when both a checkpoint and ground-truth masks
 are supplied.
- A missing Transformer dependency becomes an ERROR row instead of crashing the
 whole comparison report.
- It supports both CPU profiling for deployment evidence and CUDA profiling
 for AWS/GPU acceleration evidence.
"""

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
  import torch
except Exception as exc: # pragma: no cover
  raise RuntimeError("torch is required to run model comparison") from exc

SEG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SEG_DIR.parent
if str(SEG_DIR) not in sys.path:
  sys.path.insert(0, str(SEG_DIR))

from core.data_pairs import find_best_mask_for_image
from core.model_select import get_model
from core.save import load_state
from core.train_utils import load_yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
  RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
  RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError: # Pillow<9
  RESAMPLE_BILINEAR = Image.BILINEAR
  RESAMPLE_NEAREST = Image.NEAREST


def _resolve(path: str) -> Path:
  p = Path(path)
  return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _param_count(model: torch.nn.Module) -> int:
  return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _image_size_from_cfg(cfg: Dict) -> Tuple[int, int]:
  size = cfg.get("train", {}).get("image_size", [192, 320])
  if len(size) != 2:
    return 192, 320
  return int(size[0]), int(size[1])


def _choose_device(args, cfg: Dict) -> torch.device:
  requested = str(getattr(args, "device", "cpu")).lower()
  if requested == "cpu":
    return torch.device("cpu")
  if requested == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device("cuda")

  # auto mode keeps the older behavior: follow config when CUDA is available.
  device_want = str(cfg.get("device", "cpu")).lower()
  if device_want == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def _prep_image(path: Path, image_size_hw: Tuple[int, int]) -> torch.Tensor:
  h, w = image_size_hw
  img = Image.open(path).convert("RGB").resize((w, h), RESAMPLE_BILINEAR)
  arr = np.asarray(img).astype(np.float32) / 255.0
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  arr = (arr - mean) / std
  arr = np.transpose(arr, (2, 0, 1))
  return torch.from_numpy(arr).unsqueeze(0).float()


def _collect_images(input_dir: Optional[str]) -> List[Path]:
  if not input_dir:
    return []
  folder = _resolve(input_dir)
  if not folder.exists():
    return []
  return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _find_mask(mask_dir: Path, image_path: Path) -> Optional[Path]:
  mask, method, score, note = find_best_mask_for_image(mask_dir, image_path, allow_fuzzy=True, fuzzy_threshold=0.82)
  return mask


def _read_gt_mask(mask_path: Path, image_size_hw: Tuple[int, int]) -> np.ndarray:
  h, w = image_size_hw
  mask = Image.open(mask_path).convert("L").resize((w, h), RESAMPLE_NEAREST)
  arr = np.asarray(mask)
  return (arr > 0).astype(np.uint8)


def _boundary_map(mask: np.ndarray, width: int = 2) -> np.ndarray:
  """Return a thin binary boundary map without requiring OpenCV or SciPy."""
  mask = (mask > 0).astype(np.uint8)
  if mask.sum() == 0:
    return np.zeros_like(mask, dtype=np.uint8)
  radius = max(1, int(width))
  padded = np.pad(mask, radius, mode="constant", constant_values=0)
  eroded = np.ones_like(mask, dtype=bool)
  for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
      view = padded[radius + dy:radius + dy + mask.shape[0], radius + dx:radius + dx + mask.shape[1]]
      eroded &= (view == 1)
  return ((mask == 1) & (~eroded)).astype(np.uint8)


def _update_binary_metrics(pred: np.ndarray, gt: np.ndarray, totals: Dict[str, int]):
  pred = (pred > 0).astype(np.uint8)
  gt = (gt > 0).astype(np.uint8)
  totals["tp"] += int(((pred == 1) & (gt == 1)).sum())
  totals["tn"] += int(((pred == 0) & (gt == 0)).sum())
  totals["fp"] += int(((pred == 1) & (gt == 0)).sum())
  totals["fn"] += int(((pred == 0) & (gt == 1)).sum())

  pred_boundary = _boundary_map(pred)
  gt_boundary = _boundary_map(gt)
  totals["boundary_tp"] += int(((pred_boundary == 1) & (gt_boundary == 1)).sum())
  totals["boundary_fp"] += int(((pred_boundary == 1) & (gt_boundary == 0)).sum())
  totals["boundary_fn"] += int(((pred_boundary == 0) & (gt_boundary == 1)).sum())


def _finish_metrics(totals: Dict[str, int]) -> Dict[str, Optional[float]]:
  tp, tn, fp, fn = totals["tp"], totals["tn"], totals["fp"], totals["fn"]
  total = tp + tn + fp + fn
  if total == 0:
    return {"pixel_acc": None, "miou_binary": None, "iou_damage": None, "iou_background": None, "boundary_iou": None}
  iou_damage_den = tp + fp + fn
  iou_bg_den = tn + fp + fn
  iou_damage = tp / iou_damage_den if iou_damage_den else None
  iou_bg = tn / iou_bg_den if iou_bg_den else None
  boundary_den = totals.get("boundary_tp", 0) + totals.get("boundary_fp", 0) + totals.get("boundary_fn", 0)
  boundary_iou = totals.get("boundary_tp", 0) / boundary_den if boundary_den else None
  valid_ious = [x for x in [iou_bg, iou_damage] if x is not None]
  return {
    "pixel_acc": (tp + tn) / total,
    "miou_binary": sum(valid_ious) / len(valid_ious) if valid_ious else None,
    "iou_damage": iou_damage,
    "iou_background": iou_bg,
    "boundary_iou": boundary_iou,
  }


def _format_float(v):
  if v is None:
    return "NA"
  if isinstance(v, float):
    return f"{v:.6f}"
  return v


def _maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str], device: torch.device) -> str:
  if not ckpt_path:
    return "not_provided"
  ckpt = _resolve(ckpt_path)
  if not ckpt.exists():
    return "missing"
  load_state(str(ckpt), model, map_location=device.type)
  return "loaded"


@torch.no_grad()
def _measure_latency(
  model: torch.nn.Module,
  device: torch.device,
  image_size_hw: Tuple[int, int],
  repeats: int,
  warmup: int,
  batch_size: int = 1,
  amp: bool = False,
) -> Dict[str, Optional[float]]:
  if repeats <= 0:
    return {
      "latency_ms": None,
      "latency_std_ms": None,
      "latency_min_ms": None,
      "latency_max_ms": None,
      "throughput_fps": None,
      "cuda_peak_memory_mb": None,
      "cuda_allocated_memory_mb": None,
    }

  h, w = image_size_hw
  batch_size = max(1, int(batch_size))
  x = torch.randn(batch_size, 3, h, w, device=device)
  model.eval()

  use_amp = bool(amp and device.type == "cuda")
  autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

  for _ in range(max(0, warmup)):
    with autocast_ctx:
      _ = model(x)
  if device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

  samples = []
  for _ in range(repeats):
    if device.type == "cuda":
      torch.cuda.synchronize()
    start = time.perf_counter()
    with autocast_ctx:
      _ = model(x)
    if device.type == "cuda":
      torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1000.0)

  arr = np.asarray(samples, dtype=np.float64)
  mean_ms = float(arr.mean())
  result = {
    "latency_ms": mean_ms,
    "latency_std_ms": float(arr.std(ddof=0)),
    "latency_min_ms": float(arr.min()),
    "latency_max_ms": float(arr.max()),
    "throughput_fps": float((1000.0 * batch_size) / mean_ms) if mean_ms > 0 else None,
    "cuda_peak_memory_mb": None,
    "cuda_allocated_memory_mb": None,
  }
  if device.type == "cuda":
    result["cuda_peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    result["cuda_allocated_memory_mb"] = float(torch.cuda.memory_allocated(device) / (1024 * 1024))
  return result


@torch.no_grad()
def _evaluate_if_possible(
  model: torch.nn.Module,
  device: torch.device,
  image_paths: List[Path],
  mask_dir: Optional[str],
  image_size_hw: Tuple[int, int],
) -> Dict[str, Optional[float]]:
  if not image_paths or not mask_dir:
    return {"pixel_acc": None, "miou_binary": None, "iou_damage": None, "iou_background": None, "boundary_iou": None, "eval_images": 0}
  gt_folder = _resolve(mask_dir)
  if not gt_folder.exists():
    return {"pixel_acc": None, "miou_binary": None, "iou_damage": None, "iou_background": None, "boundary_iou": None, "eval_images": 0}

  totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
  used = 0
  model.eval()
  for img_path in image_paths:
    gt_path = _find_mask(gt_folder, img_path)
    if gt_path is None:
      continue
    x = _prep_image(img_path, image_size_hw).to(device)
    out = model(x)["out"]
    pred = torch.argmax(out, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    gt = _read_gt_mask(gt_path, image_size_hw)
    _update_binary_metrics(pred, gt, totals)
    used += 1
  metrics = _finish_metrics(totals)
  metrics["eval_images"] = used
  return metrics


def compare_one(name: str, config_path: str, ckpt_path: Optional[str], args) -> Dict:
  row = {
    "name": name,
    "config": config_path,
    "checkpoint": ckpt_path or "",
    "status": "OK",
    "error": "",
    "checkpoint_status": "not_provided",
    "device": "NA",
    "device_name": "NA",
    "cpu_threads": "NA",
    "image_size_hw": "NA",
    "params": "NA",
    "param_million": "NA",
    "param_size_mb_fp32": "NA",
    "latency_repeats": "NA",
    "latency_warmup": "NA",
    "batch_size": "NA",
    "amp": "NA",
    "latency_ms": "NA",
    "latency_std_ms": "NA",
    "latency_min_ms": "NA",
    "latency_max_ms": "NA",
    "throughput_fps": "NA",
    "cuda_peak_memory_mb": "NA",
    "cuda_allocated_memory_mb": "NA",
    "pixel_acc": "NA",
    "miou_binary": "NA",
    "iou_damage": "NA",
    "iou_background": "NA",
    "boundary_iou": "NA",
    "eval_images": 0,
    "note": "",
  }
  try:
    cfg = load_yaml(str(_resolve(config_path)))
    device = _choose_device(args, cfg)
    image_size_hw = _image_size_from_cfg(cfg)
    model = get_model(cfg).to(device)
    params = _param_count(model)
    row["device"] = device.type
    row["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    row["cpu_threads"] = torch.get_num_threads() if device.type == "cpu" else "NA"
    row["image_size_hw"] = f"{image_size_hw[0]}x{image_size_hw[1]}"
    row["params"] = params
    row["param_million"] = params / 1_000_000
    row["param_size_mb_fp32"] = params * 4 / (1024 * 1024)
    row["checkpoint_status"] = _maybe_load_checkpoint(model, ckpt_path, device)
    row["latency_repeats"] = args.latency_repeats
    row["latency_warmup"] = args.latency_warmup
    row["batch_size"] = args.batch_size
    row["amp"] = bool(args.amp and device.type == "cuda")
    row.update(_measure_latency(model, device, image_size_hw, args.latency_repeats, args.latency_warmup, args.batch_size, args.amp))

    image_paths = _collect_images(args.input_dir)
    if row["checkpoint_status"] == "loaded" and args.mask_dir:
      metrics = _evaluate_if_possible(model, device, image_paths, args.mask_dir, image_size_hw)
      row.update(metrics)
    elif args.mask_dir and row["checkpoint_status"] != "loaded":
      row["note"] = "mIoU/PixelAcc skipped because checkpoint is not loaded."
    else:
      row["note"] = "Architecture/cost comparison only. Provide --mask_dir and checkpoints for accuracy metrics."
  except Exception as exc:
    row["status"] = "ERROR"
    row["error"] = f"{type(exc).__name__}: {exc}"
  return row


def main():
  p = argparse.ArgumentParser(description="Compare capstone segmentation model candidates")
  p.add_argument("--configs", nargs="+", required=True, help="YAML configs to compare")
  p.add_argument("--names", nargs="+", default=None, help="Display names, same count as configs")
  p.add_argument("--ckpts", nargs="*", default=None, help="Optional checkpoints, same count as configs")
  p.add_argument("--input_dir", default=None, help="Optional image folder for accuracy evaluation")
  p.add_argument("--mask_dir", default=None, help="Optional GT mask folder for mIoU/PixelAcc")
  p.add_argument("--outdir", default="seg/runs/model_compare")
  p.add_argument(
    "--device",
    default="cpu",
    choices=["cpu", "cuda", "auto"],
    help="Comparison device. Use cpu for deployment evidence and cuda for AWS/GPU acceleration evidence.",
  )
  p.add_argument("--cpu_threads", type=int, default=0, help="Optional torch CPU thread count. 0 keeps PyTorch default.")
  p.add_argument("--batch_size", type=int, default=1, help="Batch size used only for latency/throughput profiling. Keep 1 for single-image service latency.")
  p.add_argument("--amp", action="store_true", help="Use CUDA autocast fp16 during latency profiling only. Accuracy metrics still run in normal precision.")
  p.add_argument("--latency_repeats", type=int, default=50, help="0 disables dummy latency test")
  p.add_argument("--latency_warmup", type=int, default=10, help="Warmup forward passes before latency timing")
  args = p.parse_args()

  if args.cpu_threads > 0:
    torch.set_num_threads(args.cpu_threads)

  names = args.names or [Path(c).stem for c in args.configs]
  if len(names) != len(args.configs):
    raise ValueError("--names count must match --configs count")
  ckpts = args.ckpts or [None] * len(args.configs)
  if len(ckpts) < len(args.configs):
    ckpts = ckpts + [None] * (len(args.configs) - len(ckpts))
  if len(ckpts) != len(args.configs):
    raise ValueError("--ckpts count must match --configs count")

  rows = [compare_one(n, c, k, args) for n, c, k in zip(names, args.configs, ckpts)]

  outdir = _resolve(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  csv_path = outdir / "model_compare_summary.csv"
  json_path = outdir / "model_compare_summary.json"

  fieldnames = list(rows[0].keys()) if rows else []
  with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: _format_float(v) for k, v in row.items()})
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

  def _with_unit(value, unit: str) -> str:
    if value in (None, "NA"):
      return "NA"
    if isinstance(value, str) and value.upper() == "NA":
      return "NA"
    return f"{_format_float(value)}{unit}"

  print("\n[MODEL COMPARISON]")
  for row in rows:
    print(
      f"- {row['name']}: status={row['status']} | "
      f"ckpt={row['checkpoint_status']} | "
      f"device={row['device']} | "
      f"params={_with_unit(row['param_million'], 'M')} | "
      f"latency={_with_unit(row['latency_ms'], 'ms')}±{_with_unit(row['latency_std_ms'], 'ms')} | "
      f"fps={_format_float(row['throughput_fps'])} | "
      f"gpu_mem={_with_unit(row.get('cuda_peak_memory_mb', 'NA'), 'MB')} | "
      f"mIoU={_format_float(row['miou_binary'])} | "
      f"BoundaryIoU={_format_float(row.get('boundary_iou', 'NA'))}"
    )
    if row["error"]:
      print(f" error: {row['error']}")
  print("\nSaved:")
  print("-", csv_path)
  print("-", json_path)


if __name__ == "__main__":
  main()
