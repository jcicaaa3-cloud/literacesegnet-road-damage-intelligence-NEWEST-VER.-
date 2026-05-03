"""Train LiteRaceSegNet only.

This is intentionally separated from SegFormer training.
- This script trains the user's custom lightweight CNN-based model: LiteRaceSegNet.
- It does NOT create or update SegFormer checkpoints.
- It saves best.pth under seg/runs/literace_boundary_degradation for the service/LLM path.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
  import torch
  import torch.nn.functional as F
  from torch import nn
  from torch.utils.data import DataLoader, Dataset
except Exception as exc: # pragma: no cover
  raise RuntimeError("torch is required for LiteRaceSegNet training") from exc

SEG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SEG_DIR.parent
if str(SEG_DIR) not in sys.path:
  sys.path.insert(0, str(SEG_DIR))

from core.data_pairs import collect_image_mask_pairs, write_pairing_report
from core.model_select import get_model
from core.save import save_state
from core.train_utils import get_device, load_yaml, make_dir, set_seed

try:
  RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
  RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError: # Pillow<9
  RESAMPLE_BILINEAR = Image.BILINEAR
  RESAMPLE_NEAREST = Image.NEAREST


def resolve_project_path(path_text: str | os.PathLike) -> Path:
  p = Path(path_text)
  return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def image_size_from_cfg(cfg: Dict) -> Tuple[int, int]:
  size = cfg.get("train", {}).get("image_size", [192, 320])
  if len(size) != 2:
    raise ValueError("train.image_size must be [height, width]")
  return int(size[0]), int(size[1])


def find_split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
  candidates = [
    (root / split / "images", root / split / "masks"),
    (root / "images" / split, root / "masks" / split),
    (root / split / "image", root / split / "mask"),
    (root / split / "imgs", root / split / "labels"),
  ]
  for img_dir, mask_dir in candidates:
    if img_dir.exists() and mask_dir.exists():
      return img_dir, mask_dir
  return candidates[0]


class BinaryPotholeDataset(Dataset):
  def __init__(self, cfg: Dict, split: str, augment: bool = False):
    self.cfg = cfg
    self.split = split
    self.augment = augment
    data_cfg = cfg.get("data", {})
    self.root = resolve_project_path(data_cfg.get("root", "datasets/pothole_binary/processed"))
    self.image_size_hw = image_size_from_cfg(cfg)
    self.positive_threshold = int(data_cfg.get("mask_positive_threshold", 0))
    self.ignore_values = set(int(v) for v in data_cfg.get("mask_ignore_values", []))
    self.ignore_index = int(data_cfg.get("ignore_index", 255))
    self.allow_fuzzy = bool(data_cfg.get("allow_fuzzy_filename_match", True))
    self.fuzzy_threshold = float(data_cfg.get("fuzzy_match_threshold", 0.82))

    self.image_dir, self.mask_dir = find_split_dirs(self.root, split)
    if not self.image_dir.exists() or not self.mask_dir.exists():
      raise FileNotFoundError(
        f"Dataset split not found for '{split}'.\n"
        f"Expected, for example:\n"
        f" {self.root / split / 'images'}\n"
        f" {self.root / split / 'masks'}\n"
      )

    self.samples, report_rows = collect_image_mask_pairs(
      self.image_dir, self.mask_dir, allow_fuzzy=self.allow_fuzzy, fuzzy_threshold=self.fuzzy_threshold
    )
    report_path = PROJECT_ROOT / "seg" / "runs" / "dataset_pairing_reports" / f"literace_{split}_pairing_report.csv"
    write_pairing_report(report_path, report_rows)
    unmatched = [r for r in report_rows if not r.get("matched_mask")]
    fuzzy = [r for r in report_rows if r.get("method") == "fuzzy"]
    print(f"[PAIR:{split}] images={len(report_rows)} paired={len(self.samples)} fuzzy={len(fuzzy)} unmatched={len(unmatched)}")
    print(f"[PAIR:{split}] report={report_path}")

    if not self.samples:
      raise FileNotFoundError(
        f"No paired image/mask files found for split '{split}'.\n"
        f"Image dir: {self.image_dir}\n"
        f"Mask dir: {self.mask_dir}\n"
        f"Check pairing report: {report_path}\n"
      )
    if unmatched:
      raise FileNotFoundError(
        f"Some images have no matched mask for split '{split}'.\n"
        f"Fix filenames or check: {report_path}"
      )

  def __len__(self) -> int:
    return len(self.samples)

  def _load_image(self, path: Path) -> np.ndarray:
    h, w = self.image_size_hw
    img = Image.open(path).convert("RGB").resize((w, h), RESAMPLE_BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))

  def _load_mask(self, path: Path) -> np.ndarray:
    h, w = self.image_size_hw
    mask = Image.open(path).resize((w, h), RESAMPLE_NEAREST)
    arr = np.asarray(mask)
    if arr.ndim == 3:
      base = np.any(arr > self.positive_threshold, axis=2)
      ignore = np.zeros(base.shape, dtype=bool)
      for value in self.ignore_values:
        ignore |= np.all(arr == value, axis=2)
    else:
      ignore = np.zeros(arr.shape, dtype=bool)
      for value in self.ignore_values:
        ignore |= arr == value
      base = arr > self.positive_threshold
    out = base.astype(np.int64)
    out[ignore] = self.ignore_index
    return out

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    img_path, mask_path = self.samples[idx]
    image = self._load_image(img_path)
    mask = self._load_mask(mask_path)
    if self.augment and random.random() < 0.5:
      image = image[:, :, ::-1].copy()
      mask = mask[:, ::-1].copy()
    return {"pixel_values": torch.from_numpy(image).float(), "labels": torch.from_numpy(mask).long()}


def make_loader(cfg: Dict, split: str, train: bool) -> DataLoader:
  ds = BinaryPotholeDataset(cfg, split=split, augment=train)
  batch_size = int(cfg.get("train", {}).get("batch_size", 4) if train else cfg.get("val", {}).get("batch_size", 2))
  num_workers = int(cfg.get("num_workers", 0))
  return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
  valid = labels != ignore_index
  if valid.sum() == 0:
    return logits.sum() * 0.0
  probs = torch.softmax(logits, dim=1)[:, 1]
  target = (labels == 1).float()
  probs = probs[valid]
  target = target[valid]
  inter = (probs * target).sum()
  union = probs.sum() + target.sum()
  return 1.0 - (2.0 * inter + eps) / (union + eps)


def boundary_target_from_labels(labels: torch.Tensor, width: int, ignore_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
  # labels: [B,H,W]
  valid = (labels != ignore_index).float().unsqueeze(1)
  target = (labels == 1).float().unsqueeze(1)
  k = max(3, int(width) | 1) # odd >=3
  pad = k // 2
  dil = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)
  ero = 1.0 - F.max_pool2d(1.0 - target, kernel_size=k, stride=1, padding=pad)
  edge = (dil - ero).clamp(0, 1)
  return edge, valid


def compute_literace_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, cfg: Dict, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
  train_cfg = cfg.get("train", {})
  loss_cfg = train_cfg.get("loss", {})
  ignore_index = int(cfg.get("data", {}).get("ignore_index", 255))
  ce_weight = float(loss_cfg.get("ce_weight", 1.0))
  dice_weight = float(loss_cfg.get("dice_weight", 0.6))
  aux_weight = float(loss_cfg.get("aux_weight", 0.2))
  boundary_weight = float(loss_cfg.get("boundary_weight", 0.05))

  out = F.interpolate(outputs["out"], size=labels.shape[-2:], mode="bilinear", align_corners=False)
  loss = out.sum() * 0.0
  if ce_weight > 0:
    loss = loss + ce_weight * F.cross_entropy(out, labels, weight=class_weights, ignore_index=ignore_index)
  if dice_weight > 0:
    loss = loss + dice_weight * dice_loss_from_logits(out, labels, ignore_index=ignore_index)

  aux = outputs.get("aux")
  if aux is not None and aux_weight > 0:
    aux = F.interpolate(aux, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    aux_loss = F.cross_entropy(aux, labels, weight=class_weights, ignore_index=ignore_index)
    aux_loss = aux_loss + dice_loss_from_logits(aux, labels, ignore_index=ignore_index)
    loss = loss + aux_weight * aux_loss

  boundary = outputs.get("boundary")
  if boundary is not None and boundary_weight > 0:
    edge, valid = boundary_target_from_labels(labels, int(train_cfg.get("boundary_width", 3)), ignore_index)
    boundary = F.interpolate(boundary, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(boundary, edge, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    loss = loss + boundary_weight * bce
  return loss


def update_metrics(logits: torch.Tensor, labels: torch.Tensor, totals: Dict[str, int], ignore_index: int = 255) -> None:
  logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
  pred = torch.argmax(logits, dim=1)
  valid = labels != ignore_index
  pred = pred[valid]
  gt = labels[valid]
  if pred.numel() == 0:
    return
  totals["tp"] += int(((pred == 1) & (gt == 1)).sum().item())
  totals["tn"] += int(((pred == 0) & (gt == 0)).sum().item())
  totals["fp"] += int(((pred == 1) & (gt == 0)).sum().item())
  totals["fn"] += int(((pred == 0) & (gt == 1)).sum().item())


def finish_metrics(totals: Dict[str, int]) -> Dict[str, float]:
  tp, tn, fp, fn = totals["tp"], totals["tn"], totals["fp"], totals["fn"]
  total = tp + tn + fp + fn
  pixel_acc = (tp + tn) / total if total else 0.0
  iou_damage = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
  iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
  return {"pixel_acc": pixel_acc, "miou_binary": (iou_damage + iou_bg) / 2.0, "iou_damage": iou_damage, "iou_background": iou_bg}


def set_poly_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, base_lr: float, min_lr: float, power: float) -> float:
  lr = min_lr + (base_lr - min_lr) * ((1.0 - min(step, total_steps) / max(1, total_steps)) ** power)
  for group in optimizer.param_groups:
    group["lr"] = lr
  return lr


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler, device, cfg, class_weights, epoch: int, total_epochs: int, global_step: int, total_steps: int):
  model.train()
  train_cfg = cfg.get("train", {})
  base_lr = float(train_cfg.get("base_lr", 0.001))
  min_lr = float(cfg.get("scheduler", {}).get("min_lr", 1e-6))
  power = float(cfg.get("scheduler", {}).get("power", 0.9))
  grad_clip = float(train_cfg.get("grad_clip", 1.0))
  print_freq = int(train_cfg.get("print_freq", 1))
  amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"

  running = 0.0
  lr = base_lr
  for i, batch in enumerate(loader, start=1):
    global_step += 1
    lr = set_poly_lr(optimizer, global_step, total_steps, base_lr, min_lr, power)
    x = batch["pixel_values"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp_enabled):
      outputs = model(x)
      loss = compute_literace_loss(outputs, y, cfg, class_weights)
    if scaler is not None and amp_enabled:
      scaler.scale(loss).backward()
      if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()
    running += float(loss.detach().item())
    if print_freq > 0 and (i == 1 or i % print_freq == 0 or i == len(loader)):
      print(f"[TRAIN-LiteRace] epoch {epoch:03d}/{total_epochs:03d} step {i:04d}/{len(loader):04d} loss={loss.item():.5f} lr={lr:.8f}")
  return running / max(1, len(loader)), global_step, lr


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device, cfg, class_weights) -> Dict[str, float]:
  model.eval()
  losses: List[float] = []
  totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
  ignore_index = int(cfg.get("data", {}).get("ignore_index", 255))
  for batch in loader:
    x = batch["pixel_values"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)
    outputs = model(x)
    loss = compute_literace_loss(outputs, y, cfg, class_weights)
    losses.append(float(loss.item()))
    update_metrics(outputs["out"], y, totals, ignore_index=ignore_index)
  metrics = finish_metrics(totals)
  metrics["loss"] = sum(losses) / max(1, len(losses))
  return metrics


def main() -> int:
  parser = argparse.ArgumentParser(description="Train LiteRaceSegNet only for binary pothole segmentation")
  parser.add_argument("--config", default="seg/config/pothole_binary_literace_train.yaml")
  parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
  parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override cfg.device")
  args = parser.parse_args()

  cfg_path = resolve_project_path(args.config)
  cfg = load_yaml(str(cfg_path))
  cfg.setdefault("model", {})["name"] = "lite_race" # hard guard: this trainer is LiteRace only.
  if args.epochs is not None:
    cfg.setdefault("train", {})["epochs"] = int(args.epochs)
  if args.device is not None:
    cfg["device"] = args.device

  set_seed(int(cfg.get("seed", 42)))
  device = get_device(cfg)
  print(f"[INFO] Training target: LiteRaceSegNet ONLY")
  print(f"[INFO] Project root: {PROJECT_ROOT}")
  print(f"[INFO] Config: {cfg_path}")
  print(f"[INFO] Device: {device}")

  train_loader = make_loader(cfg, "train", train=True)
  val_loader = make_loader(cfg, "val", train=False)
  print(f"[INFO] Train samples: {len(train_loader.dataset)}")
  print(f"[INFO] Val samples:  {len(val_loader.dataset)}")

  model = get_model(cfg).to(device)
  train_cfg = cfg.get("train", {})
  class_weights = None
  if train_cfg.get("class_weights") is not None:
    class_weights = torch.tensor(train_cfg.get("class_weights"), dtype=torch.float32, device=device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("base_lr", 0.001)), weight_decay=float(train_cfg.get("weight_decay", 0.0001)))
  amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
  scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

  epochs = int(train_cfg.get("epochs", 20))
  total_steps = max(1, epochs * len(train_loader))
  global_step = 0
  save_dir = resolve_project_path(cfg.get("save_dir", "seg/runs/literace_boundary_degradation"))
  make_dir(str(save_dir))
  log_path = save_dir / "train_log.csv"
  best_miou = -1.0
  no_improve = 0
  patience = int(train_cfg.get("early_stopping_patience", 6))

  with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "val_loss", "pixel_acc", "miou_binary", "iou_damage", "iou_background", "best"])
    writer.writeheader()
    for epoch in range(1, epochs + 1):
      train_loss, global_step, lr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, class_weights, epoch, epochs, global_step, total_steps)
      metrics = validate(model, val_loader, device, cfg, class_weights)
      miou = float(metrics["miou_binary"])
      improved = miou > best_miou
      if improved:
        best_miou = miou
        no_improve = 0
      else:
        no_improve += 1
      row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": metrics["loss"], "pixel_acc": metrics["pixel_acc"], "miou_binary": metrics["miou_binary"], "iou_damage": metrics["iou_damage"], "iou_background": metrics["iou_background"], "best": int(improved)}
      writer.writerow(row)
      f.flush()
      print(f"[VAL-LiteRace] epoch {epoch:03d}/{epochs:03d} loss={metrics['loss']:.5f} pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['miou_binary']:.4f} damageIoU={metrics['iou_damage']:.4f} best={best_miou:.4f}")
      save_state(str(save_dir / "last.pth"), model, optimizer, None, scaler, epoch, best_miou, cfg)
      if improved:
        save_state(str(save_dir / "best.pth"), model, optimizer, None, scaler, epoch, best_miou, cfg)
        print(f"[SAVE-LiteRace] best checkpoint: {save_dir / 'best.pth'}")
      if patience > 0 and no_improve >= patience:
        print(f"[EARLY STOP] No mIoU improvement for {patience} epoch(s).")
        break

  print("\n[DONE] LiteRaceSegNet training finished.")
  print(f"- best mIoU: {best_miou:.4f}")
  print(f"- checkpoint for service/LLM: {save_dir / 'best.pth'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
