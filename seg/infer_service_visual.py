import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
  RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
  RESAMPLE_NEAREST = Image.NEAREST

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _read_image(path: Path) -> Image.Image:
  if not path.exists():
    raise FileNotFoundError(f"Cannot read image: {path}")
  return Image.open(path).convert("RGB")


def _read_mask(path: Path, size_wh) -> np.ndarray:
  if not path.exists():
    raise FileNotFoundError(f"Cannot read mask: {path}")
  mask_img = Image.open(path).convert("L")
  if mask_img.size != size_wh:
    mask_img = mask_img.resize(size_wh, RESAMPLE_NEAREST)
  return (np.asarray(mask_img) > 0).astype(np.uint8)


def _safe_percentile(values: np.ndarray, q: float, default: float) -> float:
  values = np.asarray(values)
  if values.size == 0:
    return default
  return float(np.percentile(values, q))


def _concept_road_roi(w: int, h: int) -> np.ndarray:
  """Trapezoid ROI for the bundled city-driving concept image only."""
  roi_img = Image.new("L", (w, h), 0)
  roi_draw = ImageDraw.Draw(roi_img)
  roi_draw.polygon(
    [
      (int(0.23 * w), int(0.39 * h)),
      (int(0.98 * w), int(0.34 * h)),
      (w, h),
      (0, h),
      (int(0.16 * w), int(0.61 * h)),
    ],
    fill=255,
  )
  return np.asarray(roi_img) > 0


def _ordinary_road_texture_mask(img: Image.Image) -> np.ndarray:
  """Deterministic pseudo-mask for ordinary road photos.

  This is NOT a trained model. It is a presentation/demo fallback that tries
  to draw only plausible dark cracks / rough damaged asphalt evidence. It is
  intentionally conservative so ordinary input photos do not become a full
  cyan carpet.
  """
  w, h = img.size
  rgb = np.asarray(img).astype(np.int16)
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

  gray_img = img.convert("L")
  gray_raw = np.asarray(gray_img).astype(np.float32)
  gray = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(radius=1.0))).astype(np.float32)
  local = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(radius=18.0))).astype(np.float32)

  gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
  gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
  grad = gx + gy
  detail = np.abs(gray_raw - np.asarray(gray_img.filter(ImageFilter.GaussianBlur(radius=5.0))).astype(np.float32))
  texture = grad * 0.80 + detail * 1.15
  dark_contrast = np.maximum(local - gray, 0.0)

  maxc = np.maximum.reduce([r, g, b])
  minc = np.minimum.reduce([r, g, b])
  sat = maxc - minc

  # Exclude yellow/white road markings and artificial cyan overlays from
  # previous outputs. They are not road damage evidence.
  yellow_line = (r > 125) & (g > 95) & (b < 120) & ((r - b) > 35) & ((g - b) > 20)
  white_paint = (gray_raw > 185) & (sat < 45)
  cyan_overlay = (g > 120) & (b > 120) & (g > r + 22) & (b > r + 18)
  valid = ~(yellow_line | white_paint | cyan_overlay)

  valid_values = gray[valid]
  texture_values = texture[valid]
  contrast_values = dark_contrast[valid]
  if valid_values.size == 0:
    return np.zeros((h, w), dtype=np.uint8)

  dark_thr = _safe_percentile(valid_values, 34, 85)
  texture_thr = _safe_percentile(texture_values, 82, 22)
  crack_texture_thr = _safe_percentile(texture_values, 91, 35)
  contrast_thr = max(_safe_percentile(contrast_values, 78, 8), 5.0)

  # Score favors pixels that are darker than local neighborhood and have
  # rough/crack-like texture.
  global_median = _safe_percentile(valid_values, 50, 120)
  global_dark = np.maximum(global_median - gray, 0.0)
  score = dark_contrast * 1.80 + texture * 0.75 + global_dark * 0.45
  score[~valid] = 0

  candidate = (
    ((gray <= dark_thr) & (texture >= texture_thr)) |
    ((dark_contrast >= contrast_thr) & (texture >= texture_thr)) |
    ((texture >= crack_texture_thr) & (gray <= _safe_percentile(valid_values, 55, 130)))
  ) & valid

  # Keep only the strongest candidate pixels. This prevents the whole asphalt
  # texture from being painted cyan.
  candidate_scores = score[candidate]
  if candidate_scores.size > 0:
    strong_thr = _safe_percentile(candidate_scores, 58, float(candidate_scores.mean()))
    candidate = candidate & (score >= strong_thr)

  # If still too much is selected, cap by score percentile. For demo output,
  # a selective region is better than a meaningless full-frame overlay.
  coverage = float(candidate.mean())
  if coverage > 0.18:
    cap_thr = _safe_percentile(score[valid], 90, 9999)
    candidate = (score >= cap_thr) & valid
  if float(candidate.mean()) > 0.12:
    cap_thr = _safe_percentile(score[valid], 94, 9999)
    candidate = (score >= cap_thr) & valid

  cand_img = Image.fromarray((candidate * 255).astype(np.uint8))
  cand_img = cand_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MedianFilter(3))
  mask = np.asarray(cand_img) > 0

  # If extremely sparse, add only very strong thin crack evidence.
  if int(mask.sum()) < max(80, int(w * h * 0.00015)):
    edge_fallback = (texture >= _safe_percentile(texture_values, 96, 48)) & valid & (gray <= _safe_percentile(valid_values, 62, 140))
    mask = np.asarray(Image.fromarray((edge_fallback * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 0

  # Final hard cap: never allow mock/demo output to cover most of the image.
  # Keep the strongest ~10% when the heuristic is too broad.
  if float(mask.mean()) > 0.16:
    final_thr = _safe_percentile(score[valid], 92, 9999)
    mask = (score >= final_thr) & valid
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 0

  return mask.astype(np.uint8)


def _demo_mask_from_image(img: Image.Image) -> np.ndarray:
  """Create a presentation-only pseudo mask without model inference."""
  ow, oh = img.size
  max_side = 720
  scale = min(max_side / max(ow, oh), 1.0)
  if scale < 1.0:
    sw, sh = max(1, int(ow * scale)), max(1, int(oh * scale))
    small = img.resize((sw, sh))
    small_mask = _demo_mask_from_image_no_resize(small)
    return (np.asarray(Image.fromarray((small_mask * 255).astype(np.uint8)).resize((ow, oh), RESAMPLE_NEAREST)) > 0).astype(np.uint8)
  return _demo_mask_from_image_no_resize(img)


def _demo_mask_from_image_no_resize(img: Image.Image) -> np.ndarray:
  """Create a presentation-only pseudo mask without model inference.

  This mode exists only to demonstrate how a real segmentation mask would be
  visualized in a road-management service. Do not use it as performance proof.

  If the bundled concept image is used, this extracts the already drawn cyan
  crack / purple damage evidence and expands it into irregular regions. If a
  normal unannotated road image is used, it falls back to image-texture based
  pseudo detection. It never draws fixed circles, ellipses, or hard-coded
  polygon carpets.
  """
  w, h = img.size
  arr = np.asarray(img).astype(np.int16)
  r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

  concept_roi = _concept_road_roi(w, h)
  cyan_seed = (g > 120) & (b > 120) & (g > r + 25) & (b > r + 20)
  purple_seed = (r > 85) & (b > 105) & (b > g + 25) & (r > g + 15)
  colored_seed = (cyan_seed | purple_seed) & concept_roi

  if int(colored_seed.sum()) > max(200, int(w * h * 0.001)):
    seed_img = Image.fromarray((colored_seed * 255).astype(np.uint8))
    broad = (
      seed_img
      .filter(ImageFilter.MaxFilter(17))
      .filter(ImageFilter.MinFilter(7))
      .filter(ImageFilter.GaussianBlur(radius=1.6))
      .point(lambda p: 255 if p > 70 else 0)
    )
    fine = seed_img.filter(ImageFilter.MaxFilter(7))
    mask = (np.asarray(broad) > 0) | (np.asarray(fine) > 0)

    # Suppress obvious vehicle body in the concept image. This branch is not
    # used for ordinary raw road photos.
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    vehicle_body = (x_idx < int(0.38 * w)) & (y_idx < int(0.55 * h))
    mask[vehicle_body] = False
    mask &= concept_roi
    mask[: int(h * 0.35), :] = False
    return mask.astype(np.uint8)

  return _ordinary_road_texture_mask(img)


def _severity_from_ratio(ratio: float) -> str:
  if ratio >= 0.08:
    return "high"
  if ratio >= 0.025:
    return "medium"
  if ratio > 0:
    return "low"
  return "none"


def _recommendation(severity: str) -> str:
  return {
    "high": "긴급 점검 및 보수 우선순위 상위 등록",
    "medium": "정기 점검 목록 등록 및 확대 여부 추적",
    "low": "관찰 대상 등록, 다음 주행/촬영 시 재확인",
    "none": "감지된 손상 없음",
  }[severity]


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
  padded = np.pad(mask, 1, mode="constant")
  center = padded[1:-1, 1:-1]
  neighbors = [
    padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:],
    padded[:-2, :-2], padded[:-2, 2:], padded[2:, :-2], padded[2:, 2:],
  ]
  neighbor_sum = np.zeros_like(center, dtype=np.uint8)
  for n in neighbors:
    neighbor_sum += n.astype(np.uint8)
  return ((center > 0) & (neighbor_sum < 8)).astype(np.uint8)


def _overlay(img: Image.Image, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
  base = np.asarray(img).astype(np.float32)
  color = np.zeros_like(base)
  color[:, :, 0] = 20
  color[:, :, 1] = 230
  color[:, :, 2] = 255
  out = base.copy()
  active = mask > 0
  out[active] = base[active] * (1 - alpha) + color[active] * alpha
  boundary = _binary_boundary(mask)
  out[boundary > 0] = np.array([255, 255, 255], dtype=np.float32)
  return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _boundary_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
  out = np.asarray(img).copy()
  boundary = _binary_boundary(mask)
  out[boundary > 0] = np.array([255, 255, 255], dtype=np.uint8)
  return Image.fromarray(out)


def _component_stats(mask: np.ndarray, min_area: int = 8):
  h, w = mask.shape
  max_side = 100
  scale = min(max_side / max(h, w), 1.0)
  if scale < 1.0:
    small_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
      (max(1, int(w * scale)), max(1, int(h * scale))),
      RESAMPLE_NEAREST,
    )
    small = (np.asarray(small_img) > 0).astype(np.uint8)
  else:
    small = mask.astype(np.uint8)

  sh, sw = small.shape
  visited = np.zeros_like(small, dtype=bool)
  comps = []
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
  ys, xs = np.where(small > 0)

  for sy, sx in zip(ys.tolist(), xs.tolist()):
    if visited[sy, sx]:
      continue
    q = deque([(sy, sx)])
    visited[sy, sx] = True
    pts = []
    while q:
      y, x = q.popleft()
      pts.append((y, x))
      for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < sh and 0 <= nx < sw and not visited[ny, nx] and small[ny, nx] > 0:
          visited[ny, nx] = True
          q.append((ny, nx))
    if len(pts) < min_area:
      continue
    pts_arr = np.asarray(pts)
    y0, x0 = pts_arr.min(axis=0)
    y1, x1 = pts_arr.max(axis=0)
    cy, cx = pts_arr.mean(axis=0)
    inv = 1.0 / max(scale, 1e-9)
    comps.append({
      "id": len(comps) + 1,
      "area_pixels_est": int(round(len(pts) * inv * inv)),
      "bbox_xywh": [
        int(round(x0 * inv)),
        int(round(y0 * inv)),
        int(round((x1 - x0 + 1) * inv)),
        int(round((y1 - y0 + 1) * inv)),
      ],
      "centroid_xy": [round(float(cx * inv), 1), round(float(cy * inv), 1)],
    })

  comps.sort(key=lambda item: item["area_pixels_est"], reverse=True)
  for idx, comp in enumerate(comps, 1):
    comp["id"] = idx
  return comps



def _remove_small_components_fullres(mask: np.ndarray, min_area_pixels: int = 0) -> np.ndarray:
  """Remove tiny isolated foreground regions from a binary mask.

  This is a post-processing step for capstone service output. It reduces
  salt-and-pepper cyan noise before area percentage calculation.
  """
  min_area_pixels = int(min_area_pixels or 0)
  if min_area_pixels <= 1:
    return mask.astype(np.uint8)
  h, w = mask.shape
  src = (mask > 0).astype(np.uint8)
  visited = np.zeros_like(src, dtype=bool)
  keep = np.zeros_like(src, dtype=np.uint8)
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
  ys, xs = np.where(src > 0)
  for sy, sx in zip(ys.tolist(), xs.tolist()):
    if visited[sy, sx]:
      continue
    q = deque([(sy, sx)])
    visited[sy, sx] = True
    pts = []
    while q:
      y, x = q.popleft()
      pts.append((y, x))
      for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and src[ny, nx] > 0:
          visited[ny, nx] = True
          q.append((ny, nx))
    if len(pts) >= min_area_pixels:
      for y, x in pts:
        keep[y, x] = 1
  return keep


def _korean_explanation(summary: dict) -> str:
  """Rule-based explanation text. This replaces optional LLM output when no LLM is connected."""
  ratio = float(summary.get("damage_ratio_percent", 0.0))
  severity = summary.get("severity", "none")
  comp_count = int(summary.get("component_count", 0))
  if severity == "high":
    level = "높은 수준"
    action = "우선 점검 및 보수 검토가 필요합니다"
  elif severity == "medium":
    level = "중간 수준"
    action = "정기 점검 목록에 등록하고 손상 확대 여부를 추적하는 것이 좋습니다"
  elif severity == "low":
    level = "낮은 수준"
    action = "관찰 대상으로 분류하고 추가 촬영 시 재확인할 수 있습니다"
  else:
    level = "감지되지 않음"
    action = "현재 이미지에서는 뚜렷한 손상 후보가 감지되지 않았습니다"
  return (
    f"손상 후보 영역은 전체 이미지 기준 약 {ratio:.2f}%로 분석되었으며, "
    f"서비스 판정 기준상 {level}입니다. 주요 연결 영역은 {comp_count}개로 추정됩니다. "
    f"{action}."
  )

def _font(size: int):
  try:
    return ImageFont.truetype("DejaVuSans.ttf", size)
  except Exception:
    return ImageFont.load_default()


def _draw_service_card(img: Image.Image, overlay: Image.Image, summary: dict, out_path: Path):
  thumb_h = 380
  thumb_w = int(img.size[0] * (thumb_h / img.size[1]))
  left = img.resize((thumb_w, thumb_h))
  right = overlay.resize((thumb_w, thumb_h))
  panel_w = 560
  canvas = Image.new("RGB", (thumb_w * 2 + panel_w, thumb_h), (245, 245, 245))
  canvas.paste(left, (0, 0))
  canvas.paste(right, (thumb_w, 0))
  draw = ImageDraw.Draw(canvas)
  x0 = thumb_w * 2 + 28
  y = 36
  lines = [
    ("Road Damage Service View", 24),
    (f"mode: {summary['mode']}", 17),
    (f"damage ratio: {summary['damage_ratio_percent']:.2f}%", 17),
    (f"severity: {summary['severity']}", 17),
    (f"components: {summary['component_count']}", 17),
    ("recommendation:", 17),
    (summary["recommendation"], 16),
  ]
  for text, size in lines:
    draw.text((x0, y), text, fill=(30, 30, 30), font=_font(size))
    y += 38 if size >= 24 else 30
  draw.rectangle((0, thumb_h - 44, thumb_w * 2, thumb_h), fill=(0, 0, 0))
  draw.text((24, thumb_h - 32), "Left: input / Right: overlay", fill=(255, 255, 255), font=_font(17))
  canvas.save(out_path)


def process_one(image_path: Path, out_dir: Path, mask_path=None, mock: bool = False, make_card: bool = True, make_boundary: bool = True, fallback_to_mock_if_bad_mask: bool = False, min_area_pixels: int = 0):
  start = time.perf_counter()
  img = _read_image(image_path)
  w, h = img.size
  if mask_path is not None:
    mask = _read_mask(mask_path, (w, h))
    mode = "prediction_mask"
    mask_ratio = float(mask.mean())
    # Demo checkpoints sometimes predict every pixel as foreground. Showing
    # a full cyan image is worse than useless, so for demo mode we can fall
    # back to the conservative presentation-only pseudo mask and record it.
    if fallback_to_mock_if_bad_mask and (mask_ratio >= 0.90 or mask_ratio <= 0.00005):
      mask = _demo_mask_from_image(img)
      mode = "invalid_prediction_mask_fallback_to_mock_demo"
  elif mock:
    mask = _demo_mask_from_image(img)
    mode = "mock_demo_not_model_result"
  else:
    raise ValueError("Provide --mask/--mask_dir or use --mock for proposal-only demo visualization.")

  raw_damage_pixels = int(mask.sum())
  if min_area_pixels and min_area_pixels > 1:
    mask = _remove_small_components_fullres(mask, min_area_pixels=min_area_pixels)

  out_dir.mkdir(parents=True, exist_ok=True)
  base = image_path.stem
  overlay = _overlay(img, mask)
  boundary = _boundary_image(img, mask) if make_boundary else None

  mask_path_out = out_dir / f"{base}_service_mask.png"
  overlay_path_out = out_dir / f"{base}_service_overlay.png"
  boundary_path_out = out_dir / f"{base}_service_boundary.png"
  card_path_out = out_dir / f"{base}_service_card.png"
  summary_path_out = out_dir / f"{base}_service_summary.json"
  csv_path_out = out_dir / f"{base}_service_summary.csv"

  Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path_out, compress_level=0)
  overlay.save(overlay_path_out, compress_level=0)
  if make_boundary:
    boundary.save(boundary_path_out, compress_level=0)

  damage_pixels = int(mask.sum())
  image_pixels = int(mask.size)
  ratio = damage_pixels / max(image_pixels, 1)
  severity = _severity_from_ratio(ratio)
  comps = _component_stats(mask)
  summary = {
    "image": str(image_path),
    "mode": mode,
    "input_mask": str(mask_path) if mask_path else None,
    "damage_pixels": damage_pixels,
    "raw_damage_pixels_before_postprocess": raw_damage_pixels,
    "removed_noise_pixels": max(0, raw_damage_pixels - damage_pixels),
    "image_pixels": image_pixels,
    "area_basis": "full_image_pixels",
    "damage_ratio": ratio,
    "damage_ratio_percent": round(ratio * 100, 4),
    "severity": severity,
    "component_count": len(comps),
    "components_top10": comps[:10],
    "recommendation": _recommendation(severity),
    "outputs": {
      "mask": str(mask_path_out),
      "overlay": str(overlay_path_out),
      "boundary": str(boundary_path_out) if make_boundary else None,
      "service_card": str(card_path_out),
      "summary_csv": str(csv_path_out),
    },
    "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
    "disclaimer": "mock_demo_not_model_result is for proposal visualization only. Use prediction_mask mode for real model output visualization.",
  }
  summary["auto_explanation_ko"] = _korean_explanation(summary)
  summary["llm_role_note"] = "CV model or CV heuristic creates the mask and measurements; an LLM should only verbalize this summary if connected."

  if make_card:
    _draw_service_card(img, overlay, summary, card_path_out)
  else:
    summary["outputs"]["service_card"] = None
  with open(summary_path_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
  with open(csv_path_out, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=[
        "image",
        "mode",
        "damage_ratio_percent",
        "severity",
        "component_count",
        "recommendation",
        "auto_explanation_ko",
        "overlay",
        "service_card",
      ],
    )
    writer.writeheader()
    writer.writerow({
      "image": str(image_path),
      "mode": mode,
      "damage_ratio_percent": round(ratio * 100, 4),
      "severity": severity,
      "component_count": len(comps),
      "recommendation": _recommendation(severity),
      "overlay": str(overlay_path_out),
      "service_card": str(card_path_out),
    })
  return summary


def _collect_images(input_dir: Path, recursive: bool = False):
  pattern = "**/*" if recursive else "*"
  items = []
  for p in input_dir.glob(pattern):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
      items.append(p)
  return sorted(items)


def _find_mask_for_image(image_path: Path, mask_dir: Path):
  candidates = []
  for suffix in ["_mask", "_pred", "_prediction", ""]:
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
      candidates.append(mask_dir / f"{image_path.stem}{suffix}{ext}")
  for c in candidates:
    if c.exists():
      return c
  return None


def _write_batch_summary(summaries, out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / "service_batch_summary.json"
  csv_path = out_dir / "service_batch_summary.csv"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)
  with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=["image", "mode", "damage_ratio_percent", "severity", "component_count", "recommendation", "auto_explanation_ko", "overlay", "service_card"],
    )
    writer.writeheader()
    for s in summaries:
      writer.writerow({
        "image": s.get("image"),
        "mode": s.get("mode"),
        "damage_ratio_percent": s.get("damage_ratio_percent"),
        "severity": s.get("severity"),
        "component_count": s.get("component_count"),
        "recommendation": s.get("recommendation"),
        "auto_explanation_ko": s.get("auto_explanation_ko", ""),
        "overlay": s.get("outputs", {}).get("overlay"),
        "service_card": s.get("outputs", {}).get("service_card"),
      })
  return json_path, csv_path


def main():
  parser = argparse.ArgumentParser(description="Create road-damage service visualization from segmentation mask.")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--image", help="Single input road image path")
  group.add_argument("--input_dir", help="Folder containing many input road images")
  parser.add_argument("--mask", default=None, help="Predicted binary/label mask path for single --image")
  parser.add_argument("--mask_dir", default=None, help="Folder containing predicted masks for --input_dir")
  parser.add_argument("--outdir", default="runs/service_visual", help="Output directory")
  parser.add_argument("--mock", action="store_true", help="Create proposal-only demo mask when no real model mask is available")
  parser.add_argument("--recursive", action="store_true", help="Search input_dir recursively")
  parser.add_argument("--no_card", action="store_true", help="Skip service_card PNG for faster large-batch processing")
  parser.add_argument("--no_boundary", action="store_true", help="Skip boundary PNG for faster large-batch processing")
  parser.add_argument("--fallback_to_mock_if_bad_mask", action="store_true", help="If a supplied prediction mask is nearly empty/full, use conservative mock demo mask instead and record the mode.")
  parser.add_argument("--min_area_pixels", type=int, default=80, help="Remove connected damage components smaller than this pixel area before overlay/percent calculation. Set 0 to disable.")
  args = parser.parse_args()

  out_dir = Path(args.outdir)

  if args.image:
    summary = process_one(
      image_path=Path(args.image),
      out_dir=out_dir,
      mask_path=Path(args.mask) if args.mask else None,
      mock=args.mock,
      make_card=not args.no_card,
      make_boundary=not args.no_boundary,
      fallback_to_mock_if_bad_mask=args.fallback_to_mock_if_bad_mask,
      min_area_pixels=args.min_area_pixels,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return

  input_dir = Path(args.input_dir)
  if not input_dir.exists():
    raise FileNotFoundError(f"input_dir not found: {input_dir}")
  image_list = _collect_images(input_dir, recursive=args.recursive)
  if not image_list:
    raise FileNotFoundError(f"No images found in: {input_dir}")

  mask_dir = Path(args.mask_dir) if args.mask_dir else None
  summaries = []
  for idx, image_path in enumerate(image_list, 1):
    mask_path = None
    if mask_dir is not None:
      mask_path = _find_mask_for_image(image_path, mask_dir)
      if mask_path is None and not args.mock:
        raise FileNotFoundError(f"No matching mask for {image_path.name} in {mask_dir}")
    print(f"[{idx}/{len(image_list)}] {image_path}")
    summaries.append(process_one(image_path=image_path, out_dir=out_dir, mask_path=mask_path, mock=args.mock, make_card=not args.no_card, make_boundary=not args.no_boundary, fallback_to_mock_if_bad_mask=args.fallback_to_mock_if_bad_mask, min_area_pixels=args.min_area_pixels))

  json_path, csv_path = _write_batch_summary(summaries, out_dir)
  print(f"Saved batch summary: {json_path}")
  print(f"Saved batch csv: {csv_path}")


if __name__ == "__main__":
  main()
