"""Robust image-mask pairing helpers for capstone segmentation training.

Purpose
-------
Team members often create masks with slightly different names, such as:
  road_001.jpg + road_001_mask.png
  pothole-12.jpg + pothole_12_label.png
  raod_003.jpg + road_003.png  # small typo

This helper first tries exact/suffix matching, then normalized-stem matching,
and finally a conservative fuzzy match. It writes a report so the director can
check whether an automatic match was correct.
"""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = [
  "", "_mask", "-mask", " mask", "_gt", "-gt", " gt",
  "_label", "-label", " label", "_labels", "-labels",
  "_seg", "-seg", "_annotation", "-annotation", "_annot", "-annot",
  "_정답", "-정답", " 정답", "_마스크", "-마스크", " 마스크",
]

_TRAILING_MASK_WORDS = [
  "mask", "gt", "label", "labels", "seg", "annotation", "annot",
  "정답", "마스크",
]


def list_files(folder: Path, exts: Iterable[str]) -> List[Path]:
  exts = {e.lower() for e in exts}
  if not folder.exists():
    return []
  return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def normalize_stem(stem: str) -> str:
  """Normalize names while preserving numbers and Korean letters.

  Examples:
    road_001_mask -> road001
    Road-001 Label -> road001
    도로_001_마스크 -> 도로001
  """
  s = stem.lower().strip()
  changed = True
  while changed:
    changed = False
    for word in _TRAILING_MASK_WORDS:
      for pat in (f"_{word}", f"-{word}", f" {word}", word):
        if s.endswith(pat) and len(s) > len(pat):
          s = s[: -len(pat)].strip(" _-")
          changed = True
  # Keep letters/numbers/Korean only; remove separators and punctuation.
  return "".join(re.findall(r"[0-9a-z가-힣]+", s))


def number_tokens(stem: str) -> Tuple[str, ...]:
  return tuple(re.findall(r"\d+", stem))


def _exact_suffix_candidates(mask_dir: Path, image_stem: str) -> List[Path]:
  out: List[Path] = []
  for suf in MASK_SUFFIXES:
    for ext in MASK_EXTS:
      p = mask_dir / f"{image_stem}{suf}{ext}"
      if p.exists():
        out.append(p)
  return out


def find_best_mask_for_image(
  mask_dir: Path,
  image_path: Path,
  *,
  allow_fuzzy: bool = True,
  fuzzy_threshold: float = 0.82,
) -> Tuple[Optional[Path], str, float, str]:
  """Return (mask_path, method, score, note)."""
  mask_files = list_files(mask_dir, MASK_EXTS)
  if not mask_files:
    return None, "missing", 0.0, "mask folder has no mask files"

  # 1) Exact stem + common suffixes.
  exact = _exact_suffix_candidates(mask_dir, image_path.stem)
  if exact:
    return exact[0], "exact_or_suffix", 1.0, ""

  # 2) Normalized stem match.
  image_norm = normalize_stem(image_path.stem)
  image_nums = number_tokens(image_path.stem)
  normalized: Dict[str, List[Path]] = {}
  for m in mask_files:
    normalized.setdefault(normalize_stem(m.stem), []).append(m)
  candidates = normalized.get(image_norm, [])
  if candidates:
    if image_nums:
      same_num = [m for m in candidates if number_tokens(m.stem) == image_nums]
      if same_num:
        return same_num[0], "normalized_same_number", 0.98, ""
    return candidates[0], "normalized", 0.95, ""

  # 3) Conservative fuzzy matching for small typos.
  if not allow_fuzzy:
    return None, "unmatched", 0.0, "no exact/normalized match"

  scored: List[Tuple[float, Path, str]] = []
  for m in mask_files:
    mask_norm = normalize_stem(m.stem)
    if not mask_norm:
      continue
    score = SequenceMatcher(None, image_norm, mask_norm).ratio()
    mask_nums = number_tokens(m.stem)
    note = ""
    # Same numeric id is a strong signal: road_001 vs raod_001.
    if image_nums and mask_nums:
      if image_nums == mask_nums:
        score += 0.10
        note = "same number token"
      else:
        score -= 0.18
        note = "different number token; penalized"
    scored.append((score, m, note))

  if not scored:
    return None, "unmatched", 0.0, "no fuzzy candidates"
  scored.sort(key=lambda x: x[0], reverse=True)
  best_score, best_path, best_note = scored[0]
  second_score = scored[1][0] if len(scored) > 1 else -1.0

  # Avoid dangerous auto-pairing when two candidates are almost tied.
  if best_score >= fuzzy_threshold and (best_score - second_score >= 0.03 or best_note == "same number token"):
    return best_path, "fuzzy", float(best_score), best_note
  return None, "unmatched", float(best_score), f"best candidate too weak or ambiguous: {best_path.name}"


def collect_image_mask_pairs(
  image_dir: Path,
  mask_dir: Path,
  *,
  allow_fuzzy: bool = True,
  fuzzy_threshold: float = 0.82,
) -> Tuple[List[Tuple[Path, Path]], List[Dict[str, str]]]:
  pairs: List[Tuple[Path, Path]] = []
  report: List[Dict[str, str]] = []
  images = list_files(image_dir, IMG_EXTS)
  for img in images:
    mask, method, score, note = find_best_mask_for_image(
      mask_dir, img, allow_fuzzy=allow_fuzzy, fuzzy_threshold=fuzzy_threshold
    )
    if mask is not None:
      pairs.append((img, mask))
    report.append({
      "image": img.name,
      "matched_mask": mask.name if mask else "",
      "method": method,
      "score": f"{score:.4f}",
      "note": note,
    })
  return pairs, report


def write_pairing_report(path: Path, rows: Sequence[Dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = ["image", "matched_mask", "method", "score", "note"]
  with open(path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in fieldnames})
