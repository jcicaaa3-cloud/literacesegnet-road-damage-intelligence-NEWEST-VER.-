from __future__ import annotations

import argparse
import sys
from pathlib import Path

SEG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SEG_DIR.parent
if str(SEG_DIR) not in sys.path:
  sys.path.insert(0, str(SEG_DIR))

from core.data_pairs import collect_image_mask_pairs, write_pairing_report


def resolve(p: str) -> Path:
  x = Path(p)
  return x if x.is_absolute() else (PROJECT_ROOT / x).resolve()


def main() -> int:
  ap = argparse.ArgumentParser(description="Check image/mask pairing with typo-tolerant matching.")
  ap.add_argument("--root", default="datasets/pothole_binary/processed")
  ap.add_argument("--outdir", default="seg/runs/dataset_pairing_reports")
  ap.add_argument("--fuzzy_threshold", type=float, default=0.82)
  args = ap.parse_args()

  root = resolve(args.root)
  outdir = resolve(args.outdir)
  total_pairs = 0
  total_images = 0
  failed = False
  for split in ["train", "val"]:
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
      print(f"[ERROR] Missing split folders for {split}:")
      print(f" {image_dir}")
      print(f" {mask_dir}")
      failed = True
      continue
    pairs, rows = collect_image_mask_pairs(image_dir, mask_dir, allow_fuzzy=True, fuzzy_threshold=args.fuzzy_threshold)
    report_path = outdir / f"{split}_pairing_report.csv"
    write_pairing_report(report_path, rows)
    total_pairs += len(pairs)
    total_images += len(rows)
    fuzzy = sum(1 for r in rows if r.get("method") == "fuzzy")
    unmatched = sum(1 for r in rows if not r.get("matched_mask"))
    print(f"[{split}] images={len(rows)} paired={len(pairs)} fuzzy={fuzzy} unmatched={unmatched}")
    print(f"    report={report_path}")
    if unmatched:
      failed = True

  if total_images == 0:
    print("[ERROR] No dataset images found.")
    return 1
  if failed:
    print("\n[STOP] Some image/mask pairs are missing. Fix them or inspect the CSV report.")
    return 1
  print("\n[OK] Dataset pairing check passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
