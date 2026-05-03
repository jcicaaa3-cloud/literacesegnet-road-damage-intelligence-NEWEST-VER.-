"""Build final capstone evidence folder from separated model outputs.

This script does not train models. It only collects artifacts produced by:
- LiteRaceSegNet training/inference
- SegFormer-B3 training/inference
- CNN vs Transformer comparison
- LiteRace service summary used by the LLM path

Important project rule:
- LiteRaceSegNet and SegFormer-B3 remain separated.
- LLM evidence is generated only from the LiteRace service summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(path_text: str | Path) -> Path:
  p = Path(path_text)
  return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_dir(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  return path


def file_size_mb(path: Path) -> str:
  if not path.exists():
    return "MISSING"
  return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def read_json(path: Path, default):
  if not path.exists():
    return default
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def read_compare_csv(path: Path) -> List[Dict[str, str]]:
  if not path.exists():
    return []
  with open(path, "r", encoding="utf-8-sig", newline="") as f:
    return list(csv.DictReader(f))


def write_text(path: Path, text: str):
  ensure_dir(path.parent)
  with open(path, "w", encoding="utf-8") as f:
    f.write(text)


def copy_matching(src_dir: Path, dst_dir: Path, patterns: Iterable[str], limit: int = 20) -> int:
  ensure_dir(dst_dir)
  if not src_dir.exists():
    return 0
  copied = 0
  seen = set()
  for pattern in patterns:
    for src in sorted(src_dir.glob(pattern)):
      if not src.is_file() or src in seen:
        continue
      seen.add(src)
      shutil.copy2(src, dst_dir / src.name)
      copied += 1
      if copied >= limit:
        return copied
  return copied


def short_model_name(name: str) -> str:
  lowered = name.lower()
  if "literace" in lowered:
    return "LiteRaceSegNet"
  if "segformer" in lowered:
    return "SegFormer-B3"
  return name


def model_family(name: str) -> str:
  lowered = name.lower()
  if "literace" in lowered or "cnn" in lowered:
    return "CNN"
  if "segformer" in lowered or "transformer" in lowered:
    return "Transformer"
  return "-"


def model_feature(name: str) -> str:
  lowered = name.lower()
  if "literace" in lowered or "cnn" in lowered:
    return "경량 CNN, detail/context/boundary 보완"
  if "segformer" in lowered or "transformer" in lowered:
    return "전역 문맥 기반 Transformer baseline"
  return "비교 대상"


def pick(row: Dict[str, str], *keys: str, default: str = "NA") -> str:
  for key in keys:
    value = row.get(key)
    if value is not None and str(value).strip() != "":
      return str(value)
  return default


def fmt(value: str, suffix: str = "") -> str:
  value = str(value)
  if value in {"", "NA", "None", "nan"}:
    return "NA"
  try:
    num = float(value)
    return f"{num:.4f}{suffix}"
  except Exception:
    return value


def _is_number(value: str) -> bool:
  try:
    float(str(value))
    return True
  except Exception:
    return False


def _best_row(rows: List[Dict[str, str]], keyword: str) -> Optional[Dict[str, str]]:
  keyword = keyword.lower()
  for row in rows:
    if keyword in pick(row, "name", default="").lower():
      return row
  return None


def _tradeoff_note(rows: List[Dict[str, str]]) -> str:
  if not rows:
    return "아직 비교 수치가 없습니다. checkpoint와 validation data를 넣고 CPU/GPU 비교를 실행하세요."

  groups: Dict[str, List[Dict[str, str]]] = {}
  for row in rows:
    device = pick(row, "device", default="unknown").lower()
    groups.setdefault(device, []).append(row)

  notes: List[str] = []
  for device, device_rows in groups.items():
    lite = _best_row(device_rows, "literace")
    segformer = _best_row(device_rows, "segformer")
    if not lite or not segformer:
      continue

    device_label = device.upper()
    lite_size = pick(lite, "param_size_mb_fp32", default="NA")
    seg_size = pick(segformer, "param_size_mb_fp32", default="NA")
    lite_lat = pick(lite, "latency_ms", default="NA")
    seg_lat = pick(segformer, "latency_ms", default="NA")
    lite_miou = pick(lite, "miou_binary", default="NA")
    seg_miou = pick(segformer, "miou_binary", default="NA")
    lite_mem = pick(lite, "cuda_peak_memory_mb", default="NA")
    seg_mem = pick(segformer, "cuda_peak_memory_mb", default="NA")

    notes.append(f"### {device_label} 비교")
    if _is_number(lite_size) and _is_number(seg_size):
      ratio = float(seg_size) / max(float(lite_size), 1e-12)
      if ratio > 1:
        notes.append(f"- FP32 파라미터 크기: SegFormer-B3/LiteRaceSegNet = {ratio:.2f}x")
      else:
        notes.append("- FP32 파라미터 크기에서는 LiteRaceSegNet이 SegFormer-B3보다 작게 나오지 않았습니다. config와 모델 로딩 상태를 확인하세요.")
    if _is_number(lite_lat) and _is_number(seg_lat):
      ratio = float(seg_lat) / max(float(lite_lat), 1e-12)
      if ratio > 1:
        notes.append(f"- {device_label} 평균 latency: SegFormer-B3/LiteRaceSegNet = {ratio:.2f}x. LiteRaceSegNet이 더 빠릅니다.")
      else:
        notes.append(f"- {device_label} 평균 latency: LiteRaceSegNet이 SegFormer-B3보다 빠르게 나오지 않았습니다. batch size, image size, device 상태를 같이 확인하세요.")
    if device == "cuda" and _is_number(lite_mem) and _is_number(seg_mem):
      ratio = float(seg_mem) / max(float(lite_mem), 1e-12)
      notes.append(f"- CUDA peak memory: SegFormer-B3/LiteRaceSegNet = {ratio:.2f}x")
    if _is_number(lite_miou) and _is_number(seg_miou):
      diff = float(lite_miou) - float(seg_miou)
      notes.append(f"- mIoU 차이(LiteRaceSegNet - SegFormer-B3): {diff:+.4f}")

  if not notes:
    return "LiteRaceSegNet과 SegFormer-B3 두 행이 같은 device 안에 있어야 trade-off 문장을 자동 생성할 수 있습니다."
  return "\n".join(notes)

def make_comparison_md(rows: List[Dict[str, str]]) -> str:
  header = [
    "모델명", "계열", "Device", "Device Name", "Input", "Batch", "AMP",
    "Params", "FP32 Size", "Latency", "Latency Std", "FPS", "GPU Peak Mem",
    "mIoU", "Damage IoU", "Boundary IoU", "Pixel Acc", "특징"
  ]
  lines = ["# 최종 모델 비교표", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
  if not rows:
    lines.append("| 결과 없음 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 비교 결과 CSV가 아직 생성되지 않았습니다. |")
    return "\n".join(lines) + "\n"

  # CPU rows first, then CUDA rows, so the report reads as deployment -> acceleration.
  def sort_key(row: Dict[str, str]):
    device = pick(row, "device", default="")
    model = pick(row, "name", default="")
    return (0 if device == "cpu" else 1 if device == "cuda" else 2, 0 if "literace" in model.lower() else 1)

  for row in sorted(rows, key=sort_key):
    name = pick(row, "name", default="unknown")
    params = pick(row, "param_million", "params", default="NA")
    if params != "NA" and params == pick(row, "param_million", default=""):
      params = fmt(params, "M")
    size = fmt(pick(row, "param_size_mb_fp32", default="NA"), " MB")
    latency = fmt(pick(row, "latency_ms", default="NA"), " ms")
    latency_std = fmt(pick(row, "latency_std_ms", default="NA"), " ms")
    fps = fmt(pick(row, "throughput_fps", default="NA"), " fps")
    gpu_mem = fmt(pick(row, "cuda_peak_memory_mb", default="NA"), " MB")
    miou = fmt(pick(row, "miou_binary", default="NA"))
    damage_iou = fmt(pick(row, "iou_damage", default="NA"))
    boundary_iou = fmt(pick(row, "boundary_iou", default="NA"))
    pixel = fmt(pick(row, "pixel_acc", default="NA"))
    lines.append(
      "| " + " | ".join([
        short_model_name(name),
        model_family(name),
        pick(row, "device", default="NA"),
        pick(row, "device_name", default="NA"),
        pick(row, "image_size_hw", default="NA"),
        pick(row, "batch_size", default="1"),
        pick(row, "amp", default="False"),
        params,
        size,
        latency,
        latency_std,
        fps,
        gpu_mem,
        miou,
        damage_iou,
        boundary_iou,
        pixel,
        model_feature(name),
      ]) + " |"
    )

  lines.extend([
    "",
    "## 자동 해석 메모",
    "",
    _tradeoff_note(rows),
    "",
    "## 발표용 결론 문장",
    "",
    "LiteRaceSegNet은 SegFormer-B3와 같은 validation layout에서 비교되었고, 표는 segmentation 품질, boundary 품질, 모델 크기, device별 latency, throughput, GPU memory를 함께 보여준다. CPU 결과는 GPU가 없는 현장형 배포 가능성을 보여주고, GPU 결과는 AWS 같은 가속 환경에서의 추론 확장성을 보여준다. LiteRaceSegNet이 더 작은 모델 크기와 낮은 latency를 보이면서 mIoU, Damage IoU, Boundary IoU를 실사용 가능한 수준으로 유지한다면, 경량 도로 손상 segmentation 서비스에 더 적합한 trade-off를 제공한다고 말할 수 있다.",
  ])
  return "\n".join(lines) + "\n"

def make_llm_example(service_rows) -> str:
  if not service_rows:
    return (
      "# LiteRaceSegNet 기반 LLM 서비스 예시\n\n"
      "아직 seg/runs/literace_service/service_batch_summary.json 이 생성되지 않았습니다.\n"
      "먼저 run_batch_infer_service.bat 또는 06_BUILD_FINAL_EVIDENCE_ONLY.bat 를 실행하세요.\n"
    )
  first = service_rows[0]
  image = first.get("image", "unknown")
  ratio = first.get("damage_ratio_percent", "NA")
  severity = first.get("severity", "NA")
  comps = first.get("component_count", "NA")
  recommendation = first.get("recommendation", "")
  explanation = first.get("auto_explanation_ko", "")
  overlay = first.get("outputs", {}).get("overlay") or first.get("overlay", "")
  card = first.get("outputs", {}).get("service_card") or first.get("service_card", "")
  return f"""# LiteRaceSegNet 기반 LLM 서비스 예시

중요: 이 예시는 SegFormer 결과가 아니라, LiteRaceSegNet 서비스 summary를 사람이 이해하기 쉬운 문장으로 설명하는 구조입니다.

## 입력
- 이미지: `{image}`

## CV/LiteRaceSegNet 결과
- 손상 비율: {ratio}%
- 심각도: {severity}
- 주요 연결 영역 수: {comps}
- overlay: `{overlay}`
- service card: `{card}`

## 서비스 문장 예시
{explanation or recommendation or "summary JSON이 생성되었지만 설명 문장이 비어 있습니다."}

## 발표용 설명
LLM은 이미지를 직접 segmentation하지 않습니다. LiteRaceSegNet이 만든 mask, overlay, 손상 비율, summary JSON을 읽고 사용자에게 자연어로 풀어주는 보조 역할입니다.
"""


def checkpoint_manifest(literace_ckpt: Path, segformer_ckpt: Path) -> str:
  rows = [
    ("LiteRaceSegNet", "CNN", literace_ckpt),
    ("SegFormer-B3", "Transformer", segformer_ckpt),
  ]
  lines = ["# Checkpoint Manifest", "", "| 모델 | 계열 | 경로 | 상태/크기 |", "|---|---|---|---|"]
  for name, family, path in rows:
    status = file_size_mb(path)
    lines.append(f"| {name} | {family} | `{path.relative_to(PROJECT_ROOT) if path.exists() else path}` | {status} |")
  lines.append("")
  lines.append("주의: 이 스크립트는 기본적으로 checkpoint 파일을 복사하지 않고 존재 여부와 크기만 기록합니다. 대용량 .pth 중복 복사를 피하기 위한 설계입니다.")
  return "\n".join(lines) + "\n"


def make_summary_md(args, compare_rows, service_rows, copies: Dict[str, int]) -> str:
  literace_ckpt = resolve(args.literace_ckpt)
  segformer_ckpt = resolve(args.segformer_ckpt)
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  checklist = [
    ("LiteRaceSegNet best.pth", literace_ckpt.exists()),
    ("SegFormer-B3 best.pth", segformer_ckpt.exists()),
    ("모델 비교표 CSV/JSON", bool(compare_rows)),
    ("LiteRace overlay/service 결과", copies.get("literace", 0) > 0),
    ("SegFormer overlay 결과", copies.get("segformer", 0) > 0),
    ("LiteRace 기반 LLM 서비스 예시", bool(service_rows)),
  ]
  lines = [
    "# 최종 캡스톤 증거 패키지 요약",
    "",
    f"생성 시각: {now}",
    "",
    "## 최종 발표 한 문장",
    "본 프로젝트는 도로 손상 segmentation을 위해 직접 설계한 경량 CNN 기반 LiteRaceSegNet과 Transformer baseline인 SegFormer-B3를 동일 데이터셋에서 비교하고, CPU 배포 가능성과 GPU 가속 성능까지 함께 분석한 캡스톤 프로젝트입니다.",
    "",
    "## 자동 점검표",
    "",
    "| 항목 | 상태 |",
    "|---|---|",
  ]
  for item, ok in checklist:
    lines.append(f"| {item} | {'OK' if ok else 'MISSING'} |")
  lines.extend([
    "",
    "## 모델 분리 원칙",
    "- LiteRaceSegNet은 LiteRaceSegNet 전용 checkpoint만 사용합니다.",
    "- SegFormer-B3는 SegFormer-B3 전용 checkpoint만 사용합니다.",
    "- 두 모델은 비교표에서만 만납니다.",
    "- LLM 서비스 예시는 LiteRaceSegNet의 service summary만 설명합니다.",
    "",
    "## 결과 폴더 구성",
    "- `01_checkpoints/checkpoint_manifest.md`: checkpoint 존재 여부와 크기",
    "- `02_metrics_and_compare/`: CNN vs Transformer 비교 CSV/JSON. CPU/GPU latency, throughput, 모델 크기, GPU memory, mIoU, Damage IoU, Boundary IoU를 같이 확인합니다.",
    "- `03_literace_overlays/`: LiteRaceSegNet service overlay/card/mask",
    "- `04_segformer_overlays/`: SegFormer-B3 inference overlay/mask",
    "- `05_llm_service_example/`: LiteRaceSegNet summary 기반 설명 예시",
    "- `06_report_ready/`: 보고서/PPT에 바로 붙일 markdown 표와 요약",
    "",
  ])
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description="Collect final capstone evidence artifacts")
  parser.add_argument("--outdir", default="final_evidence")
  parser.add_argument("--compare_dir", default="final_evidence/02_metrics_and_compare")
  parser.add_argument("--gpu_compare_dir", default="", help="Optional CUDA comparison directory. If provided, CPU and GPU rows are merged in the final report.")
  parser.add_argument("--literace_service_dir", default="seg/runs/literace_service")
  parser.add_argument("--segformer_infer_dir", default="seg/runs/segformer_b3_infer_after_train")
  parser.add_argument("--literace_ckpt", default="seg/runs/literace_boundary_degradation/best.pth")
  parser.add_argument("--segformer_ckpt", default="seg/transformer_b3/checkpoints/segformer_b3_best.pth")
  parser.add_argument("--copy_limit", type=int, default=30)
  args = parser.parse_args()

  outdir = ensure_dir(resolve(args.outdir))
  dirs = {
    "checkpoints": ensure_dir(outdir / "01_checkpoints"),
    "compare": ensure_dir(outdir / "02_metrics_and_compare"),
    "literace": ensure_dir(outdir / "03_literace_overlays"),
    "segformer": ensure_dir(outdir / "04_segformer_overlays"),
    "llm": ensure_dir(outdir / "05_llm_service_example"),
    "report": ensure_dir(outdir / "06_report_ready"),
  }

  compare_dir = resolve(args.compare_dir)
  compare_rows = read_compare_csv(compare_dir / "model_compare_summary.csv")
  if (compare_dir / "model_compare_summary.csv").exists():
    shutil.copy2(compare_dir / "model_compare_summary.csv", dirs["compare"] / "model_compare_summary.csv")
  if (compare_dir / "model_compare_summary.json").exists():
    shutil.copy2(compare_dir / "model_compare_summary.json", dirs["compare"] / "model_compare_summary.json")

  gpu_compare_rows: List[Dict[str, str]] = []
  if args.gpu_compare_dir:
    gpu_compare_dir = resolve(args.gpu_compare_dir)
    gpu_compare_rows = read_compare_csv(gpu_compare_dir / "model_compare_summary.csv")
    gpu_dst = ensure_dir(dirs["compare"] / "gpu")
    if (gpu_compare_dir / "model_compare_summary.csv").exists():
      shutil.copy2(gpu_compare_dir / "model_compare_summary.csv", gpu_dst / "model_compare_summary_gpu.csv")
    if (gpu_compare_dir / "model_compare_summary.json").exists():
      shutil.copy2(gpu_compare_dir / "model_compare_summary.json", gpu_dst / "model_compare_summary_gpu.json")

  all_compare_rows = compare_rows + gpu_compare_rows

  service_dir = resolve(args.literace_service_dir)
  service_rows = read_json(service_dir / "service_batch_summary.json", [])
  if (service_dir / "service_batch_summary.json").exists():
    shutil.copy2(service_dir / "service_batch_summary.json", dirs["llm"] / "literace_service_batch_summary.json")
  if (service_dir / "service_batch_summary.csv").exists():
    shutil.copy2(service_dir / "service_batch_summary.csv", dirs["llm"] / "literace_service_batch_summary.csv")

  copies = {}
  copies["literace"] = copy_matching(
    service_dir,
    dirs["literace"],
    ["*_service_card.png", "*_service_overlay.png", "*_service_mask.png", "*_service_boundary.png"],
    limit=args.copy_limit,
  )
  copies["segformer"] = copy_matching(
    resolve(args.segformer_infer_dir),
    dirs["segformer"],
    ["*_overlay.png", "*_mask_color.png", "*_pred_class.png"],
    limit=args.copy_limit,
  )

  write_text(dirs["checkpoints"] / "checkpoint_manifest.md", checkpoint_manifest(resolve(args.literace_ckpt), resolve(args.segformer_ckpt)))
  write_text(dirs["report"] / "final_comparison_table.md", make_comparison_md(all_compare_rows))
  write_text(dirs["llm"] / "literace_llm_service_example.md", make_llm_example(service_rows))
  write_text(dirs["report"] / "capstone_result_summary.md", make_summary_md(args, all_compare_rows, service_rows, copies))

  # Small README directly in the root of final_evidence.
  write_text(outdir / "README_FINAL_EVIDENCE.txt", """Final evidence folder for capstone presentation/report.

Run order after data is ready:
1) 03_TRAIN_BOTH_SEPARATE.bat
2) 10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat on an AWS GPU machine, or 06_BUILD_FINAL_EVIDENCE_ONLY.bat for CPU-only evidence.

Long one-click option:
- 07_FULL_TRAIN_AND_BUILD_FINAL_EVIDENCE.bat

Core rule:
- LiteRaceSegNet and SegFormer-B3 are trained separately.
- Comparison happens only in the comparison table.
- LLM service example explains LiteRaceSegNet output summary only.
""")

  print("[OK] Final evidence package built:", outdir)
  print("-", dirs["report"] / "final_comparison_table.md")
  print("-", dirs["report"] / "capstone_result_summary.md")
  print("-", dirs["llm"] / "literace_llm_service_example.md")


if __name__ == "__main__":
  main()
