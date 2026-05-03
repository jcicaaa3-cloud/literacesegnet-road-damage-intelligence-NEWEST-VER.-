"""Interactive LLM-style chat for capstone road-damage service results.

Role separation:
- CV/model pipeline creates masks, overlays, percentages, severity, and JSON.
- This chat service only explains those JSON results.

External LLM API mode is optional:
- Set OPENAI_API_KEY to use API mode if the openai package is installed.
- Without an API key, this script still runs in local rule-based mode.
"""

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path


def load_results(path: Path):
  if not path.exists():
    raise FileNotFoundError(
      f"Result summary not found: {path}\n"
      "Run run_batch_infer_service.bat first."
    )
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  if isinstance(data, dict):
    data = [data]
  if not isinstance(data, list):
    raise ValueError("Summary JSON must be a list or dict.")
  return data


def compact_context(items):
  rows = []
  for item in items:
    outputs = item.get("outputs", {}) or {}
    rows.append({
      "image": item.get("image", ""),
      "mode": item.get("mode", ""),
      "damage_ratio_percent": item.get("damage_ratio_percent", 0),
      "severity": item.get("severity", "unknown"),
      "component_count": item.get("component_count", 0),
      "recommendation": item.get("recommendation", ""),
      "overlay": outputs.get("overlay", ""),
      "service_card": outputs.get("service_card", ""),
      "auto_explanation_ko": item.get("auto_explanation_ko", ""),
    })
  return rows


def make_overview(rows):
  if not rows:
    return "분석 결과가 없습니다."
  ratios = [float(r.get("damage_ratio_percent") or 0) for r in rows]
  sev = Counter(str(r.get("severity", "unknown")) for r in rows)
  top = sorted(rows, key=lambda r: float(r.get("damage_ratio_percent") or 0), reverse=True)[:5]
  lines = []
  lines.append(f"총 {len(rows)}장의 도로 이미지 결과가 있습니다.")
  lines.append(f"평균 손상 비율은 약 {statistics.mean(ratios):.2f}%입니다.")
  lines.append("등급 분포: " + ", ".join(f"{k}={v}" for k, v in sev.items()))
  lines.append("손상 비율 상위 이미지:")
  for i, r in enumerate(top, 1):
    lines.append(
      f" {i}. {Path(str(r.get('image',''))).name}: "
      f"{float(r.get('damage_ratio_percent') or 0):.2f}% / "
      f"{r.get('severity')} / components={r.get('component_count')}"
    )
  return "\n".join(lines)


def local_answer(question: str, rows):
  q = question.lower().strip()
  if not q:
    return "질문을 입력해 주세요."
  if q in {"exit", "quit", "q", "종료", "끝"}:
    return "__EXIT__"
  overview = make_overview(rows)
  top = sorted(rows, key=lambda r: float(r.get("damage_ratio_percent") or 0), reverse=True)
  if any(k in q for k in ["요약", "전체", "summary", "현황"]):
    return overview
  if any(k in q for k in ["가장", "높", "심각", "위험", "우선"]):
    if not top:
      return "분석 결과가 없습니다."
    r = top[0]
    return (
      f"가장 우선 확인할 이미지는 {Path(str(r.get('image',''))).name}입니다. "
      f"손상 비율은 약 {float(r.get('damage_ratio_percent') or 0):.2f}%이고, "
      f"등급은 {r.get('severity')}입니다. 권장 조치는 '{r.get('recommendation')}'입니다. "
      f"overlay 결과는 {r.get('overlay')}에서 확인할 수 있습니다."
    )
  if any(k in q for k in ["평균", "퍼센트", "%", "비율"]):
    ratios = [float(r.get("damage_ratio_percent") or 0) for r in rows]
    if not ratios:
      return "비율 계산 결과가 없습니다."
    return f"평균 손상 비율은 약 {statistics.mean(ratios):.2f}%입니다. 최대값은 {max(ratios):.2f}%, 최소값은 {min(ratios):.2f}%입니다."
  if any(k in q for k in ["파일", "결과", "저장", "어디", "경로"]):
    return (
      "주요 결과는 seg/runs/capstone_batch_service 폴더에 저장됩니다. "
      "각 이미지별 *_service_overlay.png, *_service_mask.png, *_service_card.png, *_service_summary.json이 생성되고, "
      "전체 요약은 service_batch_summary.json/csv로 저장됩니다."
    )
  if any(k in q for k in ["llm", "역할", "설명"]):
    return (
      "이 서비스에서 CV 모델 또는 CV 후처리가 mask, overlay, 손상 비율, 등급을 생성합니다. "
      "LLM은 이 JSON 결과를 사용자가 이해하기 쉬운 문장으로 설명하는 보조 역할만 합니다."
    )
  return (
    "현재 로컬 설명 모드입니다. 질문은 '전체 요약', '가장 심각한 이미지', '평균 손상 비율', "
    "'결과 파일 경로', 'LLM 역할'처럼 물어보면 됩니다.\n\n" + overview
  )


def try_external_llm(question: str, rows, model: str):
  if not os.environ.get("OPENAI_API_KEY"):
    return None
  try:
    from openai import OpenAI
  except Exception:
    return None
  system_path = Path(__file__).resolve().parent / "prompts" / "service_chat_system_ko.txt"
  system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else "결과 JSON만 근거로 답하세요."
  context = json.dumps(rows[:30], ensure_ascii=False, indent=2)
  user_prompt = (
    "다음은 도로 손상 캡스톤 서비스의 JSON 요약입니다.\n"
    f"{context}\n\n"
    f"사용자 질문: {question}\n"
  )
  client = OpenAI()
  try:
    resp = client.responses.create(
      model=model,
      input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
    )
    return getattr(resp, "output_text", None) or str(resp)
  except Exception:
    try:
      resp = client.chat.completions.create(
        model=model,
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
        ],
      )
      return resp.choices[0].message.content
    except Exception as exc:
      return f"[외부 LLM 호출 실패: {exc}]\n" + local_answer(question, rows)


def main():
  parser = argparse.ArgumentParser(description="Chat over capstone service result JSON.")
  parser.add_argument("--summary", default="seg/runs/capstone_batch_service/service_batch_summary.json")
  parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
  parser.add_argument("--no_api", action="store_true", help="Force local rule-based chat mode.")
  args = parser.parse_args()

  summary_path = Path(args.summary)
  if not summary_path.is_absolute() and not summary_path.exists():
    alt = Path(__file__).resolve().parents[1] / summary_path
    summary_path = alt
  items = load_results(summary_path)
  rows = compact_context(items)

  print("[LLM SERVICE CHAT]")
  print("CV/model result JSON loaded:", summary_path)
  print("이 채팅은 mask를 만들지 않고, 이미 생성된 결과를 설명합니다.")
  if os.environ.get("OPENAI_API_KEY") and not args.no_api:
    print("외부 LLM API 키가 감지되었습니다. API 설명 모드를 시도합니다.")
  else:
    print("로컬 설명 모드입니다. API 키 없이도 동작합니다.")
  print("\n" + make_overview(rows))
  print("\n질문 예시: 전체 요약 / 가장 심각한 이미지 / 평균 손상 비율 / 결과 파일 경로 / LLM 역할")
  print("종료: exit 또는 종료\n")

  while True:
    try:
      q = input("질문> ").strip()
    except (EOFError, KeyboardInterrupt):
      print("\n종료합니다.")
      break
    if not args.no_api:
      answer = try_external_llm(q, rows, args.model)
    else:
      answer = None
    if answer is None:
      answer = local_answer(q, rows)
    if answer == "__EXIT__":
      print("종료합니다.")
      break
    print("\n답변>")
    print(answer)
    print()


if __name__ == "__main__":
  main()
