# 논문 원고와 GitHub 저장소 연결

논문 원고의 CPU/GPU 강화 흐름과 이 저장소의 실행 흐름을 맞춘 문서다.

## 논문 4장: 제안 모델

연결 파일:

- `seg/core/lightweight_race.py`
- `seg/core/model_select.py`
- `seg/config/pothole_binary_literace_train.yaml`
- `docs/assets/literace_architecture_clean.png`

설명 포인트:

- LiteRaceSegNet은 proposed lightweight CNN model이다.
- SegFormer-B3는 제안 구조가 아니라 baseline이다.
- detail/context/boundary 흐름을 구조도와 코드가 맞춰 보여준다.

## 논문 5장: 실험 설계

연결 파일:

- `seg/compare/compare_models.py`
- `08_CPU_LIGHTWEIGHT_EVIDENCE.bat`
- `09_GPU_ACCELERATION_EVIDENCE.bat`
- `10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat`
- `scripts/run_cpu_evidence.sh`
- `scripts/run_gpu_evidence.sh`
- `scripts/run_dual_device_evidence.sh`

설명 포인트:

- CPU condition과 GPU condition을 분리한다.
- 같은 validation set, 같은 mask layout, 같은 metric 기준으로 두 모델을 비교한다.
- CPU latency와 GPU latency의 절대값을 직접 비교하지 않는다.

## 논문 6장: 결과

연결 파일:

- `final_evidence/02_metrics_and_compare_cpu/model_compare_summary.csv`
- `final_evidence/02_metrics_and_compare_gpu/model_compare_summary.csv`
- `final_evidence/06_report_ready/final_comparison_table.md`

결과 기입 원칙:

- 실험 전에는 수치를 만들지 않는다.
- `[실험 후 기입]` 항목은 위 CSV/MD에서 가져온다.
- LiteRaceSegNet이 더 작은지, 빠른지, memory가 낮은지, IoU가 어느 정도 유지되는지를 같은 device 안에서 해석한다.

## 논문 7장: 논의

연결 파일:

- `docs/RESULT_TABLE_TEMPLATE.md`
- `MODEL_CARD.md`
- `ASSET_AND_LICENSE_POLICY.md`

설명 포인트:

- 결과가 좋으면 trade-off 관점으로 주장한다.
- 결과가 애매하면 실패 사례와 한계를 같이 쓴다.
- 취업용 GitHub에서는 과장보다 재현성과 분리 설계가 더 좋게 보인다.