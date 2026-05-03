# CPU/GPU 이중 비교 연구 계획

## 주장

LiteRaceSegNet은 SegFormer-B3와 같은 validation set에서 비교한다. 
비교는 정확도만 보지 않고, 모델 크기와 하드웨어별 추론 비용까지 같이 본다.

- CPU 비교: GPU가 없는 현장형 환경에서 돌아갈 수 있는가
- GPU 비교: AWS GPU 환경에서 얼마나 빠르게 추론되는가
- 공통 비교: mIoU, Damage IoU, Pixel Accuracy, Params, FP32 Size

## 발표에서 밀 문장

이 프로젝트는 SegFormer-B3를 비교 기준으로 두고, 도로 손상 segmentation에 맞춘 경량 CNN 모델을 직접 설계하고 Transformer baseline과 같은 조건에서 비교한 것입니다. 
CPU 실험은 현장형 배포 가능성을 확인하기 위한 것이고, GPU 실험은 AWS 같은 가속 환경에서의 추론 성능과 메모리 사용량을 확인하기 위한 것입니다.

## 표 구성

| Model | Device | Params | FP32 Size | Latency | FPS | GPU Peak Memory | mIoU | Damage IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LiteRaceSegNet | CPU | | | | | NA | | |
| SegFormer-B3 | CPU | | | | | NA | | |
| LiteRaceSegNet | GPU | | | | | | | |
| SegFormer-B3 | GPU | | | | | | | |

## 해석 기준

LiteRaceSegNet이 SegFormer-B3보다 모든 수치에서 무조건 이겨야 하는 구조가 아니다. 
캡스톤에서 중요한 건 다음 문장을 수치로 말할 수 있느냐다.

> LiteRaceSegNet은 더 작은 모델 비용으로 도로 손상 mask 품질을 유지하며, CPU/GPU 환경에서 더 실용적인 추론 trade-off를 보인다.

## 실행 순서

1. `03_TRAIN_BOTH_SEPARATE.bat`
2. AWS GPU 인스턴스에서 `10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat`
3. `final_evidence/06_report_ready/final_comparison_table.md` 확인

GPU가 없는 로컬 PC에서는 `08_CPU_LIGHTWEIGHT_EVIDENCE.bat`만 실행하면 된다.