# 캡스톤 주장 정리: LiteRaceSegNet vs SegFormer-B3

## 한 줄 주장

도로 손상 segmentation을 위해 직접 설계한 경량 CNN 모델인 LiteRaceSegNet을 제안하고, SegFormer-B3 Transformer baseline과 같은 validation layout에서 비교한다. 평가는 정확도만 보지 않고 모델 크기, CPU latency, GPU latency, throughput, CUDA memory, Damage IoU, Boundary IoU까지 함께 본다.

## 질문

> 도로 손상 mask 품질을 유지하면서, SegFormer-B3보다 더 낮은 계산 비용으로 쓸 수 있는 경량 모델을 만들 수 있는가?

## 피해야 할 주장

- LiteRaceSegNet이 모든 조건에서 SegFormer-B3보다 낫다.
- GPU가 빠르니 모델이 좋다.
- 데모 overlay가 좋아 보이니 성능이 좋다.
- dataset과 checkpoint를 넣지 않았는데도 결과가 재현된다고 말한다.

## 써도 되는 주장

- LiteRaceSegNet은 제안 모델이고, SegFormer-B3는 비교 baseline이다.
- 같은 validation set, 같은 mask rule, 같은 metric table로 비교한다.
- CPU 결과는 no-GPU 배포 가능성을 보기 위한 것이다.
- GPU 결과는 AWS 가속 환경에서 throughput과 memory 사용량을 보기 위한 것이다.
- 최종 주장은 `final_comparison_table.md`의 수치를 보고 작성한다.

## 질문 대응

### Q. 그래서 네 모델이 실제로 뭐가 낫습니까?

A. 이 프로젝트에서는 최고 정확도 하나만 보지 않았습니다. LiteRaceSegNet은 도로 손상 segmentation에 맞춘 경량 CNN이고, SegFormer-B3는 Transformer baseline입니다. 같은 validation set에서 모델 크기, parameter count, CPU latency, GPU latency, throughput, memory, Damage IoU, Boundary IoU를 같이 비교했습니다. LiteRaceSegNet이 더 작은 비용으로 mask 품질을 유지한다면, 현장형 또는 비용 제한 환경에서 더 좋은 trade-off를 가진다고 볼 수 있습니다.

### Q. CPU와 GPU를 왜 둘 다 봅니까?

A. 목적이 다릅니다. CPU는 GPU가 없는 환경에서도 동작 가능한지 보는 조건이고, GPU는 AWS 같은 가속 환경에서 추론량과 memory 사용량을 보는 조건입니다. 그래서 CPU끼리, GPU끼리 비교해야 합니다.

### Q. SegFormer를 썼으면 직접 만든 게 약해 보이지 않습니까?

A. 오히려 baseline을 분리했기 때문에 비교가 더 명확합니다. SegFormer-B3는 강한 Transformer baseline이고, LiteRaceSegNet은 제안 모델입니다. 둘을 같은 표에 놓고 cost/performance를 비교하는 것이 이 프로젝트의 주요 목적입니다.

## 결과별 해석

| 결과 패턴 | 해석 |
| --- | --- |
| LiteRaceSegNet이 mIoU, Damage IoU, Boundary IoU도 높고 latency도 낮음 | 가장 강한 결과. 정확도와 비용 둘 다 장점이 있다고 말할 수 있음 |
| LiteRaceSegNet이 IoU는 비슷하고 latency/size가 낮음 | 경량 배포 trade-off가 좋다고 말하기 좋음 |
| LiteRaceSegNet이 IoU는 조금 낮고 latency/size가 크게 낮음 | 실시간성 또는 비용 제한 환경의 trade-off로 해석 가능 |
| LiteRaceSegNet이 latency도 낮지 않음 | 실패 분석과 구조 개선 방향을 정직하게 쓰는 쪽이 낫다 |

## 보고서 문장 템플릿

> LiteRaceSegNet은 SegFormer-B3와 같은 validation layout에서 비교되었다. 본 실험은 segmentation 품질뿐 아니라 parameter count, FP32 model size, CPU latency, GPU latency, throughput, CUDA peak memory, Damage IoU, Boundary IoU를 함께 측정했다. 이를 통해 제안 모델이 도로 손상 segmentation에서 어느 정도의 cost/performance trade-off를 보이는지 확인했다.