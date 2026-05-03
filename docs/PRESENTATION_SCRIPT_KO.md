# 발표 스크립트

## 1. 프로젝트 소개

이 프로젝트는 도로 이미지에서 포트홀이나 파손 영역을 픽셀 단위로 분할하는 segmentation 프로젝트입니다. 제안 모델은 LiteRaceSegNet이고, 비교 모델은 SegFormer-B3입니다.

## 2. 모델 구조

LiteRaceSegNet은 큰 Transformer를 그대로 쓰는 대신, 도로 손상에 필요한 세부 위치 정보와 경계 정보를 가볍게 보존하는 방향으로 설계했습니다. detail branch는 고해상도 정보를 유지하고, context branch는 LiteASPP로 주변 문맥을 반영합니다. boundary branch는 경계 학습을 보조하고, fusion 단계에서 detail feature를 조절합니다.

## 3. 비교 기준

비교는 정확도만 보지 않았습니다. 실제 서비스나 현장형 환경에서는 모델 크기와 추론 시간이 중요합니다. 그래서 SegFormer-B3와 같은 validation set에서 mIoU, Damage IoU, Boundary IoU, parameter count, FP32 model size, CPU latency, GPU latency, throughput, CUDA memory를 함께 비교했습니다.

## 4. CPU/GPU를 나눈 이유

CPU 결과는 GPU가 없는 환경에서도 쓸 수 있는지 보는 조건입니다. GPU 결과는 AWS GPU 환경에서 얼마나 가속되는지, memory를 얼마나 쓰는지 보는 조건입니다. CPU와 GPU의 절대 latency를 직접 비교하지 않고, 같은 device 안에서 LiteRaceSegNet과 SegFormer-B3를 비교했습니다.

## 5. 예상 질문: 그래서 뭐가 낫습니까?

LiteRaceSegNet은 SegFormer-B3보다 큰 범용 모델이 아니라, 경량 도로 손상 segmentation을 목표로 한 제안 모델입니다. 이 프로젝트의 주요은 단순 최고 정확도가 아니라, mask 품질과 계산 비용 사이의 균형입니다. LiteRaceSegNet이 더 작은 모델 크기와 낮은 latency를 보이면서 Damage IoU를 유지한다면, 제한된 환경에서 더 실용적인 trade-off를 가진다고 해석할 수 있습니다.

## 6. LLM 서비스 위치

LLM은 segmentation을 직접 수행하지 않습니다. LiteRaceSegNet이 만든 mask, overlay, 손상 비율, component count, JSON summary를 사람이 읽기 쉬운 설명으로 바꾸는 보조 레이어입니다. 모델 성능 평가는 segmentation 비교표에서 이루어집니다.

## 7. 마무리

이 프로젝트는 직접 설계한 LiteRaceSegNet을 SegFormer-B3 baseline과 분리해 비교하고, CPU/GPU 조건에서 cost/performance를 확인하는 구조입니다. 결과표가 나오면 이 표를 기준으로 모델의 장점과 한계를 함께 설명할 수 있습니다.