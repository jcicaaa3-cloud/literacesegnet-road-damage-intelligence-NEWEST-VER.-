# 변경 기록: 캡스톤 강화 버전

## 방향

기존 프로젝트 틀은 유지했다. LiteRaceSegNet, SegFormer-B3, LLM service 구조를 섞지 않았다. 강화한 부분은 “그래서 제안 모델이 뭐가 낫냐”를 말할 수 있게 만드는 비교/문서/증거 생성 레이어다.

## 수정한 주요

### 1. README 주장 강화

- LiteRaceSegNet을 제안 모델로 명확히 고정했다.
- SegFormer-B3는 Transformer baseline으로 분리했다.
- 최종 비교 기준을 accuracy만이 아니라 model size, CPU latency, throughput까지 확장했다.
- 발표에서 쓸 수 있는 해석 문장을 추가했다.

### 2. compare_models.py 강화

- comparison 기본 device를 CPU로 바꿨다.
- `--device cpu|cuda|auto` 옵션을 추가했다.
- `--latency_warmup`을 추가했다.
- latency를 한 번만 재는 게 아니라 반복 측정 후 mean/std/min/max를 저장한다.
- throughput FPS를 계산한다.
- 결과 CSV/JSON에 `device`, `image_size_hw`, `latency_std_ms`, `throughput_fps`가 들어간다.

### 3. batch 파일 강화

- `04_COMPARE_AFTER_SEGFORMER_B3_TRAIN.bat`에서 CPU 기준 latency 측정으로 고쳤다.
- `06_BUILD_FINAL_EVIDENCE_ONLY.bat`의 최종 evidence 비교도 CPU 기준으로 고쳤다.
- `08_CPU_LIGHTWEIGHT_EVIDENCE.bat`를 추가했다.

### 4. final evidence 출력 강화

- `final_comparison_table.md`가 CPU latency, latency std, FPS, Damage IoU까지 보여주도록 고쳤다.
- LiteRaceSegNet과 SegFormer-B3의 size/latency 비율을 자동으로 해석하는 메모를 추가했다.

### 5. 발표/보고서 문서 추가

- `docs/CAPSTONE_CLAIM_GUIDE_KO.md`
- `docs/PRESENTATION_SCRIPT_KO.md`
- `docs/CHANGELOG_CAPSTONE_STRENGTHENING_KO.md`

## 건드리지 않은 것

- LiteRaceSegNet 구조 자체는 바꾸지 않았다.
- SegFormer-B3 adapter 구조를 바꾸지 않았다.
- 학습 loss나 dataset pairing 로직을 바꾸지 않았다.
- LLM service를 주요 모델 평가에 섞지 않았다.


## CPU/GPU 이중 비교 추가

AWS GPU를 빌린 경우 CPU만 측정하지 말고 GPU 결과도 별도로 측정한다. CPU는 현장형 배포 가능성, GPU는 AWS 가속 추론 성능을 설명하는 근거로 쓴다. 최종 표에는 CPU 행 2개와 GPU 행 2개를 같이 넣는다.