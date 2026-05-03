# AWS GPU 실행 가이드

이 문서는 AWS GPU 인스턴스에서 LiteRaceSegNet과 SegFormer-B3를 비교하기 위한 실행 순서다.

## 1. 권장 순서

1. 저장소 업로드 또는 clone
2. Python 가상환경 생성
3. base dependency 설치
4. Transformer optional dependency 설치
5. dataset을 `datasets/pothole_binary/processed` 아래에 배치
6. LiteRaceSegNet 학습
7. SegFormer-B3 학습
8. CPU evidence 생성
9. GPU evidence 생성
10. final evidence 생성

## 2. 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_transformer_optional.txt
```

CUDA 버전과 PyTorch wheel은 AWS 이미지에 따라 다를 수 있다. PyTorch가 CUDA를 잡는지 먼저 확인한다.

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
PY
```

## 3. Dataset 확인

```bash
python seg/tools/check_dataset_pairs.py --root datasets/pothole_binary/processed
```

문제가 있으면 학습 전에 image/mask 파일명부터 맞춘다.

## 4. 학습

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_train.yaml
python seg/transformer_b3/train_segformer_b3.py --config seg/config/pothole_binary_segformer_b3_train.yaml
```

## 5. CPU evidence

```bash
bash scripts/run_cpu_evidence.sh
```

의미:

- GPU 없는 환경에서 쓸 수 있는지 확인
- CPU latency, FPS, parameter count, FP32 size 확인

## 6. GPU evidence

```bash
bash scripts/run_gpu_evidence.sh
```

의미:

- AWS GPU에서 얼마나 빨라지는지 확인
- CUDA latency, throughput, peak memory 확인

## 7. CPU + GPU 통합 evidence

```bash
bash scripts/run_dual_device_evidence.sh
```

결과 확인:

```text
final_evidence/06_report_ready/final_comparison_table.md
final_evidence/06_report_ready/capstone_result_summary.md
final_evidence/02_metrics_and_compare_cpu/model_compare_summary.csv
final_evidence/02_metrics_and_compare_gpu/model_compare_summary.csv
```

## 8. 발표/면접에서 말할 때

CPU와 GPU를 직접 비교하지 않는다.

좋은 설명:

> CPU 조건에서는 no-GPU 배포 가능성을 봤고, GPU 조건에서는 AWS 가속 환경에서 throughput과 memory를 봤습니다. 각 조건 안에서 LiteRaceSegNet과 SegFormer-B3를 같은 입력 크기와 같은 validation set으로 비교했습니다.

피해야 할 설명:

> GPU가 CPU보다 빠르니까 모델이 좋습니다.

그건 하드웨어 차이일 뿐 모델 비교가 아니다.