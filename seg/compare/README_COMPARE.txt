모델 비교 스크립트 설명
======================

주요 목적:
- LiteRaceSegNet과 SegFormer-B3를 같은 표에서 비교한다.
- 단순 정확도만 보지 않고, 파라미터 수 / FP32 모델 크기 / latency / throughput / GPU memory까지 같이 본다.
- CPU 결과는 no-GPU 현장형 배포 가능성을 보여준다.
- GPU 결과는 AWS CUDA 환경에서의 가속 추론 가능성을 보여준다.

실행 파일:
- 04_COMPARE_AFTER_SEGFORMER_B3_TRAIN.bat
- 08_CPU_LIGHTWEIGHT_EVIDENCE.bat
- 09_GPU_ACCELERATION_EVIDENCE.bat
- 10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat

CPU 직접 실행 예시:

python seg/compare/compare_models.py ^
 --configs seg/config/pothole_binary_literace_train.yaml seg/config/pothole_binary_segformer_b3_train.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer ^
 --ckpts seg/runs/literace_boundary_degradation/best.pth seg/transformer_b3/checkpoints/segformer_b3_best.pth ^
 --input_dir datasets/pothole_binary/processed/val/images ^
 --mask_dir datasets/pothole_binary/processed/val/masks ^
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir final_evidence/02_metrics_and_compare_cpu

GPU 직접 실행 예시:

python seg/compare/compare_models.py ^
 --configs seg/config/pothole_binary_literace_train.yaml seg/config/pothole_binary_segformer_b3_train.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer ^
 --ckpts seg/runs/literace_boundary_degradation/best.pth seg/transformer_b3/checkpoints/segformer_b3_best.pth ^
 --input_dir datasets/pothole_binary/processed/val/images ^
 --mask_dir datasets/pothole_binary/processed/val/masks ^
 --device cuda ^
 --batch_size 1 ^
 --latency_warmup 20 ^
 --latency_repeats 100 ^
 --outdir final_evidence/02_metrics_and_compare_gpu

비교 결과 저장 위치:
- model_compare_summary.csv
- model_compare_summary.json

비교 가능한 항목:
1. params / param_million
  모델 파라미터 수. 경량성 비교의 1차 근거다.
2. param_size_mb_fp32
  파라미터를 float32로 저장한다고 가정했을 때의 대략적인 크기다.
3. device / device_name
  비교에 사용한 장치와 하드웨어 이름이다.
4. image_size_hw
  config의 입력 크기다. latency는 입력 크기에 영향을 받기 때문에 같이 남긴다.
5. latency_ms
  warmup 이후 반복 forward pass의 평균 추론 시간이다.
6. latency_std_ms / latency_min_ms / latency_max_ms
  한 번 찍은 값이 아니라 반복 측정의 분산을 보여준다.
7. throughput_fps
  latency 평균값과 batch size로 계산한 초당 처리 이미지 수다.
8. cuda_peak_memory_mb
  CUDA profiling에서 측정한 peak allocated memory다. CPU에서는 NA다.
9. batch_size / amp
  latency profiling 조건이다. 기본 batch size는 single-image service 기준인 1이다.
10. pixel_acc / miou_binary / iou_damage / boundary_iou
  checkpoint와 정답 mask 폴더를 같이 제공했을 때만 계산된다. boundary_iou는 예측 mask와 정답 mask의 경계 품질을 비교한다.

발표에서 말할 포인트:
- SegFormer-B3는 강한 Transformer baseline이다.
- LiteRaceSegNet은 제안 모델이다.
- CPU 비교는 GPU가 없는 환경에서 경량 모델이 실사용 가능한지 보는 실험이다.
- GPU 비교는 AWS GPU 환경에서 두 모델의 추론 가속과 메모리 사용량을 보는 실험이다.
- LiteRaceSegNet이 더 작은 모델 크기와 낮은 latency를 보이면서 mIoU/Damage IoU/Boundary IoU가 쓸 만하면, 실사용 관점에서 더 좋은 trade-off라고 말할 수 있다.

주의:
- checkpoint가 없으면 정확도 지표는 NA가 된다.
- SegFormer-B3는 transformers 패키지가 설치되어야 생성된다.
- CPU/GPU latency는 하드웨어, 입력 크기, batch size, 백그라운드 작업에 따라 달라진다.
- 최종 보고서에는 CPU 이름, GPU 이름, 입력 크기, batch size, latency_repeats 값을 같이 적는 게 좋다.
- 이 스크립트의 목적은 “Transformer가 무조건 CNN을 이긴다”가 아니라, 경량성/속도/정확도/메모리를 같은 표에서 비교하기 위한 것이다.
