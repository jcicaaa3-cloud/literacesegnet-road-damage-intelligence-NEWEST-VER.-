@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [09] GPU acceleration evidence: LiteRaceSegNet vs SegFormer-B3
echo ============================================================
echo This does NOT train anything.
echo It builds CUDA latency / throughput / GPU memory evidence.
echo Run this on the AWS GPU instance after both checkpoints exist.
echo.

if not exist "seg\runs\literace_boundary_degradation\best.pth" (
  echo [ERROR] LiteRaceSegNet checkpoint not found.
  echo Run 03A_TRAIN_LITERACESEGNET_ONLY.bat first.
  pause
  exit /b 1
)

if not exist "seg\transformer_b3\checkpoints\segformer_b3_best.pth" (
  echo [ERROR] Fine-tuned SegFormer-B3 checkpoint not found.
  echo Run 03B_TRAIN_SEGFORMER_B3_ONLY.bat first.
  pause
  exit /b 1
)

python seg\compare\compare_models.py ^
 --configs seg\config\pothole_binary_literace_train.yaml seg\config\pothole_binary_segformer_b3_train.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer ^
 --ckpts seg\runs\literace_boundary_degradation\best.pth seg\transformer_b3\checkpoints\segformer_b3_best.pth ^
 --input_dir datasets\pothole_binary\processed\val\images ^
 --mask_dir datasets\pothole_binary\processed\val\masks ^
 --device cuda ^
 --batch_size 1 ^
 --latency_warmup 20 ^
 --latency_repeats 100 ^
 --outdir final_evidence\02_metrics_and_compare_gpu
if errorlevel 1 goto failed

echo.
echo [OK] GPU acceleration evidence is ready:
echo - final_evidence\02_metrics_and_compare_gpu\model_compare_summary.csv
echo - final_evidence\02_metrics_and_compare_gpu\model_compare_summary.json
echo.
pause
exit /b 0

:failed
echo.
echo [FAILED] Check the error message above.
echo If CUDA is unavailable, run this on the AWS GPU instance, not local CPU.
pause
exit /b 1
