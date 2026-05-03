@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [10] Dual-device research evidence: CPU + GPU comparison
echo ============================================================
echo This does NOT train anything.
echo It runs the same LiteRaceSegNet vs SegFormer-B3 comparison twice:
echo [A] CPU: deployment / no-GPU field scenario
echo [B] GPU: AWS acceleration scenario
echo Then it builds one report-ready table with both devices.
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

echo.
echo [A] CPU comparison
python seg\compare\compare_models.py ^
 --configs seg\config\pothole_binary_literace_train.yaml seg\config\pothole_binary_segformer_b3_train.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer ^
 --ckpts seg\runs\literace_boundary_degradation\best.pth seg\transformer_b3\checkpoints\segformer_b3_best.pth ^
 --input_dir datasets\pothole_binary\processed\val\images ^
 --mask_dir datasets\pothole_binary\processed\val\masks ^
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir final_evidence\02_metrics_and_compare_cpu
if errorlevel 1 goto failed

echo.
echo [B] GPU comparison
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
echo [C] Build final report-ready table with CPU + GPU rows
python seg\tools\build_final_evidence_package.py ^
 --outdir final_evidence ^
 --compare_dir final_evidence\02_metrics_and_compare_cpu ^
 --gpu_compare_dir final_evidence\02_metrics_and_compare_gpu ^
 --literace_service_dir seg\runs\literace_service ^
 --segformer_infer_dir seg\runs\segformer_b3_infer_after_train
if errorlevel 1 goto failed

echo.
echo [OK] Dual-device evidence is ready:
echo - final_evidence\06_report_ready\final_comparison_table.md
echo - final_evidence\02_metrics_and_compare_cpu\model_compare_summary.csv
echo - final_evidence\02_metrics_and_compare_gpu\model_compare_summary.csv
echo.
pause
exit /b 0

:failed
echo.
echo [FAILED] Check the error message above.
echo For step [B], this must run on a CUDA-enabled AWS GPU instance.
pause
exit /b 1
