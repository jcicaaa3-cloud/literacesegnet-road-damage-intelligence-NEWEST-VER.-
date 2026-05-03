@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Compare] LiteRaceSegNet checkpoint vs SegFormer-B3 checkpoint
echo ============================================================
echo This does NOT train anything. It only compares separated outputs.
echo For AWS GPU evidence, run 09_GPU_ACCELERATION_EVIDENCE.bat or 10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat.
echo.

if not exist "seg\runs\literace_boundary_degradation\best.pth" (
  echo [ERROR] LiteRaceSegNet checkpoint not found.
  echo Run 03A_TRAIN_LITERACESEGNET_ONLY.bat first.
  pause
  exit /b 1
)

if not exist "seg\transformer_b3\checkpoints\segformer_b3_best.pth" (
  echo [ERROR] Fine-tuned SegFormer checkpoint not found.
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
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir seg\runs\model_compare_literace_vs_segformer

pause
