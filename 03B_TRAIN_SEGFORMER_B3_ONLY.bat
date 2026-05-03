@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Train B] SegFormer-B3 ONLY - Transformer baseline
echo ============================================================
echo This creates:
echo  seg\transformer_b3\checkpoints\segformer_b3_best.pth
echo It does NOT train LiteRaceSegNet.
echo.

if not exist "seg\transformer_b3\hf_pretrained\segformer_b3_ade\config.json" (
  echo [WARN] HuggingFace SegFormer pretrained folder was not found.
  echo [INFO] Running setup first...
  call 02_SETUP_SEGFORMER_B3_HF.bat
  if errorlevel 1 (
    echo [ERROR] SegFormer setup failed.
if not defined NO_PAUSE pause
    exit /b 1
  )
)

if not exist "datasets\pothole_binary\processed\train\images" (
  echo [ERROR] Dataset folder is missing.
  echo Expected:
  echo  datasets\pothole_binary\processed\train\images
  echo  datasets\pothole_binary\processed\train\masks
  echo  datasets\pothole_binary\processed\val\images
  echo  datasets\pothole_binary\processed\val\masks
if not defined NO_PAUSE pause
  exit /b 1
)

echo [CHECK] image-mask pairing with typo-tolerant matching...
python seg\tools\check_dataset_pairs.py --root datasets\pothole_binary\processed
if errorlevel 1 (
  echo [ERROR] Dataset pairing failed. Check seg\runs\dataset_pairing_reports\*.csv
if not defined NO_PAUSE pause
  exit /b 1
)

python seg\transformer_b3\train_segformer_b3.py --config seg\config\pothole_binary_segformer_b3_train.yaml
if errorlevel 1 (
  echo [FAILED] SegFormer-B3 training failed.
if not defined NO_PAUSE pause
  exit /b 1
)

echo.
echo [OK] SegFormer-B3 training finished.
echo Checkpoint: seg\transformer_b3\checkpoints\segformer_b3_best.pth
if not defined NO_PAUSE pause