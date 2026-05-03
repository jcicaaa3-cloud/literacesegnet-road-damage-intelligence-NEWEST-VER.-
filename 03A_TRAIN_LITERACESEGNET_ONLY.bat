@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Train A] LiteRaceSegNet ONLY - lightweight CNN model
echo ============================================================
echo This creates:
echo  seg\runs\literace_boundary_degradation\best.pth
echo It does NOT train SegFormer.
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not available from CMD.
if not defined NO_PAUSE pause
  exit /b 1
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

python seg\train_literace.py --config seg\config\pothole_binary_literace_train.yaml
if errorlevel 1 (
  echo [FAILED] LiteRaceSegNet training failed.
if not defined NO_PAUSE pause
  exit /b 1
)

echo.
echo [OK] LiteRaceSegNet training finished.
echo Checkpoint: seg\runs\literace_boundary_degradation\best.pth
if not defined NO_PAUSE pause