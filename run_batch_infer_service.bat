@echo off
setlocal
cd /d %~dp0

if not exist assets\service_demo\input_batch mkdir assets\service_demo\input_batch
if not exist seg\runs\literace_service mkdir seg\runs\literace_service

echo [CAPSTONE BATCH SERVICE - LiteRaceSegNet ONLY]
echo 1. Put road images into: assets\service_demo\input_batch
echo 2. This runs LiteRaceSegNet checkpoint only.
echo 3. SegFormer output is NOT used here.
echo 4. LLM chat reads this LiteRace service summary only.
echo.

if not exist "seg\runs\literace_boundary_degradation\best.pth" (
  echo [ERROR] LiteRaceSegNet checkpoint not found.
  echo Run 03A_TRAIN_LITERACESEGNET_ONLY.bat first.
  echo.
  pause
  exit /b 1
)

python -c "import PIL, numpy" >nul 2>nul
if errorlevel 1 (
  echo [STOP] Required packages are missing. Run 00_INSTALL_REQUIREMENTS.bat first.
  pause
  exit /b 1
)

python seg\capstone_batch_service.py ^
 --input_dir assets/service_demo/input_batch ^
 --outdir seg/runs/literace_service ^
 --model_output_dir seg/runs/literace_service_raw_output ^
 --config seg/config/pothole_binary_literace_train.yaml ^
 --ckpt seg/runs/literace_boundary_degradation/best.pth ^
 --mode model ^
 --min_area_pixels 80

if errorlevel 1 (
  echo.
  echo [FAILED] LiteRace batch service failed. Read traceback above.
  pause
  exit /b 1
)

echo.
echo [OK] LiteRace service result folder: seg\runs\literace_service
echo Main result files:
echo - *_service_overlay.png
echo - *_service_mask.png
echo - *_service_card.png
echo - *_service_summary.json
echo - service_batch_summary.csv/json
echo.
echo Next optional step: run_LLM_CHAT_SERVICE.bat
pause
