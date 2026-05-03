@echo off
setlocal
cd /d %~dp0
set NO_PAUSE=1

echo ============================================================
echo [Train Both - Strictly Separated]
echo 1) LiteRaceSegNet ONLY
echo 2) SegFormer-B3 ONLY
echo ============================================================
echo No mixed checkpoint. No shared output folder.
echo.

echo [STEP 1/2] LiteRaceSegNet training starts.
call 03A_TRAIN_LITERACESEGNET_ONLY.bat
if errorlevel 1 (
  echo [STOP] LiteRaceSegNet training failed. SegFormer will NOT start.
  pause
  exit /b 1
)

echo.
echo [STEP 2/2] SegFormer-B3 training starts.
call 03B_TRAIN_SEGFORMER_B3_ONLY.bat
if errorlevel 1 (
  echo [STOP] SegFormer-B3 training failed.
  pause
  exit /b 1
)

echo.
echo [OK] Both models trained separately.
echo LiteRaceSegNet: seg\runs\literace_boundary_degradation\best.pth
echo SegFormer-B3: seg\transformer_b3\checkpoints\segformer_b3_best.pth
echo.
echo Next: 04_COMPARE_AFTER_SEGFORMER_B3_TRAIN.bat
pause
