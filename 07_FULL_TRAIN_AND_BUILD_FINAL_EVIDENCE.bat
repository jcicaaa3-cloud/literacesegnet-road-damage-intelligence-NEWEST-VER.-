@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [07] FULL PIPELINE: train both models, then build final evidence
echo ============================================================
echo This can take a long time, especially SegFormer-B3.
echo Models remain strictly separated:
echo - LiteRaceSegNet trains only LiteRaceSegNet checkpoint.
echo - SegFormer-B3 trains only SegFormer checkpoint.
echo - LLM evidence reads LiteRaceSegNet service summary only.
echo.

echo [0] Dataset pairing check
python seg\tools\check_dataset_pairs.py ^
 --root datasets\pothole_binary\processed ^
 --outdir seg\runs\dataset_pairing_reports
if errorlevel 1 goto failed

echo.
echo [1] Train LiteRaceSegNet only
python seg\train_literace.py --config seg\config\pothole_binary_literace_train.yaml
if errorlevel 1 goto failed

echo.
echo [2] Train SegFormer-B3 only
python seg\transformer_b3\train_segformer_b3.py --config seg\config\pothole_binary_segformer_b3_train.yaml
if errorlevel 1 goto failed

echo.
echo [3] Build final evidence folder
call 06_BUILD_FINAL_EVIDENCE_ONLY.bat
if errorlevel 1 goto failed

echo.
echo [OK] Full training and CPU evidence pipeline finished.
echo If this is the AWS GPU machine, run 10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat next for CPU+GPU research evidence.
pause
exit /b 0

:failed
echo.
echo [FAILED] Full pipeline stopped. Check the error above.
pause
exit /b 1
