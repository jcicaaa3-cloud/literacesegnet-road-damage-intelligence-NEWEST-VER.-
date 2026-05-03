@echo off
setlocal
cd /d %~dp0

echo [ALIAS] This runs SegFormer-B3 ONLY.
echo For both separated trainings, run: 03_TRAIN_BOTH_SEPARATE.bat
echo.
call 03B_TRAIN_SEGFORMER_B3_ONLY.bat
