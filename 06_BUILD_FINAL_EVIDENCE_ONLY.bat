@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [06] Build final capstone evidence folder - NO TRAINING
echo ============================================================
echo This step assumes both checkpoints already exist.
echo - LiteRaceSegNet: seg\runs\literace_boundary_degradation\best.pth
echo - SegFormer-B3:  seg\transformer_b3\checkpoints\segformer_b3_best.pth
echo.
echo It will create/update: final_evidence\
echo.

if not exist "seg\runs\literace_boundary_degradation\best.pth" (
  echo [ERROR] LiteRaceSegNet checkpoint missing.
  echo Run 03A_TRAIN_LITERACESEGNET_ONLY.bat or 03_TRAIN_BOTH_SEPARATE.bat first.
  pause
  exit /b 1
)

if not exist "seg\transformer_b3\checkpoints\segformer_b3_best.pth" (
  echo [ERROR] SegFormer-B3 checkpoint missing.
  echo Run 03B_TRAIN_SEGFORMER_B3_ONLY.bat or 03_TRAIN_BOTH_SEPARATE.bat first.
  pause
  exit /b 1
)

if not exist "assets\service_demo\input_batch" mkdir "assets\service_demo\input_batch"

python seg\tools\check_dataset_pairs.py ^
 --root datasets\pothole_binary\processed ^
 --outdir seg\runs\dataset_pairing_reports
if errorlevel 1 goto failed

echo.
echo [A] LiteRaceSegNet service inference for LLM path
python seg\capstone_batch_service.py ^
 --input_dir assets/service_demo/input_batch ^
 --outdir seg/runs/literace_service ^
 --model_output_dir seg/runs/literace_service_raw_output ^
 --config seg/config/pothole_binary_literace_train.yaml ^
 --ckpt seg/runs/literace_boundary_degradation/best.pth ^
 --mode model ^
 --min_area_pixels 80
if errorlevel 1 goto failed

echo.
echo [B] SegFormer-B3 inference only - separated from LLM path
python seg\infer_seg.py ^
 --config seg\config\pothole_binary_segformer_b3_train.yaml ^
 --ckpt seg\transformer_b3\checkpoints\segformer_b3_best.pth ^
 --input_dir assets\service_demo\input_batch ^
 --output_dir seg\runs\segformer_b3_infer_after_train
if errorlevel 1 goto failed

echo.
echo [C] CNN vs Transformer CPU comparison table
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
echo [D] Collect report-ready outputs
python seg\tools\build_final_evidence_package.py ^
 --outdir final_evidence ^
 --compare_dir final_evidence\02_metrics_and_compare_cpu ^
 --literace_service_dir seg\runs\literace_service ^
 --segformer_infer_dir seg\runs\segformer_b3_infer_after_train
if errorlevel 1 goto failed

echo.
echo [OK] Final evidence folder is ready: final_evidence\
echo Open these files:
echo - final_evidence\06_report_ready\final_comparison_table.md
echo - final_evidence\06_report_ready\capstone_result_summary.md
echo - final_evidence\05_llm_service_example\literace_llm_service_example.md
echo.
pause
exit /b 0

:failed
echo.
echo [FAILED] Check the error message above.
pause
exit /b 1
