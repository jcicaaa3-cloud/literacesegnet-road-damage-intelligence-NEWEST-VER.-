@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Compare] LiteRaceSegNet/CNN candidate vs HuggingFace SegFormer-B3
echo CPU latency is measured for lightweight deployment evidence.
echo ============================================================
echo.

if not exist "seg\transformer_b3\hf_pretrained\segformer_b3_ade\config.json" (
  echo [WARN] Local HuggingFace SegFormer folder was not found.
  echo Run 02_SETUP_SEGFORMER_B3_HF.bat first.
  pause
  exit /b 1
)

python seg\compare\compare_models.py ^
 --configs seg\config\pothole_binary_literace_train.yaml seg\config\pothole_binary_segformer_b3_hf.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_HF_Transformer ^
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir seg\runs\model_compare_hf

pause
