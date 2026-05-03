@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [SegFormer-B3 Setup]
echo 1) Install optional HuggingFace dependencies
echo 2) Download/cache SegFormer-B3 pretrained weights
echo 3) Save a local config for comparison scripts
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not available from CMD.
  echo Install Python or activate your virtual environment first.
if not defined NO_PAUSE pause
  exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements_transformer_optional.txt
if errorlevel 1 (
  echo [ERROR] Optional dependency installation failed.
if not defined NO_PAUSE pause
  exit /b 1
)

python seg\transformer_b3\download_segformer_b3.py --model-id nvidia/segformer-b3-finetuned-ade-512-512 --outdir seg\transformer_b3\hf_pretrained\segformer_b3_ade --write-config
if errorlevel 1 (
  echo [ERROR] SegFormer-B3 download/setup failed.
if not defined NO_PAUSE pause
  exit /b 1
)

echo.
echo [OK] SegFormer-B3 HuggingFace baseline is ready.
echo Local model folder:
echo  seg\transformer_b3\hf_pretrained\segformer_b3_ade
echo.
echo Next command example:
echo  run_COMPARE_MODELS_HF.bat
echo.
if not defined NO_PAUSE pause