@echo off
cd /d %~dp0
echo [INSTALL] Capstone service dependencies
python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python numpy PyYAML Pillow tqdm
echo.
echo [OPTIONAL] Transformer baseline support
echo If you want to run SegFormer-B3 comparison, also run:
echo 01_INSTALL_TRANSFORMER_OPTIONAL.bat
echo.
echo [OPTIONAL] LLM API chat support
echo If you want external LLM API mode, also run:
echo python -m pip install openai
echo.
pause
