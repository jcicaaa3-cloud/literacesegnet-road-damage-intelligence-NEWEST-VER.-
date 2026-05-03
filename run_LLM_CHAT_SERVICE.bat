@echo off
setlocal
cd /d %~dp0

echo [LLM SERVICE CHAT - LiteRaceSegNet result ONLY]
echo This chat reads: seg\runs\literace_service\service_batch_summary.json
echo It does not create masks. It only explains LiteRaceSegNet CV/model results.
echo SegFormer summaries are not connected to this LLM chat path.
echo.

if not exist "seg\runs\literace_service\service_batch_summary.json" (
  echo [ERROR] LiteRace service summary not found.
  echo Run run_batch_infer_service.bat first.
  pause
  exit /b 1
)

python llm_service\chat_service.py --summary seg\runs\literace_service\service_batch_summary.json

if errorlevel 1 (
  echo.
  echo [FAILED] LLM chat service failed. Run LiteRace batch inference first or check the summary path.
  pause
  exit /b 1
)
pause
