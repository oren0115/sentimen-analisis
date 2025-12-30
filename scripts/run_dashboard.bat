@echo off
echo Starting Dashboard...
echo.
cd /d "%~dp0\.."
python -m streamlit run scripts/dashboard.py
pause

