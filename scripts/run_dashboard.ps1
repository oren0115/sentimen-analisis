# PowerShell script to run Streamlit dashboard
Write-Host "Starting Dashboard..." -ForegroundColor Green
Write-Host ""
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot
python -m streamlit run scripts/dashboard.py

