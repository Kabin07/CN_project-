$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host " IoT Malware Detection - CNN + INT8"
Write-Host "============================================"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python 3.10 virtual environment..."
    py -3.10 -m venv .venv
}

Write-Host "Activating environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Running complete pipeline..."
python cn_project.py

Write-Host "Generating additional graphs..."
python make_graphs.py

Write-Host ""
Write-Host "============================================"
Write-Host " PROJECT COMPLETE"
Write-Host "============================================"
Write-Host "Results: .\results\project_results_summary.csv"
Write-Host "Graphs : .\plots\"
Write-Host "Models : .\models\"
