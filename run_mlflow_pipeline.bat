@echo off
cd /d C:\MLflow\mlflow-day1

call .venv312\Scripts\activate.bat

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

python automated_pipeline.py >> pipeline.log 2>&1