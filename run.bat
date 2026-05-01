@echo off
title HackerRank Orchestrate AI Agent
color 0A

echo ============================================================
echo      HackerRank Orchestrate AI Support Triage Agent
echo ============================================================
echo.

echo [1/3] Checking dependencies...
pip install -r code/requirements.txt --quiet
if %errorlevel% neq 0 (
    color 0C
    echo Error: Failed to install dependencies. Please ensure Python is installed.
    pause
    exit /b %errorlevel%
)
echo [OK] Dependencies are up to date.
echo.

echo [2/3] Checking environment configuration...
if not exist ".env" (
    color 0E
    echo Warning: .env file not found. Creating one from .env.example...
    copy .env.example .env > nul
    echo Please ensure your ANTHROPIC_API_KEY is properly set in the .env file.
    echo.
) else (
    echo [OK] .env file found.
)
echo.

echo [3/3] Executing Main Agent Pipeline...
echo ------------------------------------------------------------
python code/main.py
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo Error: Agent pipeline encountered a failure.
    pause
    exit /b %errorlevel%
)

echo.
echo ------------------------------------------------------------
echo Running Cross-Verification against Sample Tickets...
echo ------------------------------------------------------------
python code/evaluate.py
if %errorlevel% neq 0 (
    color 0E
    echo.
    echo Warning: Evaluation script encountered an issue.
)

echo.
echo ============================================================
echo   All tasks complete! Check support_tickets/output.csv
echo ============================================================
pause
