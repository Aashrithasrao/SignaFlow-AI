@echo off
echo ========================================================
echo Starting SignaFlow AI on Windows...
echo ========================================================

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    python -m streamlit run app.py
) else (
    echo [NOTE] Virtual environment not found in .venv.
    echo If you need to set up the environment:
    echo   1. Install Python 3.11 or 3.12 from python.org
    echo   2. Run: py -3.11 -m venv .venv
    echo   3. Run: .venv\Scripts\activate
    echo   4. Run: pip install -r requirements.txt
    echo.
    echo Attempting to launch with system Streamlit...
    python -m streamlit run app.py
)
pause
