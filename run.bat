@echo off
echo Starting A_Team_Agent...

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if .env file has API key
findstr /c:"your_actual_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo WARNING: Please edit .env file and add your OpenAI API key
    echo Current placeholder: your_actual_api_key_here
    echo.
    pause
)

echo Starting Streamlit app...
streamlit run app.py

pause