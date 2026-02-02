@echo off
REM Quick test script before pushing to GitHub
REM Run this on Windows to verify everything is working

echo ================================================================================
echo SAM3 PRE-PUSH VERIFICATION
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please create one with: python -m venv .venv
    echo Then install dependencies: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
echo     [OK] Virtual environment activated
echo.

REM Run verification script
echo [2/4] Running setup verification...
python scripts\verify_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup verification failed!
    echo Please fix the issues above before pushing to GitHub.
    pause
    exit /b 1
)
echo.

REM Check git status
echo [3/4] Checking git status...
git status --short
echo.

REM List changed files
echo [4/4] Files ready to commit:
echo.
git status --short | findstr /R "^[AM]"
echo.

echo ================================================================================
echo VERIFICATION COMPLETE
echo ================================================================================
echo.
echo Everything looks good! Ready to push to GitHub.
echo.
echo Next steps:
echo   1. Review changed files above
echo   2. git add .
echo   3. git commit -m "Fix SAM3 inference scripts and add documentation"
echo   4. git push origin main
echo   5. Pull on EC2: git pull origin main
echo   6. Run inference on EC2!
echo.
echo ================================================================================
pause
