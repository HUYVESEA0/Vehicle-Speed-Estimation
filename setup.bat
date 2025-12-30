@echo off
REM ============================================================
REM AMD GPU Vehicle Speed Estimation - Setup Script
REM ============================================================

echo.
echo ============================================================
echo   AMD GPU VEHICLE SPEED ESTIMATION - SETUP
echo ============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/7] Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10+ required!
    pause
    exit /b 1
)
echo       ✓ Python OK

echo.
echo [2/7] Creating virtual environment...
if exist venv (
    echo       ! venv already exists, skipping...
) else (
    python -m venv venv
    echo       ✓ Virtual environment created
)

echo.
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
echo       ✓ Environment activated

echo.
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo       ✓ pip upgraded

echo.
echo [5/7] Installing NumPy (version <2.0)...
pip install "numpy>=1.24.0,<2.0.0"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install NumPy
    pause
    exit /b 1
)
echo       ✓ NumPy installed

echo.
echo [6/7] Installing PyTorch CPU...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo       ✓ PyTorch installed

echo.
echo [7/7] Installing remaining dependencies...
pip install "opencv-python>=4.8.0,<4.10.0"
pip install ultralytics supervision
pip install pyyaml python-dotenv tqdm colorama loguru
pip install pandas scikit-learn pillow matplotlib seaborn scipy
pip install imageio imageio-ffmpeg
echo       ✓ Dependencies installed

echo.
echo ============================================================
echo   Installing AMD GPU Support (DirectML)
echo ============================================================
echo.
pip install torch-directml --no-cache-dir
if %errorlevel% neq 0 (
    echo [WARNING] DirectML installation failed. You can install it later.
    echo           pip install torch-directml --no-cache-dir
) else (
    echo       ✓ DirectML installed successfully
)

echo.
echo ============================================================
echo   Creating project structure...
echo ============================================================
echo.

REM Create directories
if not exist "backend\core" mkdir backend\core
if not exist "backend\utils" mkdir backend\utils
if not exist "config" mkdir config
if not exist "scripts" mkdir scripts
if not exist "data" mkdir data
if not exist "output\videos" mkdir output\videos
if not exist "output\data" mkdir output\data
if not exist "logs" mkdir logs
if not exist "models" mkdir models

echo       ✓ Directories created

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Test installation: python test_installation.py
echo   3. Read documentation: START_HERE.txt
echo.
echo ============================================================
pause
