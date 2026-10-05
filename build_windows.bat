@echo off
REM ============================================================
REM  GeneVariate - Windows Build Script
REM  Builds GeneVariate.exe with embedded icon
REM ============================================================

echo ============================================================
echo  GeneVariate Windows Build
echo ============================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller Pillow

REM Generate .icns is not needed on Windows, but ensure .ico exists
if not exist "src\genevariate\assets\icon.ico" (
    echo Generating icon.ico from icon.png...
    python -c "from PIL import Image; img=Image.open('src/genevariate/assets/icon.png'); img.save('src/genevariate/assets/icon.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
)

REM Build with PyInstaller
echo Building GeneVariate.exe...
pyinstaller genevariate.spec --noconfirm

echo.
echo ============================================================
if exist "dist\GeneVariate\GeneVariate.exe" (
    echo  BUILD SUCCESSFUL
    echo  Executable: dist\GeneVariate\GeneVariate.exe
    echo.
    echo  To create a desktop shortcut, right-click GeneVariate.exe
    echo  and select "Create shortcut", then move it to your Desktop.
) else (
    echo  BUILD FAILED - check errors above
)
echo ============================================================
pause
