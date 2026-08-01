@echo off
setlocal EnableExtensions
chcp 65001 >nul
title A4拼图视觉算法自检
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import cv2, numpy" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import cv2, numpy" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)
if not defined PYTHON_CMD (
    echo [错误] 未找到可用的 Python/OpenCV/NumPy 环境。
    echo 请先双击“安装依赖_首次运行.bat”。
    pause
    exit /b 1
)

%PYTHON_CMD% main.py --config config.json self-test --output-dir self_test_output
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo [通过] A4、分界线、固定拼图、未知拼图和禁止镜像检查均通过。
) else (
    echo [失败] 自检未通过，错误码 %EXIT_CODE%。
)
pause
exit /b %EXIT_CODE%
