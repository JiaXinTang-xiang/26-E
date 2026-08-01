@echo off
setlocal EnableExtensions
chcp 65001 >nul
title A4拼图视觉上位机 - Windows PC
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
    echo [错误] 未找到同时装有 OpenCV 和 NumPy 的 Python 3。
    echo 请先双击“安装依赖_首次运行.bat”。
    pause
    exit /b 1
)

set "CAMERA_SOURCE=%~1"
if not defined CAMERA_SOURCE set "CAMERA_SOURCE=usb:0"
set "START_MODE=%~2"
if not defined START_MODE set "START_MODE=fixed"
set "RUN_SECONDS=%~3"
set "SERVER_PORT=%~4"
if not defined SERVER_PORT set "SERVER_PORT=8000"

echo 摄像头：%CAMERA_SOURCE%
echo 初始模式：%START_MODE%
echo 浏览器将在服务就绪后自动打开：http://127.0.0.1:%SERVER_PORT%/
echo 关闭服务请回到此窗口按 Ctrl+C。
echo.
if defined RUN_SECONDS (
    %PYTHON_CMD% upper_computer_pc.py --source "%CAMERA_SOURCE%" --mode "%START_MODE%" --source-region upper --port "%SERVER_PORT%" --no-browser --run-seconds "%RUN_SECONDS%"
) else (
    %PYTHON_CMD% upper_computer_pc.py --source "%CAMERA_SOURCE%" --mode "%START_MODE%" --source-region upper --port "%SERVER_PORT%"
)
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [错误] 上位机异常退出，错误码 %EXIT_CODE%。
    echo 请查看“..\05_说明书\故障排查指南.md”，或在本窗口向上查看报错。
    pause
)
exit /b %EXIT_CODE%
