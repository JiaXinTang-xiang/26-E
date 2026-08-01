@echo off
setlocal EnableExtensions
chcp 65001 >nul
title 安装 A4 拼图视觉依赖
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [错误] 未找到 Python 3，请先安装 64 位 Python 3.10 至 3.12。
    pause
    exit /b 1
)

echo 正在安装 Windows PC 版依赖……
%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 goto :failed
%PYTHON_CMD% -m pip install -r requirements_pc.txt
if errorlevel 1 goto :failed

echo.
echo [完成] 依赖安装成功，现在可双击“启动电脑版上位机.bat”。
pause
exit /b 0

:failed
echo.
echo [失败] 依赖安装未完成，请检查网络、Python 版本和代理设置。
pause
exit /b 2
