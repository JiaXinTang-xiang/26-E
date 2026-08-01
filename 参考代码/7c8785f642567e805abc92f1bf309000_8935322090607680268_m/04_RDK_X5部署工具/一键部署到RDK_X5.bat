@echo off
setlocal EnableExtensions
chcp 65001 >nul
title 一键部署 A4 拼图视觉到 RDK X5
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD set "PYTHON_CMD=python"

%PYTHON_CMD% -c "import paramiko" >nul 2>&1
if errorlevel 1 (
    echo 首次部署需要安装 Paramiko……
    %PYTHON_CMD% -m pip install -r requirements_deploy.txt
    if errorlevel 1 (
        echo [错误] Paramiko 安装失败。
        pause
        exit /b 1
    )
)

set /p "RDK_HOST=请输入 RDK X5 IP（直接回车使用 192.168.1.9）："
if not defined RDK_HOST set "RDK_HOST=192.168.1.9"
set /p "START_MODE=请输入初始模式 fixed/unknown-white/unknown-pattern（默认 fixed）："
if not defined START_MODE set "START_MODE=fixed"

%PYTHON_CMD% deploy_rdk.py --host "%RDK_HOST%" --project-dir "%~dp0..\01_RDK_X5_有上位机版" --mode "%START_MODE%" --bind-host 0.0.0.0
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo [完成] RDK X5 已更新并重启视觉服务。
) else (
    echo [失败] 部署错误码 %EXIT_CODE%，请查看部署说明和上方错误。
)
pause
exit /b %EXIT_CODE%
