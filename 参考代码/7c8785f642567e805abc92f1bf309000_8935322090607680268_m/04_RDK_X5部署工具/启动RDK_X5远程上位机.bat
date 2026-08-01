@echo off
setlocal EnableExtensions
chcp 65001 >nul
title 启动 RDK X5 远程上位机
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD set "PYTHON_CMD=python"

set /p "RDK_HOST=请输入 RDK X5 IP（直接回车使用 192.168.1.9）："
if not defined RDK_HOST set "RDK_HOST=192.168.1.9"
%PYTHON_CMD% upper_computer.py --host "%RDK_HOST%" --mode fixed --source-region upper
if errorlevel 1 pause
