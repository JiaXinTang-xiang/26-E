@echo off
setlocal
chcp 65001 >nul
call "%~dp0一键部署到RDK_X5.bat"
exit /b %ERRORLEVEL%
