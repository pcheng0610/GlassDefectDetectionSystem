@echo off
chcp 65001 >nul
title 玻璃缺陷检测系统 - 启动服务器

echo ========================================
echo    玻璃缺陷检测系统
echo    正在启动服务器...
echo ========================================
echo.

cd /d %~dp0glass-yolov8

echo [1/2] 启动 Flask 后端服务器...
echo 服务器地址：http://127.0.0.1:5002
echo.

python app.py

pause
