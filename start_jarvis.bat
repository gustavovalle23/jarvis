@echo off
title J.A.R.V.I.S
cd /d D:\Jarvis
set HF_HOME=D:\Jarvis\models
call .venv\Scripts\activate.bat
python jarvis.py %*
if errorlevel 1 pause
