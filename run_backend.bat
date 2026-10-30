@echo off
cd /d D:\Final_Projects\Flood_Guard_AI
call D:\Flood_Ai\.venv\Scripts\activate.bat
uvicorn backend.main:app --host 127.0.0.1 --port 8001
pause
