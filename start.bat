@echo off
chcp 65001 >nul
echo.
echo  PreTrainAudit v1.1
echo  ==================
echo.
echo  Installing dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt -q --disable-pip-version-check
echo.
echo  Starting backend...
echo.
echo  Open your browser at:  http://localhost:8000
echo  API docs at:           http://localhost:8000/docs
echo  Health check at:       http://localhost:8000/health
echo.
echo  DO NOT open frontend\index.html directly.
echo  Always use http://localhost:8000 in the browser.
echo.
echo  Press Ctrl+C to stop the server.
echo  ===================================
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
