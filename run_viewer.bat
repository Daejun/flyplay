@echo off
rem Start the sandbox viewer and open it in the browser once the port answers.
rem This is what the desktop shortcut runs. The window stays open so that
rem Ctrl+C stops the fly and the session is saved on the way out.
title flyplay sandbox viewer
cd /d "%~dp0"
set PORT=8000

if not exist ".venv\Scripts\python.exe" (
  echo .venv not found in "%CD%" -- see README.md, section 2.
  pause
  exit /b 1
)

rem The viewer takes about 20 s to compile the model, so wait for the page to
rem answer instead of opening a browser at something nothing is serving yet.
rem The wait asks for the page rather than opening a socket and dropping it:
rem a dropped connection makes http.server print a WinError 10053 traceback
rem into this window, which looks like a crash and is not one.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 240;$i++){try{$null=Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Uri 'http://localhost:%PORT%/';Start-Process 'http://localhost:%PORT%/';break}catch{Start-Sleep -Milliseconds 500}}"

rem Anything typed after the shortcut (or passed by a copy of it) is handed on,
rem so --session other --fresh and the rest of 09_web_viewer.py's flags work.
.venv\Scripts\python.exe scripts\09_web_viewer.py --sandbox --port %PORT% %*

echo.
echo The viewer has stopped. Press any key to close this window.
pause >nul
