@echo off
setlocal
if exist "%~dp0..\python_embeded\python.exe" (
  "%~dp0..\python_embeded\python.exe" "%~dp0install_models.py" %*
) else if exist "%~dp0..\..\python_embeded\python.exe" (
  "%~dp0..\..\python_embeded\python.exe" "%~dp0install_models.py" %*
) else if exist "%~dp0..\venv\Scripts\python.exe" (
  "%~dp0..\venv\Scripts\python.exe" "%~dp0install_models.py" %*
) else (
  python "%~dp0install_models.py" %*
)
set "INSTALL_RESULT=%ERRORLEVEL%"
pause
exit /b %INSTALL_RESULT%
