@echo off
rem loki-app -- the window. Starts the local server and opens Edge or Chrome as an app window. Set LOKI_HOME to the repo if this file lives elsewhere.
setlocal
set "REPO=%~dp0.."
if exist "%LOKI_HOME%\loki\app.py" set "REPO=%LOKI_HOME%"
python "%REPO%\loki\app.py" %*
