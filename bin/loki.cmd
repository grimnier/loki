@echo off
rem loki -- the terminal client. Copy to %USERPROFILE%\.local\bin\loki.cmd (or anywhere on PATH); it finds the repo from its own location, or LOKI_HOME.
setlocal
set "REPO=%~dp0.."
if exist "%LOKI_HOME%\loki\chat.py" set "REPO=%LOKI_HOME%"
python "%REPO%\loki\chat.py" %*
