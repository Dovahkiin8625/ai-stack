@echo off
setlocal
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cd /d C:\project\ai-stack
call npx --no-install tauri dev