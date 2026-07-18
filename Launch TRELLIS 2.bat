@echo off
setlocal
cd /d "%~dp0"
start "TRELLIS.2" powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0windows\launch.ps1"
endlocal
