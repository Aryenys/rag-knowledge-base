@echo off
chcp 65001 >nul
title Start MySQL80 Service

NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Requesting administrator privilege, please click YES on the UAC dialog...
    PowerShell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo === [1/4] Starting MySQL80 service ===
net start MySQL80
echo.

echo === [2/4] Waiting for port 3306 ===
set tries=0
:wait
set /a tries+=1
netstat -ano | findstr ":3306 " | findstr "LISTENING" >nul
if %errorlevel%==0 goto ready
if %tries% GEQ 30 (
    echo    TIMEOUT: MySQL did not come up in 60 seconds.
    echo    Open services.msc, find MySQL80, start it manually.
    pause
    exit /b 1
)
timeout /t 2 >nul
goto wait

:ready
echo    OK: MySQL is listening on 3306
echo.

echo === [3/4] Service state ===
sc query MySQL80 | findstr "STATE"
echo.

echo === [4/4] Set start type to Automatic (optional, avoid manual start next time) ===
sc config MySQL80 start= delayed-auto
echo.

echo ============================================
echo  MySQL is up. Go back and tell me "done".
echo ============================================
pause
