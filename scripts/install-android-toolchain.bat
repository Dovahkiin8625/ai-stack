@echo off
REM One-shot installer for the AI Stack Android build toolchain.
REM See top of this file for usage; structure here uses goto labels for error
REM handling because cmd forbids nested if statements inside (...) blocks.

setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%CD%"
set "SDK=%LOCALAPPDATA%\Android\Sdk"
set "JDK=C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot"

echo.
echo === 1. JDK 17 ===
if exist "%JDK%\bin\java.exe" goto jdk_ok
echo Downloading Temurin 17 MSI ...
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.13+11/OpenJDK17U-jdk_x64_windows_hotspot_17.0.13_11.msi' -OutFile '%TEMP%\temurin17.msi' -UseBasicParsing"
if errorlevel 1 goto fail
echo Installing MSI (UAC prompt will appear) ...
msiexec /i "%TEMP%\temurin17.msi" /passive /norestart INSTALLDIR="%JDK%"
if errorlevel 1 goto fail
:jdk_ok
setx JAVA_HOME "%JDK%" >nul
if errorlevel 1 goto fail
echo OK: JDK 17 ready at %JDK%

echo.
echo === 2. Android SDK cmdline-tools ===
set "CMDLINE=%SDK%\cmdline-tools\latest\bin\sdkmanager.bat"
if exist "%CMDLINE%" goto cmdline_ok
echo Downloading commandlinetools ...
if not exist "%SDK%\cmdline-tools\latest" mkdir "%SDK%\cmdline-tools\latest"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip' -OutFile '%TEMP%\commandlinetools.zip' -UseBasicParsing"
if errorlevel 1 goto fail
echo Extracting ...
powershell -NoProfile -Command "Expand-Archive -Path '%TEMP%\commandlinetools.zip' -DestinationPath '%SDK%\cmdline-tools\latest' -Force"
if errorlevel 1 goto fail
if exist "%SDK%\cmdline-tools\latest\cmdline-tools" (
    move /Y "%SDK%\cmdline-tools\latest\cmdline-tools\*" "%SDK%\cmdline-tools\latest\" >nul 2>&1
    rmdir "%SDK%\cmdline-tools\latest\cmdline-tools"
)
:cmdline_ok
setx ANDROID_HOME "%SDK%" >nul
setx ANDROID_SDK_ROOT "%SDK%" >nul
set "PATH=%SDK%\cmdline-tools\latest\bin;%SDK%\platform-tools;%PATH%"
echo OK: cmdline-tools ready

echo.
echo === 3. Accept licenses and install components ===
echo Accepting SDK licenses ...
REM Feed a stack of "y" lines so the interactive accept proceeds without a TTY.
set "Y=y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y& y"
set "Y=%Y: =y&%"
set "Y=%Y:~2%"
set "Y=y&%Y%"
echo %Y% | "%CMDLINE%" --licenses >nul 2>&1
if errorlevel 1 goto fail
echo OK: licenses accepted

call "%CMDLINE%" "platform-tools" >nul 2>&1
if errorlevel 1 goto fail
echo OK: platform-tools installed

call "%CMDLINE%" "platforms;android-35" >nul 2>&1
if errorlevel 1 goto fail
echo OK: platforms;android-35 installed

call "%CMDLINE%" "build-tools;35.0.0" >nul 2>&1
if errorlevel 1 goto fail
echo OK: build-tools;35.0.0 installed

echo.
echo === 4. NDK r27.1.12297006 ===
set "NDK=%SDK%\ndk\27.1.12297006"
if exist "%NDK%\bin\ndk-build.cmd" goto ndk_ok
echo Installing NDK r27.1.12297006 (about 1.5 GB, slow) ...
call "%CMDLINE%" "ndk;27.1.12297006" >nul 2>&1
if errorlevel 1 goto fail
:ndk_ok
setx NDK_HOME "%NDK%" >nul
echo OK: NDK ready at %NDK%

echo.
echo === 5. Rust Android targets ===
where cargo >nul 2>&1
if errorlevel 1 set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
call rustup target add aarch64-linux-android >nul 2>&1
if errorlevel 1 goto fail
echo OK: aarch64-linux-android added
call rustup target add armv7-linux-androideabi >nul 2>&1
if errorlevel 1 goto fail
echo OK: armv7-linux-androideabi added

echo.
echo === Done ===
echo JAVA_HOME   = %JDK%
echo ANDROID_HOME = %SDK%
echo NDK_HOME     = %NDK%
echo.
echo Close this window, open a new normal cmd, then run:
echo   cd /d %ROOT%
echo   cargo check --target aarch64-linux-android
echo Send me that output and I'll continue with tauri android init + Task 14.
endlocal
exit /b 0

:fail
echo.
echo === FAILED ===
echo Last command exited with errorlevel %errorlevel%.
echo Check the message above; the rest of the toolchain was not installed.
endlocal
exit /b 1
