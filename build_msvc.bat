@echo off
REM Build qwen-asr with MSVC
REM Usage: build_msvc.bat
REM Output: qwen_asr.exe in the same directory

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Auto-setup MSVC environment if not already configured
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Setting up MSVC environment...
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo ERROR: vswhere.exe not found. Install Visual Studio with C++ workload.
        exit /b 1
    )
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -property installationPath`) do set "VSINSTALL=%%i"
    if not defined VSINSTALL (
        echo ERROR: Could not find Visual Studio installation
        exit /b 1
    )
    call "!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    where cl.exe >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to set up MSVC environment
        exit /b 1
    )
)

echo Building qwen_asr.exe...

set CFLAGS=/nologo /O2 /arch:AVX2 /fp:fast /std:c17 /W3 /D_CRT_SECURE_NO_WARNINGS /DNDEBUG
set SOURCES=main.c qwen_asr.c qwen_asr_audio.c qwen_asr_decoder.c qwen_asr_encoder.c qwen_asr_kernels.c qwen_asr_kernels_avx.c qwen_asr_kernels_generic.c qwen_asr_safetensors.c qwen_asr_tokenizer.c
set BLAS_CFLAGS=
set BLAS_LIBS=

REM Use OpenBLAS if available
set "OPENBLAS_DIR=%DEPS_ROOT%\openblas"
if exist "%OPENBLAS_DIR%\openblas_msvc.lib" (
    echo Using OpenBLAS from %OPENBLAS_DIR%
    set BLAS_CFLAGS=/DUSE_BLAS /I"%OPENBLAS_DIR%\include"
    set BLAS_LIBS="%OPENBLAS_DIR%\openblas_msvc.lib"
) else (
    echo OpenBLAS not found, building without BLAS
)

cl %CFLAGS% %BLAS_CFLAGS% %SOURCES% /Fe:qwen_asr.exe /link /SUBSYSTEM:CONSOLE %BLAS_LIBS%
if %ERRORLEVEL% NEQ 0 (
    echo Build FAILED
    exit /b 1
)

echo Build OK: qwen_asr.exe
del /q *.obj >nul 2>&1
