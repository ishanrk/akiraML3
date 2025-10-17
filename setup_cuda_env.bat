@echo off
REM Setup CUDA environment for VS Code
REM This batch file sets up the environment variables needed for CUDA development

echo Setting up CUDA development environment...

REM Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH_V12_6=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

REM Add CUDA to PATH
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

REM Set Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Environment setup complete!
echo You can now run VS Code from this command prompt or use the configured tasks in VS Code.

REM Optional: Launch VS Code with the current environment
REM code .

pause
