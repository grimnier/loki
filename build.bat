@echo off
rem Builds src\server_core_fused.exe. Needs: CUDA Toolkit (nvcc, sm_89), Visual Studio 2022 Build Tools, and a CUDA build of
rem llama.cpp's ggml. Set these before running (or edit the three lines):
rem   GGML_INC = the ggml include directory (ggml.h, ggml-backend.h, ggml-cuda.h)
rem   GGML_LIB = the directory with ggml.lib, ggml-base.lib, ggml-cuda.lib
rem   CUDA_PATH = the CUDA Toolkit root
setlocal
if "%GGML_INC%"=="" set "GGML_INC=ggml\include"
if "%GGML_LIB%"=="" set "GGML_LIB=ggml\lib"
if "%CUDA_PATH%"=="" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=sm_89 -allow-unsupported-compiler -Xcompiler "/EHsc /MD" -c -o src\fused_layer.obj src\fused_layer.cu
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /MD /std:c++17 src\server_core_fused.cpp src\fused_layer.obj /I "%GGML_INC%" /I "%CUDA_PATH%\include" /Fe:src\server_core_fused.exe /Fo:src\server_core_fused.obj /link "%GGML_LIB%\ggml.lib" "%GGML_LIB%\ggml-base.lib" "%GGML_LIB%\ggml-cuda.lib" "%CUDA_PATH%\lib\x64\cudart.lib"
