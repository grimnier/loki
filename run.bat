@echo off
rem Runs the engine. GGML_BIN = the directory holding the ggml CUDA runtime DLLs (ggml.dll, ggml-base.dll, ggml-cuda.dll).
if "%GGML_BIN%"=="" set "GGML_BIN=ggml\bin"
set PATH=%GGML_BIN%;%PATH%
set GGML_CUDA_DISABLE_FUSION=1
"%~dp0src\server_core_fused.exe" %*
