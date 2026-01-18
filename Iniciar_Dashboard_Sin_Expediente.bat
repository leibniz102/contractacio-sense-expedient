@echo off
REM ============================================================
REM Script para ejecutar el Dashboard de Contractacio Sense Expedient
REM Ejecuta desde WSL (Windows Subsystem for Linux)
REM Universitat Jaume I - Vicegerencia de Recursos Humanos
REM ============================================================

title Dashboard Contractacio Sense Expedient - UJI

echo.
echo ========================================
echo   CONTRACTACIO SENSE EXPEDIENT - UJI
echo   Comparativa 2024 vs 2025
echo ========================================
echo.

REM Verificar que WSL esta disponible
where wsl >nul 2>nul
if errorlevel 1 (
    echo ERROR: WSL no esta instalado o no esta disponible
    echo.
    echo Por favor, instala WSL desde la Microsoft Store
    echo o verifica que esta habilitado en tu sistema.
    echo.
    pause
    exit /b 1
)

echo [OK] WSL encontrado
echo.

REM Verificar que el entorno virtual existe en WSL
wsl bash -c "test -f ~/claude-test-project/venv_python/bin/activate && echo OK || echo FAIL" > %TEMP%\wsl_venv_check.txt
set /p VENV_CHECK=<%TEMP%\wsl_venv_check.txt
del %TEMP%\wsl_venv_check.txt

if "%VENV_CHECK%" == "FAIL" (
    echo ERROR: No se encuentra el entorno virtual en WSL
    echo.
    echo Ruta esperada: ~/claude-test-project/venv_python
    echo.
    pause
    exit /b 1
)

echo [OK] Entorno virtual encontrado
echo.

REM Verificar que Streamlit esta instalado
wsl bash -c "source ~/claude-test-project/venv_python/bin/activate && python3 -c 'import streamlit' 2>/dev/null && echo OK || echo FAIL" > %TEMP%\wsl_streamlit_check.txt
set /p STREAMLIT_CHECK=<%TEMP%\wsl_streamlit_check.txt
del %TEMP%\wsl_streamlit_check.txt

if "%STREAMLIT_CHECK%" == "FAIL" (
    echo [!] Streamlit no esta instalado
    echo.
    echo Instalando Streamlit y dependencias...
    wsl bash -c "source ~/claude-test-project/venv_python/bin/activate && pip install streamlit pandas plotly openpyxl pyarrow"
    echo.
)

echo [OK] Streamlit instalado
echo.

REM Verificar archivo de datos (opcional - el usuario puede subirlo desde la interfaz)
wsl bash -c "test -f ~/claude-test-project/SCAG/datos/'SIN EXPEDIENTE.xlsx' && echo OK || echo FAIL" > %TEMP%\wsl_data_check.txt
set /p DATA_CHECK=<%TEMP%\wsl_data_check.txt
del %TEMP%\wsl_data_check.txt

if "%DATA_CHECK%" == "OK" (
    echo [OK] Archivo de datos encontrado
) else (
    echo [!] Archivo de datos no encontrado - podras subirlo desde la interfaz
)
echo.

REM Puerto especifico para esta aplicacion
set PORT=8520

echo ========================================
echo   Iniciando aplicacion Streamlit...
echo ========================================
echo.
echo La aplicacion se abrira automaticamente en tu navegador.
echo Si no se abre, accede manualmente a: http://localhost:%PORT%
echo.
echo Para DETENER la aplicacion, presiona Ctrl+C
echo.
echo ========================================
echo.

REM Abrir navegador automaticamente despues de 3 segundos (en segundo plano)
start /B cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:%PORT%"

REM Ejecutar Streamlit desde WSL
wsl bash -c "cd ~/claude-test-project/SCAG && source ../venv_python/bin/activate && streamlit run app_sin_expediente.py --server.port %PORT% --server.headless true"

REM Si hay error, pausar para ver el mensaje
if errorlevel 1 (
    echo.
    echo ========================================
    echo   ERROR al ejecutar la aplicacion
    echo ========================================
    echo.
    pause
)
