@echo off
title Dashboard Contractacio Sense Expedient - UJI
echo.
echo ========================================
echo  Dashboard Contractacio Sense Expedient
echo  Universitat Jaume I
echo ========================================
echo.
echo Iniciant aplicacio...
echo.

wsl -d Ubuntu -e bash -c "cd ~/claude-test-project/SCAG && source ../venv_python/bin/activate && streamlit run app_sin_expediente.py --server.headless true"

pause
