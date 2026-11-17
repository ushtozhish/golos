@echo off
rem Golos без окна: golos-cli.cmd info ^| voices ^| synth --job job.json --out DIR ^| test-en
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONPATH=%~dp0"
"%~dp0python\python.exe" -m golos.cli %*
