# Golos: установка/обновление компонентов (Python 3.12 + библиотеки под вашу видеокарту).
# Запускается установщиком; можно запустить повторно из меню «Пуск» → «Golos — обновить компоненты».
param([switch]$NoPause)
$ErrorActionPreference = "Continue"  # uv пишет прогресс в stderr — это не ошибка
$app = Split-Path -Parent $MyInvocation.MyCommand.Path
$uv = Join-Path $app "uv.exe"
$venv = Join-Path $app ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $app "python"
$env:UV_CACHE_DIR = Join-Path $env:LOCALAPPDATA "Golos\uv-cache"
$env:UV_LINK_MODE = "copy"
$Host.UI.RawUI.WindowTitle = "Golos — установка компонентов"

function Fail($msg) {
    Write-Host ""
    Write-Host "ОШИБКА: $msg" -ForegroundColor Red
    Write-Host "Пришлите этот текст в чат — разберёмся." -ForegroundColor Yellow
    Read-Host "Нажмите Enter, чтобы закрыть"
    exit 1
}

Write-Host "=== Golos: установка компонентов ===" -ForegroundColor Cyan
Write-Host "Будет скачано около 3–4 ГБ (Python, PyTorch с CUDA, библиотеки)."
Write-Host "Модели голоса (~8 ГБ) скачаются при первом использовании."
Write-Host ""

try {
    if (-not (Test-Path $py)) {
        Write-Host "[1/3] Python 3.12…" -ForegroundColor Cyan
        & $uv venv $venv --python 3.12 --python-preference only-managed
        if ($LASTEXITCODE -ne 0) { Fail "не удалось создать окружение Python" }
    } else {
        Write-Host "[1/3] Python уже установлен" -ForegroundColor Cyan
    }

    Write-Host "[2/3] Библиотеки (PyTorch под вашу видеокарту, Qwen3-TTS, Whisper, Qt)…" -ForegroundColor Cyan
    & $uv pip install --python $py --upgrade --torch-backend auto -r (Join-Path $app "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Автоопределение не сработало, ставлю сборку CUDA 12.8…" -ForegroundColor Yellow
        & $uv pip install --python $py --upgrade --torch-backend cu128 -r (Join-Path $app "requirements.txt")
        if ($LASTEXITCODE -ne 0) { Fail "не удалось установить библиотеки" }
    }

    Write-Host "[3/3] Проверка видеокарты…" -ForegroundColor Cyan
    & $py -c "import torch;print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'видеокарта NVIDIA не найдена — будет медленно (CPU)')"
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch не запускается" }
    & $py -c "import PySide6, sounddevice, soundfile, qwen_tts, transformers; print('Компоненты в порядке')"
    if ($LASTEXITCODE -ne 0) { Fail "не все компоненты импортируются" }
}
catch { Fail $_.Exception.Message }

Write-Host ""
Write-Host "Готово!" -ForegroundColor Green
if (-not $NoPause) { Start-Sleep -Seconds 3 }
