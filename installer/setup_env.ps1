# Golos: установка/обновление компонентов (Python 3.12 + библиотеки под вашу видеокарту).
# Запускается установщиком; можно запустить повторно из меню «Пуск» → «Golos — обновить компоненты».
param([switch]$NoPause)
$ErrorActionPreference = "Continue"  # uv пишет прогресс в stderr — это не ошибка
$app = Split-Path -Parent $MyInvocation.MyCommand.Path
$uv = Join-Path $app "uv.exe"
$py = Join-Path $app "python\python.exe"
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
    if (-not (Test-Path $py)) { Fail "не найден встроенный Python: $py — переустановите Golos" }
    Write-Host "[1/4] Python: " -NoNewline -ForegroundColor Cyan; & $py --version

    # --- видеокарта и драйвер ---
    Write-Host "[2/4] Видеокарта…" -ForegroundColor Cyan
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
    $backend = "cpu"; $drv = 0
    if ($gpu) {
        $p = $gpu.DriverVersion.Split(".")
        $d = ($p[2] + $p[3]); $d = $d.Substring($d.Length - 5)
        $drv = [double]($d.Substring(0, 3) + "." + $d.Substring(3))
        # CUDA 13.0 нужен драйвер 580+; иначе сборка CUDA 12.6 (работает с драйверами 528+)
        $backend = if ($drv -ge 580) { "cu130" } else { "cu126" }
        Write-Host ("    {0}, драйвер {1:N2}" -f $gpu.Name, $drv)
        if ($drv -lt 528) { Fail ("драйвер NVIDIA {0:N2} слишком старый. Установите свежий драйвер NVIDIA (Studio) и запустите «Golos — обновить компоненты»" -f $drv) }
    } else {
        Write-Host "    Видеокарта NVIDIA не найдена — будет работать на процессоре (медленно)" -ForegroundColor Yellow
    }

    # --- библиотеки ---
    Write-Host "[3/4] Библиотеки (PyTorch под видеокарту, Qwen3-TTS, Whisper, Qt)…" -ForegroundColor Cyan
    $req = Join-Path $app "requirements.txt"
    & $uv pip install --python $py --system --break-system-packages --upgrade --torch-backend auto -r $req
    if ($LASTEXITCODE -ne 0 -and $backend -ne "cpu") {
        & $uv pip install --python $py --system --break-system-packages --upgrade --torch-backend $backend -r $req
    }
    if ($LASTEXITCODE -ne 0) { Fail "не удалось установить библиотеки" }
    # страховка: на компьютере с NVIDIA не должно остаться сборки PyTorch для процессора
    $cur = (& $py -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>$null)
    if ($gpu -and ($cur -notmatch " True")) {
        Write-Host "    Встал PyTorch без видеокарты ($cur) — ставлю сборку $backend" -ForegroundColor Yellow
        & $uv pip install --python $py --system --break-system-packages --upgrade --torch-backend $backend --reinstall-package torch --reinstall-package torchaudio -r $req
        if ($LASTEXITCODE -ne 0 -and $backend -eq "cu130") {
            $backend = "cu126"
            & $uv pip install --python $py --system --break-system-packages --upgrade --torch-backend $backend --reinstall-package torch --reinstall-package torchaudio -r $req
        }
    }

    # --- проверка ---
    Write-Host "[4/4] Проверка…" -ForegroundColor Cyan
    $info = & $py -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch не запускается" }
    Write-Host "    PyTorch: $info"
    if ($gpu -and ($info -notmatch " True ")) {
        Fail ("PyTorch не видит видеокарту (драйвер {0:N2}). Обновите драйвер NVIDIA (Studio) и запустите «Golos — обновить компоненты»" -f $drv)
    }
    & $py -c "import PySide6, sounddevice, soundfile, qwen_tts, transformers; print('    Компоненты в порядке')"
    if ($LASTEXITCODE -ne 0) { Fail "не все компоненты импортируются" }
}
catch { Fail $_.Exception.Message }

Write-Host ""
Write-Host "Готово!" -ForegroundColor Green
if (-not $NoPause) { Start-Sleep -Seconds 3 }
