import os
import sys

# Запуск без консоли (pythonw.exe): у процесса нет stdout/stderr, и любая запись туда
# (полоска загрузки моделей Hugging Face, предупреждения библиотек) падает с
# «'NoneType' object has no attribute 'write'». Направляем вывод в журнал golos.log.
if sys.stdout is None or sys.stderr is None:
    _log_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Golos")
    os.makedirs(_log_dir, exist_ok=True)
    _log = open(os.path.join(_log_dir, "golos.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

# Модели кэшируются в папке пользователя, а не в папке программы
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from golos.app import main
else:
    from .app import main

main()
