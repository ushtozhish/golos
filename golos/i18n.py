"""Простая локализация интерфейса RU/EN: T("по-русски", "in English")."""
LANG = "ru"


def set_lang(lang: str):
    global LANG
    LANG = "en" if lang == "en" else "ru"


def T(ru: str, en: str) -> str:
    return en if LANG == "en" else ru
