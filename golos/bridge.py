"""Перевод через чат с Claude: текст запроса в буфер обмена и разбор ответа из буфера."""
from __future__ import annotations

import json
import re

MARK = "GOLOS-TRANSLATE v1"

LANG_NAMES_RU = {"en": "английский", "kk": "казахский", "ru": "русский", "de": "немецкий",
                 "fr": "французский", "es": "испанский", "it": "итальянский",
                 "pt": "португальский", "zh": "китайский", "ja": "японский", "ko": "корейский"}


def build_request(phrases, langs: list[str], source_lang: str = "ru", context: str = "") -> str:
    """phrases: список Phrase (id, text, dur)."""
    names = ", ".join(f"{l} ({LANG_NAMES_RU.get(l, l)})" for l in langs)
    data = {"source": source_lang,
            "phrases": [{"id": p.id, "dur": round(p.dur, 2), "text": p.text} for p in phrases]}
    lines = [
        f"[{MARK}]",
        f"Переведи текст диктора на: {names}.",
        "Правила:",
        "- Каждая фраза должна произноситься не дольше своей длительности dur (секунды) "
        "в спокойном темпе диктора. Если дословный перевод длиннее — сократи, сохранив смысл.",
        "- Живой разговорный язык рекламного ролика, без канцелярита.",
        "- Названия программ и терминов (GradeStream, DaVinci Resolve, Clean Feed, Waveform и т. п.) не переводи.",
        "- Числа пиши словами так, как их нужно произнести.",
        "- Количество и id фраз не меняй.",
    ]
    if context.strip():
        lines.append(f"- Контекст: {context.strip()}")
    lines += [
        "Ответь одним блоком ```json``` строго такого вида:",
        '{"golos": 1, "translations": {"<язык>": [{"id": 1, "text": "..."}]}}',
        "",
        "```json",
        json.dumps(data, ensure_ascii=False, indent=1),
        "```",
    ]
    return "\n".join(lines)


def looks_like_answer(text: str) -> bool:
    return bool(text) and '"translations"' in text and '"golos"' in text


def parse_answer(text: str) -> dict[str, dict[int, str]]:
    """Извлекает {lang: {id: text}} из ответа (с любым текстом вокруг блока)."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if not candidates:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            candidates = [text[a:b + 1]]
    for c in reversed(candidates):
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        tr = data.get("translations")
        if isinstance(tr, dict):
            out = {}
            for lang, items in tr.items():
                if isinstance(items, list):
                    out[lang] = {int(it["id"]): str(it["text"]).strip() for it in items
                                 if isinstance(it, dict) and "id" in it and "text" in it}
                elif isinstance(items, dict):
                    out[lang] = {int(k): str(v).strip() for k, v in items.items()}
            if out:
                return out
    raise ValueError("В буфере не найден блок перевода")
