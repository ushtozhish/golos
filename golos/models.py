"""Загрузка моделей на видеокарту: Whisper (распознавание) и Qwen3-TTS / AIT-Syn (синтез).

В памяти держим не больше одной модели синтеза — 8 ГБ видеопамяти хватает
на Whisper (~1,6 ГБ) + одну TTS-модель 1.7B в fp16 (~4 ГБ).
"""
from __future__ import annotations

import gc
import inspect
from typing import Callable

import numpy as np

from . import dsp

WHISPER_ID = "openai/whisper-large-v3-turbo"
TTS_BASE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TTS_KK_ID = "nur-dev/ait-syn"
TTS_DESIGN_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TTS_PRESET_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PRESET_SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]

# Языки синтеза: код → (модель, имя языка для модели, подпись)
TTS_LANGS = {
    "ru": (TTS_BASE_ID, "russian", "Русский"),
    "en": (TTS_BASE_ID, "english", "English"),
    "kk": (TTS_KK_ID, "kazakh", "Қазақша"),
    "de": (TTS_BASE_ID, "german", "Deutsch"),
    "fr": (TTS_BASE_ID, "french", "Français"),
    "es": (TTS_BASE_ID, "spanish", "Español"),
    "it": (TTS_BASE_ID, "italian", "Italiano"),
    "pt": (TTS_BASE_ID, "portuguese", "Português"),
    "zh": (TTS_BASE_ID, "chinese", "中文"),
    "ja": (TTS_BASE_ID, "japanese", "日本語"),
    "ko": (TTS_BASE_ID, "korean", "한국어"),
}
WHISPER_LANG = {"ru": "russian", "en": "english", "kk": "kazakh", "de": "german", "fr": "french",
                "es": "spanish", "it": "italian", "pt": "portuguese", "zh": "chinese",
                "ja": "japanese", "ko": "korean"}

Log = Callable[[str], None]


def torch_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name}, {p.total_memory / 2**30:.0f} GB, CUDA {torch.version.cuda}, torch {torch.__version__}"
        return f"CPU (CUDA недоступна), torch {torch.__version__}"
    except Exception as e:  # noqa: BLE001
        return f"torch: {e}"


_SETUP_DONE = False


def setup_torch(log: Log | None = None) -> str:
    """Один раз: потолок видеопамяти (чтобы не уходить в общую память и не вешать экран)
    и ограничение потоков процессора (чтобы мышь и окна оставались живыми)."""
    global _SETUP_DONE
    import os
    import torch
    if not _SETUP_DONE:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.85, 0)
            except Exception:  # noqa: BLE001
                pass
        _SETUP_DONE = True
    if torch.cuda.is_available():
        dev = f"видеокарта {torch.cuda.get_device_name(0)}"
    else:
        dev = "ПРОЦЕССОР (видеокарта не используется — будет медленно)"
    if log:
        log(f"Вычисления: {dev}")
    return dev


def vram_used() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ"
    except Exception:  # noqa: BLE001
        pass
    return "-"


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def models_dir():
    """Папка моделей программы: %LOCALAPPDATA%\\Golos\\models (обычные файлы, без символических ссылок —
    Windows без режима разработчика не даёт их создавать, ошибка WinError 1314)."""
    import os
    from pathlib import Path
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Golos" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _no_symlinks():
    """Страховка: если какая-то модель докачивает файлы в общий кэш Hugging Face
    (например, вспомогательный токенизатор), кэш тоже должен обходиться без символических ссылок."""
    try:
        import huggingface_hub.file_download as fd
        fd.are_symlinks_supported = lambda *a, **k: False
    except Exception:  # noqa: BLE001
        pass


_no_symlinks()


def ensure_downloaded(repo_id: str, log: Log) -> str:
    """Скачивает модель с Hugging Face (при первом запуске) и возвращает путь к её папке."""
    from huggingface_hub import snapshot_download
    target = models_dir() / repo_id.replace("/", "--")
    done = target / ".golos_complete"
    if done.exists():
        return str(target)
    log(f"Модель {repo_id}: скачиваю (один раз)…")
    snapshot_download(repo_id, local_dir=str(target))
    done.write_text("ok")
    log(f"Модель {repo_id}: готова")
    return str(target)


# ---------------- Whisper ----------------
class Whisper:
    def __init__(self, log: Log):
        import time
        import torch
        from transformers import pipeline
        self.log = log
        path = ensure_downloaded(WHISPER_ID, log)
        setup_torch(log)
        t0 = time.time()
        cuda = torch.cuda.is_available()
        dt = torch.float16 if cuda else torch.float32
        dev = "cuda:0" if cuda else "cpu"
        try:
            self.pipe = pipeline("automatic-speech-recognition", model=path, dtype=dt, device=dev)
        except TypeError:
            self.pipe = pipeline("automatic-speech-recognition", model=path, torch_dtype=dt, device=dev)
        log(f"Whisper загружен за {time.time() - t0:.1f} с")

    def words(self, audio: np.ndarray, sr: int, lang: str) -> list[tuple[str, float, float]]:
        import time
        import torch
        x16 = dsp.resample(audio, sr, 16000)
        t0 = time.time()
        # batch_size=1: расчёт тайминга слов хранит матрицы внимания — пачкой они не влезают в 8 ГБ
        with torch.inference_mode():
            out = self.pipe(
                {"raw": x16, "sampling_rate": 16000},
                return_timestamps="word",
                chunk_length_s=30,
                batch_size=1,
                generate_kwargs={"language": WHISPER_LANG.get(lang, lang), "task": "transcribe"},
            )
        self.log(f"Распознано {len(x16) / 16000:.0f} с звука за {time.time() - t0:.1f} с, видеопамять: {vram_used()}")
        dur = len(x16) / 16000
        res = []
        for ch in out.get("chunks", []):
            s, e = ch["timestamp"]
            if s is None:
                continue
            e = e if e is not None else min(dur, s + 0.4)
            res.append((ch["text"], float(s), float(e)))
        return res

    def text(self, audio: np.ndarray, sr: int, lang: str) -> str:
        x16 = dsp.resample(audio, sr, 16000)
        import torch
        with torch.inference_mode():
            out = self.pipe({"raw": x16, "sampling_rate": 16000},
                        generate_kwargs={"language": WHISPER_LANG.get(lang, lang), "task": "transcribe"})
        return out["text"].strip()

    def close(self):
        del self.pipe
        free_gpu()


# ---------------- Qwen3-TTS ----------------
def _import_qwen():
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    return Qwen3TTSModel


class TTS:
    """Обёртка над qwen-tts. RTX 20xx не поддерживает bf16 и flash-attn,
    поэтому fp16 + штатное внимание PyTorch (sdpa), при сбое — fp32/eager."""

    def __init__(self, repo_id: str, log: Log, precision: str = "fp16"):
        import torch
        self.repo_id = repo_id
        self.log = log
        path = ensure_downloaded(repo_id, log)
        setup_torch(log)
        Model = _import_qwen()
        cuda = torch.cuda.is_available()
        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[precision]
        if not cuda:
            dtype = torch.float32
        self.precision = precision if cuda else "fp32"
        last = None
        for attn in ("sdpa", "eager"):
            try:
                self.model = Model.from_pretrained(path, device_map="cuda:0" if cuda else "cpu",
                                                   dtype=dtype, attn_implementation=attn)
                log(f"{repo_id}: {self.precision}, attention={attn}")
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"Не удалось загрузить {repo_id}: {last}")
        try:
            self.model.model.eval()
        except Exception:  # noqa: BLE001
            pass
        self._prompt = None
        self._prompt_key = None
        self._batch_ok = None
        sig = inspect.signature(self.model.generate_voice_clone)
        self._accepts = set(sig.parameters)
        self._var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    def set_reference(self, ref_audio_24k: np.ndarray, ref_text: str):
        key = (len(ref_audio_24k), ref_text)
        if key == self._prompt_key:
            return
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=(ref_audio_24k.astype(np.float32), 24000), ref_text=ref_text,
            x_vector_only_mode=False)
        self._prompt_key = key

    def _kw(self, **kw):
        if self._var_kw:
            return kw
        return {k: v for k, v in kw.items() if k in self._accepts}

    def _call(self, text, language, temperature, max_seconds):
        last = None
        for lang in (language, language.capitalize()):
            lg = [lang] * len(text) if isinstance(text, list) else lang
            kw = self._kw(text=text, language=lg, voice_clone_prompt=self._prompt,
                          non_streaming_mode=True, do_sample=True, temperature=temperature,
                          top_k=50, top_p=1.0, repetition_penalty=1.05,
                          max_new_tokens=int(max_seconds * 12.5) + 24)
            try:
                import torch
                with torch.inference_mode():
                    return self.model.generate_voice_clone(**kw)
            except (ValueError, KeyError, AssertionError) as e:
                last = e
        raise last

    def generate(self, texts: list[str], language: str, temperature: float = 0.8,
                 max_seconds: float = 30.0) -> tuple[list[np.ndarray], int]:
        try:
            if self._batch_ok is False:
                raise NotImplementedError
            wavs, sr = self._call(list(texts), language, temperature, max_seconds)
            wavs = list(wavs)
            if len(wavs) != len(texts):
                raise ValueError("batch size mismatch")
        except FloatingPointError:
            raise
        except Exception as e:  # noqa: BLE001 — пакетный режим не поддержан: по одной
            if self._batch_ok is None:
                self.log(f"пакетный синтез недоступен ({type(e).__name__}), генерирую по одной")
            self._batch_ok = False
            wavs, sr = [], 24000
            for t in texts:
                w, sr = self._call(t, language, temperature, max_seconds)
                wavs.append(w[0] if isinstance(w, (list, tuple)) else w)
        out = []
        for w in wavs:
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(w)) or float(np.max(np.abs(w), initial=0)) < 1e-4:
                raise FloatingPointError("NaN или тишина в синтезе")
            out.append(w)
        return out, int(sr)

    # --- создание голосов (модели VoiceDesign / CustomVoice) ---
    def design(self, text: str, language: str, instruct: str):
        for lang in (language, language.capitalize()):
            try:
                wavs, sr = self.model.generate_voice_design(text=text, language=lang, instruct=instruct)
                return np.asarray(wavs[0], np.float32).reshape(-1), int(sr)
            except (ValueError, KeyError, AssertionError) as e:
                last = e
        raise last

    def preset(self, text: str, language: str, speaker: str, instruct: str = ""):
        for lang in (language, language.capitalize()):
            try:
                wavs, sr = self.model.generate_custom_voice(text=text, language=lang, speaker=speaker,
                                                            instruct=instruct or None)
                return np.asarray(wavs[0], np.float32).reshape(-1), int(sr)
            except (ValueError, KeyError, AssertionError) as e:
                last = e
        raise last

    def speakers(self) -> list[str]:
        try:
            sp = self.model.get_supported_speakers()
            return list(sp.keys()) if isinstance(sp, dict) else list(sp)
        except Exception:  # noqa: BLE001
            return PRESET_SPEAKERS

    def close(self):
        del self.model
        self._prompt = None
        free_gpu()
