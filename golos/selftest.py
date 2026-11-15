"""Самопроверка интерфейса без моделей: python -m golos.selftest (запускается при сборке на GitHub)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings


def main():
    app = QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp())
    os.environ["APPDATA"] = str(tmp)
    s = QSettings("GolosApp", "Golos")
    s.setValue("root", str(tmp)); s.setValue("last_project", ""); s.setValue("open_last", "true")
    from golos.app import MainWindow
    from golos.project import Word, SR, DubPhrase, Variant
    from golos import exporter, voices
    import soundfile as sf

    w = MainWindow()
    w.new_project()
    t = np.arange(SR * 8) / SR
    tone = (np.sin(2 * np.pi * 220 * t) * 0.3 * ((t % 2) < 1.2)).astype(np.float32)
    w.p.set_audio(tone + np.random.randn(len(t)).astype(np.float32) * 0.001)
    w.set_project(w.p)
    w.resize(1300, 850); w.show(); app.processEvents()
    # монтаж
    w.on_seek(3.0); w.split(); assert len(w.p.clips) == 2
    w.on_seek(5.0); w.split(); assert len(w.p.clips) == 3
    w.tl.view.selected = {1}; w.delete_sel(); assert len(w.p.clips) == 2, w.p.clips
    w.undo(); assert len(w.p.clips) == 3
    w.redo(); assert len(w.p.clips) == 2
    w.tl.view.range = (0.5, 1.0); w.delete_sel()
    w.on_seek(0.2); w.trim("start")
    w.shorten_pauses(); w.undo()
    w.p.append_audio(tone[:SR]); w.edits_changed()
    w.tl.view.zoom_all(); app.processEvents()
    w.grab().save(str(tmp / "edit.png"))
    # распознанный текст (имитация)
    words = [("Клиент", 0.3, 0.7), ("хочет", 0.75, 1.1), ("видеть", 1.15, 1.5), ("грейд.", 1.55, 2.0),
             ("GradeStream", 2.6, 3.3), ("бесплатно.", 3.4, 4.2)]
    w.p.set_words([Word(*x) for x in words])
    assert w.p.words_fresh
    w.text.rebuild(); w.tl.view.update(); app.processEvents()
    ph = w.p.phrases(); assert len(ph) == 2, ph
    w.on_phrase_edited(1, "Клиент хочет видеть цвет.")
    assert w.p.phrases()[0].text == "Клиент хочет видеть цвет.", w.p.phrases()[0].text
    w.text.item(1, 2).setText("GradeStream — бесплатно.")   # правка прямо в таблице
    assert w.p.phrases()[1].text == "GradeStream — бесплатно.", w.p.phrases()[1].text
    n = w.p.apply_script("Клиент хочет видеть грейд, понимаете? GradeStream — бесплатно!")
    assert n == 2 and w.p.phrases()[1].text == "GradeStream — бесплатно!", [x.text for x in w.p.phrases()]
    w.text.rebuild(); w.tabs.setCurrentIndex(0); app.processEvents()
    w.grab().save(str(tmp / "edit.png"))
    ph = w.p.phrases()
    w.copy_request()
    ans = '```json\n{"golos":1,"translations":{"en":[{"id":1,"text":"The client wants to see the grade."},' \
          '{"id":2,"text":"GradeStream is free."}]}}\n```'
    w.auto_dub.setChecked(False)
    w.apply_translation(ans, quiet=True)
    assert w.p.translations["en"][2] == "GradeStream is free."
    # голос из «файла»
    vc = voices.save_voice("Тест", "file", tone[:SR * 6], SR, "тестовый образец")
    w.reload_voices(); assert w.voice_box.findData(vc.id) >= 0
    w.p.dub["en"] = []
    for x in ph:
        f = w.p.dub_dir("en") / f"p{x.id}.wav"
        sf.write(str(f), (np.random.randn(int(48000 * x.dur)) * 0.05).astype(np.float32), 48000)
        w.p.dub["en"].append(DubPhrase(id=x.id, text=w.p.translations["en"][x.id], voice=w.p.voice,
                                       variants=[Variant(file=f.name, dur=x.dur)], chosen=0))
    w.refresh_dub_view(); w.tabs.setCurrentIndex(1); app.processEvents()
    w.table.item(0, 1).setText("2.5")                      # своя длительность фразы 1
    assert abs(w.p.phrases()[0].dur - 2.5) < 1e-6, w.p.phrases()[0].dur
    app.processEvents()
    for i in range(30):
        w.log(f"проверка прокрутки {i}")
    sb = w.logbox.verticalScrollBar(); assert sb.value() == sb.maximum()
    # эталон и набор для дообучения
    f0 = w.p.dub_dir("en") / f"p{ph[0].id}.wav"
    w.make_etalon(f0, w.p.dub["en"][0], w.p.dub["en"][0].variants[0])
    assert w.p.voice.endswith("_etalon"), w.p.voice
    import shutil as _sh
    _sh.copy(f0, w.p.dub_dir("en") / f"p{ph[0].id:03d}_v9_123.wav")      # «потерянный» вариант
    assert w.p.recover_variants("en", ph[0].id) == 1
    mins = voices.add_to_dataset(f0, "Клиент хочет", "en", w.p.voice, "test_1")
    assert mins > 0
    # ударения
    from golos import stress
    assert stress.for_model("зам+ок", "ru", "caps") == "замОк"
    w._set_phrase_text(1, "Клиент хочет видеть гр+ейд.")
    assert stress.load_dict().get("грейд") == "гр+ейд"
    w.refresh_dub_view(); app.processEvents()
    w.grab().save(str(tmp / "dub.png"))
    z = exporter.export_package(w.p, ["en"], w.log)
    assert z.exists()
    w.copy_file(z)
    w.tabs.setCurrentIndex(2); app.processEvents()
    w.grab().save(str(tmp / "send.png"))
    # проект из SRT
    from golos import media
    srt = tmp / "promo.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500\nКлиент хочет видеть\n\n2\n00:00:02,600 --> 00:00:03,800\nгрейд.\n\n"
                   "3\n00:00:05,000 --> 00:00:07,000\nGradeStream бесплатно!\n", "utf-8")
    from PySide6.QtWidgets import QFileDialog
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(srt), ""))
    w.open_srt()
    ph = w.p.phrases()
    assert w.p.from_srt and len(ph) == 2 and ph[0].text == "Клиент хочет видеть грейд." and abs(ph[0].start - 0.97) < 0.05, \
        [(x.text, x.start) for x in ph]
    app.processEvents(); w.grab().save(str(tmp / "srt.png"))
    z2 = exporter.export_package(w.p, [], w.log)
    assert z2.exists()
    # ударение по записи / клику
    from golos.app import StressWordDialog
    sd = StressWordDialog(w, "замок", "Замок висит", 0)
    sd.pick(1); assert sd.result() == "зам+ок", sd.result()
    sr_ = 48000
    def _syl(dur, amp):
        tt = np.arange(int(sr_ * dur)) / sr_
        return (amp * np.sin(np.pi * tt / dur) ** 2 * np.sin(2 * np.pi * 160 * tt)).astype(np.float32)
    gap = np.zeros(int(sr_ * 0.05), np.float32)
    k, _ = stress.detect_from_audio(np.concatenate([gap, _syl(0.3, 0.6), gap, _syl(0.12, 0.3), gap]), sr_, "замок")
    assert k == 0, k
    sd.show(); app.processEvents(); sd.grab().save(str(tmp / "stress.png")); sd.close()
    # проекты: список, сохранить как, недавние
    from golos.app import ProjectsDialog, project_info
    w.new_project("Промо GradeStream")
    assert w.p.name == "Промо GradeStream"
    pd = ProjectsDialog(w)
    assert pd.table.rowCount() >= 3, pd.table.rowCount()
    pd.show(); app.processEvents(); pd.grab().save(str(tmp / "projects.png")); pd.close()
    assert len(w._recent()) >= 2
    print("SELFTEST OK", tmp)
    out = os.environ.get("GOLOS_SHOTS")
    if out:
        import shutil
        Path(out).mkdir(parents=True, exist_ok=True)
        for n in ("edit.png", "dub.png", "send.png", "srt.png", "projects.png", "stress.png"):
            shutil.copy(tmp / n, Path(out) / n)
    w.close()
    # модули синтеза должны импортироваться (ловим несовместимые версии torch/torchaudio)
    import importlib
    for m in ("torch", "torchaudio", "transformers", "librosa", "noisereduce", "pyloudnorm", "qwen_tts", "ruaccent"):
        mod = importlib.import_module(m)
        print("import ok:", m, getattr(mod, "__version__", ""))
    from golos import models  # noqa: F401
    from golos.models import _import_qwen
    print("Qwen3TTSModel:", _import_qwen().__name__)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
