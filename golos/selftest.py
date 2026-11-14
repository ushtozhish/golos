"""Самопроверка интерфейса без моделей: python -m golos.selftest (используется в сборке на GitHub)."""
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
    s = QSettings("GolosApp", "Golos")
    s.setValue("root", str(tmp)); s.setValue("last_project", "")
    from golos.app import MainWindow
    from golos.project import Word, SR, DubPhrase, Variant
    from golos import bridge, exporter
    import soundfile as sf

    w = MainWindow()
    w.new_project()
    t = np.arange(SR * 8) / SR
    w.p.set_audio((np.sin(2 * np.pi * 220 * t) * 0.2).astype(np.float32))
    w.set_project(w.p)
    words = [("Клиент", 0.5, 0.9), ("хочет", 0.95, 1.3), ("Клиент", 2.0, 2.4), ("хочет", 2.45, 2.8),
             ("видеть", 2.85, 3.3), ("грейд.", 3.35, 3.9), ("GradeStream", 5.0, 5.8), ("бесплатно.", 5.9, 6.8)]
    w.p.set_words([Word(*x) for x in words])
    w.wave.view.refresh(); w.trans.rebuild(); w.refresh_dub_view(); w._update_state()
    w.resize(1300, 850); w.show(); app.processEvents()
    w.wave.view.grab().save(str(tmp / "wave.png"))
    w.grab().save(str(tmp / "main.png"))
    w.toggle_words([0, 1], []); w.undo(); w.redo()
    w.wave.view.sel = (4.0, 4.5); w.cut_selection(); w.undo()
    ph = w.p.phrases()
    assert len(ph) == 2, ph
    w.copy_request()
    ans = '```json\n{"golos":1,"translations":{"en":[{"id":1,"text":"The client wants to see the grade."},' \
          '{"id":2,"text":"GradeStream is free."}]}}\n```'
    w.apply_translation(ans, quiet=True)
    assert w.p.translations["en"][2] == "GradeStream is free."
    w.p.dub["en"] = []
    for x in ph:
        f = w.p.dub_dir("en") / f"p{x.id}.wav"
        sf.write(str(f), (np.random.randn(int(48000 * x.dur)) * 0.05).astype(np.float32), 48000)
        w.p.dub["en"].append(DubPhrase(id=x.id, text=w.p.translations["en"][x.id],
                                       variants=[Variant(file=f.name, dur=x.dur)], chosen=0))
    w.refresh_dub_view(); w.tabs.setCurrentIndex(1); app.processEvents()
    w.grab().save(str(tmp / "dub.png"))
    z = exporter.export_package(w.p, ["en"])
    assert z.exists()
    w.copy_file(z)
    print("SELFTEST OK", tmp)
    out = os.environ.get("GOLOS_SHOTS")
    if out:
        import shutil
        Path(out).mkdir(parents=True, exist_ok=True)
        for n in ("wave.png", "main.png", "dub.png"):
            shutil.copy(tmp / n, Path(out) / n)
    w.close()


if __name__ == "__main__":
    main()
