# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import glob
import os
import sys
import time
import wave

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np

t0 = time.perf_counter()
from aivis_core import AivisCore, SynthParams

t1 = time.perf_counter()
print(f"import {t1-t0:.2f}s")
A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")
core = AivisCore(os.path.dirname(BERT), sorted(glob.glob(A + r"\Models\*.aivmx")), threads=int(os.environ.get("TH", "0")))
t2 = time.perf_counter()
print(f"init (bert) {t2-t1:.2f}s")
for v, s in core.styles.items():
    print(" ", v, s.display_name)
vid = [v for v, s in core.styles.items() if s.model_name == "Anneli"][0]
p = SynthParams(voice_id=vid)
core.get_model(core.styles[vid].model_uuid)
t3 = time.perf_counter()
print(f"model load {t3-t2:.2f}s")
core.synthesize("ウォームアップ。", p)
texts = sys.argv[1:] or [
    "こんにちは。",
    "本日は晴天なり。スクリーンリーダーで使える速度になったでしょうか。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
]
for i, text in enumerate(texts):
    t = time.perf_counter()
    first = None
    chunks = []
    for c in core.synthesize_stream(text, p):
        if first is None:
            first = time.perf_counter() - t
        chunks.append(c)
    total = time.perf_counter() - t
    wav = np.concatenate(chunks)
    dur = len(wav) / 44100
    print(f"[{i}] first={first*1000:.0f}ms total={total:.2f}s audio={dur:.2f}s RTF={total/dur:.2f} peak={np.abs(wav).max()} chunks={len(chunks)} text={text[:20]}")
    with wave.open(rf"C:\Users\yuki\Downloads\aivis\core_dev\work\out{i}.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(wav.tobytes())
