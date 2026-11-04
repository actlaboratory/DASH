# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""NVDA で読み上げられがちな短い発話を順に合成し、最初の音までの時間を計測する (毎回異なる文 = 異なる入力形状)。"""
import glob
import os
import sys
import time

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np
import onnxruntime as ort

import aivis_core.engine as E
from aivis_core import AivisCore, SynthParams

if os.environ.get("NOMEMPAT"):
    _orig = E.make_session_options

    def _patched(threads=0):
        so = _orig(threads)
        so.enable_mem_pattern = False
        return so

    E.make_session_options = _patched

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")
core = AivisCore(os.path.dirname(BERT), sorted(glob.glob(A + r"\Models\*.aivmx")), use_gpu=os.environ.get("GPU") == "1")
vid = [v for v, s in core.styles.items() if s.model_name == "Anneli"][0]
p = SynthParams(voice_id=vid)
core.synthesize("ウォームアップです。", p)
texts = [l for l in open(r"C:\Users\yuki\Downloads\aivis\core_dev\scripts\nvda_texts.txt", encoding="utf-8").read().splitlines() if l.strip()]
firsts, rtfs = [], []
for text in texts:
    t = time.perf_counter()
    first = None
    n = 0
    for c in core.synthesize_stream(text, p):
        if first is None:
            first = time.perf_counter() - t
        n += c.size
    total = time.perf_counter() - t
    firsts.append(first * 1000)
    rtfs.append(total / (n / 44100))
print(f"utterances={len(texts)} first-audio ms: median={np.median(firsts):.0f} p90={np.percentile(firsts, 90):.0f} max={np.max(firsts):.0f} min={np.min(firsts):.0f}  RTF median={np.median(rtfs):.2f}")
print("all:", " ".join(f"{x:.0f}" for x in firsts))
