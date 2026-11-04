# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""文の長さごとに 最初の音までの内訳 (g2p / BERT / front / 最初の dec) を計測する。"""
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np

from aivis_core.engine import AivisModel, make_session_options, default_providers
from aivis_core.nlp import bert_models
from aivis_core.nlp.japanese.g2p import g2p
from aivis_core.nlp.japanese.normalizer import normalize_text
from aivis_core import frontend
from aivis_core.winperf import set_process_power_throttling
_pt = os.environ.get("PT")
if _pt is not None: print("set throttling", _pt, set_process_power_throttling({"on": True, "off": False}.get(_pt)))

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT_DIR = Path(os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")))
gpu = os.environ.get("GPU") == "1"
th = int(os.environ.get("TH", "0"))
prov = default_providers(gpu)
bert_models.init(BERT_DIR, make_session_options(th), prov)
model = AivisModel(sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0], prov, th)

base = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。"
base = base.replace("。", "、")
frontend.text_to_features("ウォームアップ")
for n in [5, 10, 20, 40, 80]:
    text = base[:n]
    res = []
    for _ in range(int(os.environ.get("REP", "3"))):
        t0 = time.perf_counter()
        r = normalize_text(text, return_analysis_text=True)
        ph, to, w2p, sep, _, _ = g2p(r.text, analysis_text=r.analysis_text)
        t1 = time.perf_counter()
        f = frontend.text_to_features(text)  # g2p をもう一度含むので BERT 時間は差分で出す
        t2 = time.perf_counter()
        feeds = model._front_feeds(f, 0, model.style_vector(0, 1.0), 1.0, 0.2, 0.6, 0.8)
        z, g = model.front.run([model.z_name, model.g_name], feeds)
        t3 = time.perf_counter()
        e = min(z.shape[2], 16 + 12)
        model.dec.run(None, {model.z_name: z[:, :, :e], model.g_name: g})
        t4 = time.perf_counter()
        res.append((t1 - t0, (t2 - t1) - (t1 - t0), t3 - t2, t4 - t3))
    g2p_t, bert_t, front_t, dec_t = np.median(np.array(res), axis=0)
    print("   front runs:", " ".join(f"{r[2]*1000:.0f}" for r in res))
    print(f"chars={n:3d} phones={len(f.phones):4d} frames={z.shape[2]:4d}  g2p={g2p_t*1000:5.0f}ms bert={bert_t*1000:5.0f}ms front={front_t*1000:5.0f}ms dec1={dec_t*1000:5.0f}ms  first-audio={sum((g2p_t,bert_t,front_t,dec_t))*1000:5.0f}ms")
