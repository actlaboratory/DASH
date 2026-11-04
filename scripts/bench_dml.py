# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""DirectML で各段 (BERT / 前段 / デコーダ) を動かしたときの時間。初めて見る入力の形で遅くならないかも見る。"""

import glob
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
sys.path.insert(0, ROOT)
from aivis_core import AivisCore  # noqa: E402
from aivis_core import frontend  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
core = AivisCore(bert, models)
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
st = core.styles[vid]
model = core.get_model(st.model_uuid)

texts = [l for l in open(os.path.join(ROOT, "scripts", "nvda_texts.txt"), encoding="utf-8").read().splitlines() if l.strip()]
feats = [frontend.text_to_features(t, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra) for t in texts[:20]]
feats = [f for f in feats if f is not None]

m = onnx.load(str(model.path))
ex = Extractor(m)
front_b = ex.extract_model(model.input_names, [model.z_name, model.g_name]).SerializeToString()
dec_b = ex.extract_model([model.z_name, model.g_name], [model.output_name]).SerializeToString()
del ex, m

so = ort.SessionOptions()
so.enable_mem_pattern = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
DML = [("DmlExecutionProvider", {"device_id": 0})]
t = time.perf_counter()
dec = ort.InferenceSession(dec_b, so, providers=DML)
front = ort.InferenceSession(front_b, so, providers=DML)
print(f"DML sessions created in {time.perf_counter() - t:.1f}s")

sv = model.style_vector(st.style_local_id, 1.0)
cpu_front_ms, dml_front_ms, zs = [], [], []
for f in feats:
    feeds = model._front_feeds(f, st.speaker_local_id, sv, 1.0, 0.2, 0.6, 0.8)
    t = time.perf_counter()
    z, g = model.front.run([model.z_name, model.g_name], feeds)
    cpu_front_ms.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    front.run([model.z_name, model.g_name], feeds)
    dml_front_ms.append((time.perf_counter() - t) * 1000)
    zs.append((z, g))
print("front CPU ms:", [round(x) for x in cpu_front_ms])
print("front DML ms:", [round(x) for x in dml_front_ms], "(初めての形ばかり)")

# デコーダ: 実際の区間の切り方 (16→32→64→128, のりしろ 12) で、初めての形を含めて測る
hop, sr = model.hop_length, model.sampling_rate
for name, sess in (("CPU", model.dec), ("DML", dec)):
    calls = []
    tot_dec = tot_audio = 0.0
    for z, g in zs:
        total = z.shape[2]
        pos, n = 0, 16
        while pos < total:
            end = min(total, pos + n)
            if total - end < n // 2:
                end = total
            s, e = max(0, pos - 12), min(total, end + 12)
            t = time.perf_counter()
            sess.run(None, {model.z_name: z[:, :, s:e], model.g_name: g})
            dt = time.perf_counter() - t
            calls.append((e - s, dt * 1000))
            tot_dec += dt
            pos = end
            n = min(128, n * 2)
        tot_audio += total * hop / sr
    firsts = [round(ms) for fr, ms in calls if fr == 28]
    print(f"dec {name}: RTF={tot_dec / tot_audio:.3f}  first-chunk(28fr) ms={firsts[:10]}  "
          f"max call={max(ms for _, ms in calls):.0f}ms")

# 区間の長さごとの時間 (2 周目)、初めての長さの 1 回目 / 2 回目
seen = {}
for z, g in zs:
    total = z.shape[2]
    pos, n = 0, 16
    while pos < total:
        end = min(total, pos + n)
        if total - end < n // 2:
            end = total
        s, e = max(0, pos - 12), min(total, end + 12)
        t = time.perf_counter()
        dec.run(None, {model.z_name: z[:, :, s:e], model.g_name: g})
        seen.setdefault(e - s, []).append(round((time.perf_counter() - t) * 1000))
        pos = end
        n = min(128, n * 2)
print("2nd pass DML per length:", {k: v[:4] for k, v in sorted(seen.items())})
zl = np.concatenate([z for z, _ in zs], axis=2)
fresh = []
for L in (301, 333, 377, 401, 155, 157, 600, 601):
    t = time.perf_counter()
    dec.run(None, {model.z_name: zl[:, :, :L], model.g_name: zs[0][1]})
    t1 = time.perf_counter()
    dec.run(None, {model.z_name: zl[:, :, :L], model.g_name: zs[0][1]})
    fresh.append((L, round((t1 - t) * 1000), round((time.perf_counter() - t1) * 1000)))
print("fresh length (L, first ms, second ms):", fresh)
