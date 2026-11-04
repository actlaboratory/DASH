# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""前段 (front) のセッション作成が遅い件: 最適化の段階と、最適化済みグラフを保存して読み込む場合を比べる (CPU)。
bench_model_load.py が work/split_cache/front.onnx を作っておくこと。"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aivis_core.engine import AivisCore, make_session_options  # noqa: E402
from aivis_core.frontend import text_to_features  # noqa: E402

work = ROOT / "work" / "split_cache"
front = work / "front.onnx"
data = ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data"
core = AivisCore(data / "bert", sorted((data / "models").glob("*.aivmx")), use_gpu=False)
st = next(iter(core.styles.values()))
model = core.get_model(st.model_uuid)
feats = [text_to_features(t, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
         for t in ["ファイル", "今日はいい天気ですね。散歩に行きましょう。"]]
L = ort.GraphOptimizationLevel


def feeds_for(sess, f):
    # model.run_front と同じ入力を作るため、AivisModel の front を一時的に差し替えて呼ぶ
    return f


def bench(sess):
    saved = model.front
    model.front = sess
    try:
        outs, times = [], []
        for f in feats:
            model.run_front(f, st.speaker_local_id, st.style_local_id)
            xs = []
            for _ in range(5):
                t = time.perf_counter()
                o = model.run_front(f, st.speaker_local_id, st.style_local_id)
                xs.append((time.perf_counter() - t) * 1000)
            times.append(statistics.median(xs))
            outs.append(o[0])
        return times, outs
    finally:
        model.front = saved


ref = None
for name, level, src, save in [
    ("ALL (now)", L.ORT_ENABLE_ALL, front, work / "front_opt_all.onnx"),
    ("EXTENDED", L.ORT_ENABLE_EXTENDED, front, work / "front_opt_ext.onnx"),
    ("BASIC", L.ORT_ENABLE_BASIC, front, work / "front_opt_basic.onnx"),
    ("DISABLE_ALL", L.ORT_DISABLE_ALL, front, None),
]:
    for rep in range(2):
        so = make_session_options(0)
        so.graph_optimization_level = level
        if save is not None and rep == 0:
            so.optimized_model_filepath = str(save)
        t = time.perf_counter()
        s = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        ct = (time.perf_counter() - t) * 1000
    times, outs = bench(s)
    ref = ref or outs
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, outs))
    print(f"{name:12s} create {ct:6.0f} ms  front run {[round(x) for x in times]} ms  max diff {diff:.2e}", flush=True)

for name, path in [("saved ALL", work / "front_opt_all.onnx"), ("saved EXT", work / "front_opt_ext.onnx"),
                   ("saved BASIC", work / "front_opt_basic.onnx")]:
    if not path.exists():
        continue
    for rep in range(2):
        so = make_session_options(0)
        so.graph_optimization_level = L.ORT_DISABLE_ALL
        t = time.perf_counter()
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        ct = (time.perf_counter() - t) * 1000
    times, outs = bench(s)
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, outs))
    print(f"{name:12s} create {ct:6.0f} ms  front run {[round(x) for x in times]} ms  max diff {diff:.2e}  "
          f"size {path.stat().st_size / 2**20:.0f} MB", flush=True)
