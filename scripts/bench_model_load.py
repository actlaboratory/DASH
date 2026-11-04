# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""声を切り替えたときのモデル読み込み (AivisModel) が、どの段階に時間を使っているかを測る。

  MODE=cpu|gpu  tools\\py313\\python.exe -I scripts\\bench_model_load.py [model.aivmx]
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE = os.environ.get("MODE", "cpu")
if MODE == "gpu":
    sys.path.insert(0, str(ROOT / "scripts"))
    import _cuda_loader  # noqa: E402

    sys.path.pop(0)
    _cuda_loader.load(str(ROOT / "tools" / "ort_cuda11"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx.utils import Extractor  # noqa: E402

from aivis_core.engine import AivisModel, default_providers, make_session_options  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    (ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data" / "models").glob("*.aivmx"))[0]
providers = default_providers(MODE == "gpu")
cpu = ["CPUExecutionProvider"]
print(f"mode={MODE} model={path.name} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)

T = {}
t = time.perf_counter()


def lap(name):
    global t
    now = time.perf_counter()
    T[name] = (now - t) * 1000
    t = now


m = onnx.load(str(path))
lap("onnx.load")
z = g = None
for node in m.graph.node:
    if node.name == "/dec/conv_pre/Conv":
        z = node.input[0]
    elif node.name == "/dec/cond/Conv":
        g = node.input[0]
inputs = [i.name for i in m.graph.input]
out = m.graph.output[0].name
lap("find boundary")
ex = Extractor(m)
lap("Extractor()")
front = ex.extract_model(inputs, [z, g])
lap("extract front")
dec = ex.extract_model([z, g], [out])
lap("extract dec")
fb, db = front.SerializeToString(), dec.SerializeToString()
lap("serialize")
so = make_session_options(0)
fs = ort.InferenceSession(fb, so, providers=cpu)
lap("session front (CPU)")
ds = ort.InferenceSession(db, so, providers=providers)
lap(f"session dec ({MODE})")
zi, gi = ds.get_inputs()
ds.run(None, {zi.name: np.zeros((1, 192, 30), np.float32), gi.name: np.zeros((1, 512, 1), np.float32)})
lap("first dec run")
for k, v in T.items():
    print(f"  {k:22s} {v:7.0f} ms")
print(f"  {'total':22s} {sum(T.values()):7.0f} ms", flush=True)

# 比較: 分割済みのグラフをファイルに保存しておき、そこから直接セッションを作る場合
work = ROOT / "work" / "split_cache"
work.mkdir(parents=True, exist_ok=True)
(work / "front.onnx").write_bytes(fb)
(work / "dec.onnx").write_bytes(db)
del m, ex, front, dec, fb, db, fs, ds
t0 = time.perf_counter()
fs = ort.InferenceSession(str(work / "front.onnx"), so, providers=cpu)
t1 = time.perf_counter()
ds = ort.InferenceSession(str(work / "dec.onnx"), so, providers=providers)
t2 = time.perf_counter()
print(f"  cached: session front {(t1 - t0) * 1000:.0f} ms, session dec {(t2 - t1) * 1000:.0f} ms, "
      f"total {(t2 - t0) * 1000:.0f} ms", flush=True)

# 現行の AivisModel 全体
t0 = time.perf_counter()
AivisModel(path, providers, 0, cpu)
print(f"  AivisModel() now: {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
