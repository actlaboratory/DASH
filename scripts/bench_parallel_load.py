# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA のとき、前段 (CPU) とデコーダ (GPU) のセッション作成 + 最初のデコードを並行にすると速くなるかを測る。

  tools\\py313\\python.exe -I scripts\\bench_parallel_load.py
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _cuda_loader  # noqa: E402

sys.path.pop(0)
_cuda_loader.load(str(ROOT / "tools" / "ort_cuda11"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx.utils import Extractor  # noqa: E402

from aivis_core.engine import default_providers, make_session_options  # noqa: E402

path = sorted((ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data" / "models").glob("*.aivmx"))[0]
gpu = default_providers(True)
cpu = ["CPUExecutionProvider"]

# CUDA の初期化 (NVDA では起動時に BERT を作るときに済んでいる) を先に済ませる
_w = ort.InferenceSession(onnx.helper.make_model(onnx.helper.make_graph(
    [onnx.helper.make_node("Relu", ["x"], ["y"])], "g",
    [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
    [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])]),
    opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8).SerializeToString(), providers=gpu)
_w.run(None, {"x": np.zeros(1, np.float32)})

m = onnx.load(str(path))
z = g = None
for node in m.graph.node:
    if node.name == "/dec/conv_pre/Conv":
        z = node.input[0]
    elif node.name == "/dec/cond/Conv":
        g = node.input[0]
ex = Extractor(m)
fb = ex.extract_model([i.name for i in m.graph.input], [z, g]).SerializeToString()
db = ex.extract_model([z, g], [m.graph.output[0].name]).SerializeToString()
del ex, m


def make_front():
    return ort.InferenceSession(fb, make_session_options(0), providers=cpu)


def make_dec():
    s = ort.InferenceSession(db, make_session_options(0), providers=gpu)
    zi, gi = s.get_inputs()
    s.run(None, {zi.name: np.zeros((1, 192, 30), np.float32), gi.name: np.zeros((1, 512, 1), np.float32)})
    return s


for rep in range(3):
    t = time.perf_counter()
    make_front()
    a = time.perf_counter()
    make_dec()
    b = time.perf_counter()
    print(f"sequential: front {(a - t) * 1000:.0f} + dec(+first run) {(b - a) * 1000:.0f} = {(b - t) * 1000:.0f} ms",
          flush=True)
    t = time.perf_counter()
    th = threading.Thread(target=make_dec)
    th.start()
    make_front()
    th.join()
    print(f"parallel:   {(time.perf_counter() - t) * 1000:.0f} ms", flush=True)
