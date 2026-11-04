# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""GPU が休止したあとの 1 文目だけ CPU で BERT を計算する案のために、int8 に量子化した BERT の
メモリ・速さ・特徴量のずれを fp32 版と比べる (CPU)。量子化したモデルは work/ に作る。"""

import ctypes
import gc
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aivis_core.engine import make_session_options  # noqa: E402

FP32 = ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data" / "bert" / "model.onnx"
INT8 = ROOT / "work" / "bert_int8.onnx"


def rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("a", wintypes.DWORD), ("p", ctypes.c_size_t), ("w", ctypes.c_size_t)] + [
            (f"x{i}", ctypes.c_size_t) for i in range(6)]
    k, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.w / 2**20


if not INT8.exists():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t = time.perf_counter()
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    print(f"quantized in {time.perf_counter() - t:.0f}s: {INT8.stat().st_size / 2**20:.0f} MB")

rng = np.random.default_rng(0)
inputs = [rng.integers(5, 6000, size=(1, n)).astype(np.int64) for n in (6, 10, 18, 30)]
results = {}
for name, path in (("fp32", FP32), ("int8", INT8)):
    gc.collect()
    r0 = rss_mb()
    s = ort.InferenceSession(str(path), make_session_options(0), providers=["CPUExecutionProvider"])
    mem = rss_mb() - r0
    names = [i.name for i in s.get_inputs()]
    outs, times = [], []
    for ids in inputs:
        feeds = {names[0]: ids, names[1]: np.ones_like(ids)}
        s.run(None, feeds)
        t = time.perf_counter()
        for _ in range(5):
            o = s.run(None, feeds)[0]
        times.append((time.perf_counter() - t) / 5 * 1000)
        outs.append(o)
    results[name] = outs
    print(f"{name}: memory +{mem:.0f} MB, ms per call (6/10/18/30 tokens) = {[round(x) for x in times]}")
    del s
for i, (a, b) in enumerate(zip(results["fp32"], results["int8"])):
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cos = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    print(f"  {inputs[i].shape[1]} tokens: cos mean {cos.mean():.4f} min {cos.min():.4f}")
