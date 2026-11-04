# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import os, sys, time, numpy as np, onnxruntime as ort
path = "C:/Users/yuki/Downloads/aivis/core_dev/work/dec.onnx"
rng = np.random.default_rng(0); F=242
z = rng.standard_normal((1,192,F)).astype(np.float32); g = rng.standard_normal((1,512,1)).astype(np.float32)*0.1
def mk(th, spin):
    so = ort.SessionOptions(); so.intra_op_num_threads = th
    so.add_session_config_entry("session.intra_op.allow_spinning", spin)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
def bench(s, label):
    feed={"/Mul_9_output_0": z, "/Unsqueeze_output_0": g}; s.run(None, feed)
    ts=[]
    for _ in range(4):
        t=time.perf_counter(); s.run(None, feed); ts.append(time.perf_counter()-t)
    print(label, " ".join(f"{x:.3f}" for x in ts))
for th in [4,6,8]:
    for spin in ["1","0"]:
        bench(mk(th,spin), f"th={th} spin={spin}")
extra = [mk(8,"1") for _ in range(2)]
bench(mk(8,"1"), "th=8 spin=1 with 2 extra sessions")
bench(mk(8,"0"), "th=8 spin=0 with 2 extra sessions")
