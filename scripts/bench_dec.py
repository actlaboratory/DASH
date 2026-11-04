# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import sys, time, numpy as np, onnxruntime as ort
path = "C:/Users/yuki/Downloads/aivis/core_dev/work/dec.onnx"
rng = np.random.default_rng(0)
F = int(sys.argv[1]) if len(sys.argv)>1 else 242
z = rng.standard_normal((1,192,F)).astype(np.float32); g = rng.standard_normal((1,512,1)).astype(np.float32)*0.1
for th in [0, 4, 8, 16]:
  for opt in [ort.GraphOptimizationLevel.ORT_ENABLE_ALL]:
    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.graph_optimization_level = opt
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    s.run(None, {"/Mul_9_output_0": z, "/Unsqueeze_output_0": g})
    t=time.time(); n=3
    for _ in range(n): s.run(None, {"/Mul_9_output_0": z, "/Unsqueeze_output_0": g})
    dt=(time.time()-t)/n; print(f"threads={th} frames={F} audio={F*512/44100:.2f}s time={dt:.3f}s RTF={dt/(F*512/44100):.2f}")
so = ort.SessionOptions()
s = ort.InferenceSession(path, so, providers=[("DmlExecutionProvider",{"device_id":0}),"CPUExecutionProvider"])
s.run(None, {"/Mul_9_output_0": z, "/Unsqueeze_output_0": g})
t=time.time(); s.run(None, {"/Mul_9_output_0": z, "/Unsqueeze_output_0": g}); dt=time.time()-t
print(f"DML time={dt:.3f}s RTF={dt/(F*512/44100):.2f}")
