# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import os, sys, time, numpy as np, onnx, onnxruntime as ort
from onnx.utils import Extractor
src = sys.argv[1]
t=time.time(); m = onnx.load(src); print("load", time.time()-t)
Z, G = "/Mul_9_output_0", "/Unsqueeze_output_0"
inputs = [i.name for i in m.graph.input]
ex = Extractor(m)
t=time.time()
front = ex.extract_model(inputs, [Z, G]); dec = ex.extract_model([Z, G], ["output"])
print("extract", time.time()-t, "front nodes", len(front.graph.node), "dec nodes", len(dec.graph.node))
print("dec size MB", dec.ByteSize()/1e6, "front MB", front.ByteSize()/1e6)
so = ort.SessionOptions(); so.intra_op_num_threads = int(__import__("os").environ.get("TH","8")); so.add_session_config_entry("session.intra_op.allow_spinning","0"); os.environ.get("DAZ") and so.add_session_config_entry("session.set_denormal_as_zero","1")
full_s = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
front_s = ort.InferenceSession(front.SerializeToString(), so, providers=["CPUExecutionProvider"])
dec_s = ort.InferenceSession(dec.SerializeToString(), so, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0); T = 80
feed = dict(x_tst=rng.integers(1,100,(1,T)).astype(np.int64), x_tst_lengths=np.array([T],np.int64),
  sid=np.array([0],np.int64), tones=rng.integers(0,2,(1,T)).astype(np.int64), language=np.ones((1,T),np.int64),
  bert=rng.standard_normal((1,1024,T)).astype(np.float32), style_vec=rng.standard_normal((1,256)).astype(np.float32)*0.1,
  length_scale=np.array(1.0,np.float32), sdp_ratio=np.array(0.0,np.float32), noise_scale=np.array(0.0,np.float32), noise_scale_w=np.array(0.0,np.float32))
for k in range(2):
    t=time.time(); full = full_s.run(["output"], feed)[0]; tf=time.time()-t
    t=time.time(); z,g = front_s.run([Z,G], feed); t1=time.time()-t
    t=time.time(); y = dec_s.run(["output"], {Z:z, G:g})[0]; t2=time.time()-t
    print(f"full {tf:.3f}s  front {t1:.3f}s  dec {t2:.3f}s  z{z.shape} g{g.shape} wav{full.shape} maxdiff {np.abs(full-y).max():.2e}")
onnx.save(dec, "C:/Users/yuki/Downloads/aivis/core_dev/work/dec.onnx") if len(sys.argv)>2 else None
