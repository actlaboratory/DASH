# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""デコーダの処理速度 (RTF) を設定ごとに比べる。

- スレッド数、グラフ最適化レベル、denormal を 0 扱い、P コアへのスレッド固定、区間長、DirectML
"""

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
from aivis_core import AivisCore, SynthParams  # noqa: E402
from aivis_core import frontend  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
core = AivisCore(bert, models)
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
st = core.styles[vid]
model = core.get_model(st.model_uuid)
text = "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"
f = frontend.text_to_features(text, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
z, g = model.run_front(f, st.speaker_local_id, st.style_local_id)
total = z.shape[2]
audio_sec = total * model.hop_length / model.sampling_rate
print(f"frames={total} audio={audio_sec:.2f}s")

m = onnx.load(str(model.path))
dec_bytes = Extractor(m).extract_model([model.z_name, model.g_name], [model.output_name]).SerializeToString()
del m


def make(threads, level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, denormal=False, pin=False, providers=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.graph_optimization_level = level
    if denormal:
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
    if pin:
        # P コアの物理コア (論理 0,2,4,..) に 1 スレッドずつ。スレッド 0 は呼び出し元なので n-1 個指定 (1 始まり)
        so.add_session_config_entry("session.intra_op_thread_affinities",
                                    ";".join(str(2 * i + 1) for i in range(1, threads)))
    return ort.InferenceSession(dec_bytes, so, providers=providers or ["CPUExecutionProvider"])


def run(sess, chunk, margin=12, reps=3):
    best = 1e9
    first = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        pos = 0
        n = chunk
        k = 0
        while pos < total:
            end = min(total, pos + n)
            if total - end < n // 2:
                end = total
            s, e = max(0, pos - margin), min(total, end + margin)
            sess.run(None, {model.z_name: z[:, :, s:e], model.g_name: g})
            if k == 0:
                first = min(first, time.perf_counter() - t0)
            k += 1
            pos = end
        best = min(best, time.perf_counter() - t0)
    return best / audio_sec, first


cases = [
    ("th8 ext (現行)", dict(threads=8)),
    ("th4", dict(threads=4)),
    ("th6", dict(threads=6)),
    ("th12", dict(threads=12)),
    ("th16", dict(threads=16)),
    ("th8 ALL", dict(threads=8, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)),
    ("th8 denormal0", dict(threads=8, denormal=True)),
    ("th8 pinP", dict(threads=8, pin=True)),
]
for name, kw in cases:
    s = make(**kw)
    run(s, 128, reps=1)
    out = [f"chunk{c}: RTF={run(s, c)[0]:.3f}" for c in (128, 256, 512)]
    print(f"{name:16s}", "  ".join(out), f" first16={run(s, 16)[1] * 1000:.0f}ms")
    del s

if "DmlExecutionProvider" in ort.get_available_providers():
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(dec_bytes, so, providers=[("DmlExecutionProvider", {"device_id": 0})])
    for c in (16, 128, 256, 512):
        run(s, c, reps=1)
    out = [f"chunk{c}: RTF={run(s, c)[0]:.3f}" for c in (128, 256, 512)]
    print(f"{'DML':16s}", "  ".join(out), f" first16={run(s, 16)[1] * 1000:.0f}ms")
