# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

from aivis_core import frontend
from aivis_core.engine import AivisModel, make_session_options
from aivis_core.nlp import bert_models

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT_DIR = Path(os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")))
bert_models.init(BERT_DIR, make_session_options(0))
path = sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0]
m = onnx.load(path)
inputs = [i.name for i in m.graph.input]
Z, G = "/Mul_9_output_0", "/Unsqueeze_output_0"
front = Extractor(m).extract_model(inputs, [Z, G]).SerializeToString()
del m
text = "吾輩は猫である、名前はまだ無い、どこで生れたかとんと見当がつかぬ、何でも薄暗いじめじめした所で"
f = frontend.text_to_features(text)
import numpy as np

sv = np.zeros(256, np.float32)


class Dummy:
    input_names = inputs
    is_jp_extra = True


feeds = AivisModel._front_feeds(Dummy, f, 0, sv, 1.0, 0.2, 0.6, 0.8)


def run(label, so, prov):
    s = ort.InferenceSession(front, so, providers=prov)
    s.run([Z, G], feeds)
    ts = []
    for _ in range(5):
        if os.environ.get("WITH_BERT"):
            frontend.text_to_features(text)
        t = time.perf_counter()
        s.run([Z, G], feeds)
        ts.append(time.perf_counter() - t)
    print(f"{label:45s} median={np.median(ts)*1000:6.0f}ms  " + " ".join(f"{x*1000:.0f}" for x in ts))
    del s


cpu_plain = ["CPUExecutionProvider"]
cpu_same = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
run("ours, default arena", make_session_options(0), cpu_plain)
so = make_session_options(0); so.enable_cpu_mem_arena = False
run("no arena", so, cpu_plain)
so = make_session_options(0); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
run("opt disabled, default arena", so, cpu_plain)
so = make_session_options(0); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
run("opt extended, default arena", so, cpu_plain)
so = ort.SessionOptions()
run("ORT defaults (all threads, spinning)", so, cpu_plain)
