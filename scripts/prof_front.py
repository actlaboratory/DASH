# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import collections
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import onnxruntime as ort

from aivis_core import frontend
from aivis_core.engine import AivisModel, make_session_options
from aivis_core.nlp import bert_models

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT_DIR = Path(os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")))
bert_models.init(BERT_DIR, make_session_options(0))
model = AivisModel(sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0])
import onnx
from onnx.utils import Extractor

m = onnx.load(str(model.path))
front = Extractor(m).extract_model(model.input_names, [model.z_name, model.g_name])
so = make_session_options(0)
so.enable_profiling = True
so.profile_file_prefix = r"C:\Users\yuki\Downloads\aivis\core_dev\work\prof"
s = ort.InferenceSession(front.SerializeToString(), so, providers=["CPUExecutionProvider"])
text = "吾輩は猫である、名前はまだ無い、どこで生れたかとんと見当がつかぬ、何でも薄暗いじめじめした所で"
f = frontend.text_to_features(text)
feeds = model._front_feeds(f, 0, model.style_vector(0, 1.0), 1.0, 0.2, 0.6, 0.8)
for _ in range(2):
    s.run([model.z_name, model.g_name], feeds)
pf = s.end_profiling()
ev = json.load(open(pf))
by_node = collections.Counter()
by_prefix = collections.Counter()
by_op = collections.Counter()
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        name = e["name"][: -len("_kernel_time")]
        by_node[name] += e["dur"]
        by_prefix["/".join(name.split("/")[:3])] += e["dur"]
        by_op[e["args"].get("op_name")] += e["dur"]
tot = sum(by_node.values())
print("total kernel ms (2 runs)", tot / 1000)
for k, v in by_prefix.most_common(12):
    print(f"{v/1000:8.1f}ms {k}")
print("--- ops")
for k, v in by_op.most_common(10):
    print(f"{v/1000:8.1f}ms {k}")
print("--- nodes")
for k, v in by_node.most_common(10):
    print(f"{v/1000:8.1f}ms {k}")
