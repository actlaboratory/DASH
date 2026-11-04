# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np
import onnxruntime as ort

from aivis_core import frontend
from aivis_core.engine import AivisModel, make_session_options
from aivis_core.nlp import bert_models

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT_DIR = Path(os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")))
bert_models.init(BERT_DIR, make_session_options(int(os.environ.get("TH", "0"))))
model = AivisModel(sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0], None, int(os.environ.get("TH", "0")))
text = "吾輩は猫である、名前はまだ無い、どこで生れたかとんと見当がつかぬ、何でも薄暗いじめじめした所で"
f = frontend.text_to_features(text)
print("phones", len(f.phones))
for sdp, nz, nw in [(0.0, 0.0, 0.0), (0.2, 0.6, 0.8), (0.0, 0.6, 0.8), (0.2, 0.0, 0.0)]:
    feeds = model._front_feeds(f, 0, model.style_vector(0, 1.0), 1.0, sdp, nz, nw)
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        z, g = model.front.run([model.z_name, model.g_name], feeds)
        ts.append(time.perf_counter() - t)
    print(f"sdp={sdp} noise={nz} noise_w={nw} frames={z.shape[2]} times=" + " ".join(f"{x*1000:.0f}" for x in ts))
