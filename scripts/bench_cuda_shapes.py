# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA で「初めての入力の長さ」のときにどれだけ遅くなるか (デコーダ / BERT)。"""

import glob
import os
import sys
import time

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _cuda_loader  # noqa: E402

sys.path.pop(0)
ort = _cuda_loader.load(os.environ.get("CUDA_PACK", os.path.join(ROOT, "tools", "ort_cuda11")))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

print("onnxruntime", ort.__version__)
from aivis_core import AivisCore  # noqa: E402
from aivis_core.nlp import bert_models  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
core = AivisCore(bert, models, use_gpu=True)
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
model = core.get_model(core.styles[vid].model_uuid)
rng = np.random.default_rng(0)
g = rng.standard_normal((1, 512, 1), dtype=np.float32) * 0.1
model.dec.run(None, {model.z_name: rng.standard_normal((1, 192, 64), dtype=np.float32), model.g_name: g})


def t_dec(L):
    z = rng.standard_normal((1, 192, L), dtype=np.float32)
    t = time.perf_counter()
    model.dec.run(None, {model.z_name: z, model.g_name: g})
    return (time.perf_counter() - t) * 1000


print("dec (L: first, second) ms:", [(L, round(t_dec(L)), round(t_dec(L))) for L in (40, 41, 47, 77, 100, 152, 153, 200)])

sess = bert_models._bert_session


def t_bert(n):
    ids = np.full((1, n), 5, dtype=np.int64)
    feeds = {sess.get_inputs()[0].name: ids, sess.get_inputs()[1].name: np.ones_like(ids)}
    t = time.perf_counter()
    sess.run(None, feeds)
    return (time.perf_counter() - t) * 1000


t_bert(10)
print("bert (n: first, second) ms:", [(n, round(t_bert(n)), round(t_bert(n))) for n in (8, 9, 17, 30, 31, 60)])
