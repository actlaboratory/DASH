# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA (onnxruntime-gpu, tools/ort_cuda11 に隔離して導入) で各段を動かしたときの時間と CPU 使用時間。

実際の使い方に近づけるため、テキストの間に待機 (GPU が省電力状態に落ちる) を入れた場合も測る。
"""

import glob
import os
import sys
import time

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
if not os.environ.get("NO_CUDA_PACK"):
    sys.path.insert(0, os.environ.get("CUDA_PACK", os.path.join(ROOT, "tools", "ort_cuda11")))
sys.path.insert(0, ROOT)

if os.environ.get("NO_CUDA_PACK"):
    import onnxruntime as ort  # noqa: E402
else:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import _cuda_loader  # noqa: E402

    sys.path.pop(0)
    ort = _cuda_loader.load(os.environ.get("CUDA_PACK", os.path.join(ROOT, "tools", "ort_cuda11")))
print("onnxruntime", ort.__version__, ort.__file__)
print("providers:", ort.get_available_providers())

from aivis_core import AivisCore  # noqa: E402
from aivis_core import engine  # noqa: E402

if os.environ.get("ALGO"):
    _dp = engine.default_providers

    def _patched(use_gpu):
        return [(p[0], {**p[1], "cudnn_conv_algo_search": os.environ["ALGO"]}) if isinstance(p, tuple) and p[0] == "CUDAExecutionProvider" else p
                for p in _dp(use_gpu)]

    engine.default_providers = _patched
from aivis_core import frontend  # noqa: E402
from aivis_core.nlp import bert_models  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
MODE = os.environ.get("MODE", "cuda")
t = time.perf_counter()
core = AivisCore(bert, models, use_gpu=(MODE != "cpu"))
print("mode", MODE, "providers", core.providers, f"init {time.perf_counter() - t:.1f}s")
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
st = core.styles[vid]
model = core.get_model(st.model_uuid)
print("front providers:", model.front.get_providers(), "dec:", model.dec.get_providers())

_b = [0.0, 0.0]
_orig = bert_models.extract_bert_feature


def timed_bert(*a, **k):
    t, c = time.perf_counter(), time.process_time()
    r = _orig(*a, **k)
    _b[0] += time.perf_counter() - t
    _b[1] += time.process_time() - c
    return r


bert_models.extract_bert_feature = timed_bert

texts = [l for l in open(os.path.join(ROOT, "scripts", "nvda_texts.txt"), encoding="utf-8").read().splitlines() if l.strip()]
texts = texts[:15]
hop, sr = model.hop_length, model.sampling_rate


def one(text):
    _b[0] = _b[1] = 0.0
    t0, c0 = time.perf_counter(), time.process_time()
    f = frontend.text_to_features(text, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
    t1, c1 = time.perf_counter(), time.process_time()
    z, g = model.run_front(f, st.speaker_local_id, st.style_local_id)
    t2, c2 = time.perf_counter(), time.process_time()
    first = None
    n = 0
    for ch in model.decode_stream(z, g, first_chunk_frames=32, chunk_frames=128, margin_frames=12):
        if first is None:
            first = time.perf_counter() - t2
        n += ch.size
    t3, c3 = time.perf_counter(), time.process_time()
    return dict(bert=_b[0], bert_cpu=_b[1], g2p=t1 - t0 - _b[0], front=t2 - t1, front_cpu=c2 - c1,
                dec=t3 - t2, dec_cpu=c3 - c2, dec_first=first, audio=n / sr,
                to_first_audio=(t2 - t0) + first, total=t3 - t0, cpu=c3 - c0)


for t in texts[:3]:
    one(t)  # ウォームアップ

for idle in (0.0, 1.0):
    rows = []
    for t in texts:
        if idle:
            time.sleep(idle)
        rows.append(one(t))
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2] * 1000  # noqa: E731
    print(f"[{MODE} idle={idle}s] median ms: bert={med('bert'):.0f} front={med('front'):.0f} dec_first={med('dec_first'):.0f} "
          f"to_first_audio={med('to_first_audio'):.0f} | RTF total={s['total'] / s['audio']:.3f} dec={s['dec'] / s['audio']:.3f} "
          f"| CPU sec per audio sec={s['cpu'] / s['audio']:.3f} (bert {s['bert_cpu'] / s['audio']:.3f}, "
          f"front {s['front_cpu'] / s['audio']:.3f}, dec {s['dec_cpu'] / s['audio']:.3f})")
