# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""GPU (CUDA) が休止したあとの 1 発話目の遅れと、その対策を、今の組み合わせ (onnxruntime 1.20.1 + cuDNN 8.9) で比べる。

  MODE=cpu|gpu  KEEPALIVE=<秒>  COLD=none|bert|bert+dec  tools\\py313\\python.exe -I scripts\\bench_gpu_idle.py

- 黙っている時間 (IDLES 秒) のあとに短い文を読み、最初の音までの時間を測る (等倍、最初に溜める量は既定の 150ms)
- KEEPALIVE: 別スレッドで小さなデコード (16 フレーム) を一定間隔で流し、GPU を起こしておく
- COLD=bert: GPU が COLD_SEC 以上使われていなければ、その文の BERT は CPU のセッションで計算する
- COLD=bert+dec: さらに最初の区間のデコードも CPU で行い、同時に別スレッドで GPU を起こしておく
BERT は同梱と同じ fp32 版 (build/addon の data/bert)。CUDA は tools/ort_cuda11 (CUDA パックの元)。
"""

import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE = os.environ.get("MODE", "gpu")
if MODE == "gpu":
    sys.path.insert(0, str(ROOT / "scripts"))
    import _cuda_loader  # noqa: E402

    sys.path.pop(0)
    ort = _cuda_loader.load(str(ROOT / "tools" / "ort_cuda11"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from aivis_core import engine  # noqa: E402
from aivis_core.engine import AivisCore, SynthParams, make_session_options  # noqa: E402
from aivis_core.frontend import text_to_features  # noqa: E402
from aivis_core.nlp import bert_models  # noqa: E402

IDLES = [float(x) for x in os.environ.get("IDLES", "0,0.5,2,5,15").split(",")]
KEEPALIVE = float(os.environ.get("KEEPALIVE", "0"))
COLD = os.environ.get("COLD", "none")
COLD_SEC = float(os.environ.get("COLD_SEC", "0.7"))
TEXTS = ["ファイル", "編集", "表示", "ボタン", "リンク ホーム", "閉じる"]

data = ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data"
core = AivisCore(data / "bert", sorted((data / "models").glob("*.aivmx")), use_gpu=(MODE == "gpu"))
vid = next(iter(core.styles))
st = core.styles[vid]
model = core.get_model(st.model_uuid)
print(f"mode={MODE} providers={core.providers} keepalive={KEEPALIVE} cold={COLD}", flush=True)

# --- 対策用の CPU のセッション (COLD のとき) ---
bert_cpu = dec_cpu = None
if MODE == "gpu" and COLD in ("bert", "bert+dec"):
    so = make_session_options(0)
    bert_cpu = ort.InferenceSession(str(data / "bert" / "model.onnx"), so, providers=["CPUExecutionProvider"])
if MODE == "gpu" and COLD in ("bert+dec", "dec"):
    import onnx
    from onnx.utils import Extractor

    m = onnx.load(str(model.path))
    dec_bytes = Extractor(m).extract_model([model.z_name, model.g_name], [model.output_name]).SerializeToString()
    del m
    dec_cpu = ort.InferenceSession(dec_bytes, make_session_options(0), providers=["CPUExecutionProvider"])

last_gpu = [time.perf_counter()]
lock = threading.Lock()


def dummy_gpu():
    zi, gi = model.dec.get_inputs()
    z = np.zeros((1, 192, 16), np.float32)
    g = np.zeros((1, gi.shape[1] if isinstance(gi.shape[1], int) else 512, 1), np.float32)
    with model.dec_lock:
        model.dec.run(None, {zi.name: z, gi.name: g})
    last_gpu[0] = time.perf_counter()


KEEP_BERT = os.environ.get("KEEP_BERT") == "1"
KEEP_DEC = os.environ.get("KEEP_DEC", "1") == "1"


def dummy_bert():
    sess = bert_models._bert_session
    ids = np.full((1, 8), 1, dtype=np.int64)
    with bert_models._bert_lock:
        sess.run(None, {sess.get_inputs()[0].name: ids, sess.get_inputs()[1].name: np.ones_like(ids)})


if MODE == "gpu" and KEEPALIVE > 0:
    # KEEP_BERT=1 なら BERT にも小さな入力を流す (デコーダだけでは BERT の 1 回目が遅いままか確かめる)
    def keep():
        while True:
            if KEEP_DEC:
                dummy_gpu()
            if KEEP_BERT:
                dummy_bert()
            time.sleep(KEEPALIVE)

    threading.Thread(target=keep, daemon=True).start()

# BERT の呼び出しを、COLD のときだけ CPU のセッションに振り分ける
orig_extract = bert_models.extract_bert_feature
use_cpu_now = [False]


def extract(text, word2ph):
    if use_cpu_now[0] and bert_cpu is not None:
        saved = bert_models._bert_session
        bert_models._bert_session = bert_cpu
        try:
            return orig_extract(text, word2ph)
        finally:
            bert_models._bert_session = saved
    return orig_extract(text, word2ph)


bert_models.extract_bert_feature = extract
import aivis_core.frontend as frontend  # noqa: E402

frontend.bert_models.extract_bert_feature = extract


SPIN_MS = float(os.environ.get("SPIN_MS", "0"))


def first_audio(text):
    """最初の音 (150ms 分) ができるまでの時間 [ms]。SPIN_MS があれば、その時間 CPU を空回りさせてから始める (時間に含む)。"""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000 < SPIN_MS:
        pass
    cold = MODE == "gpu" and COLD != "none" and (t0 - last_gpu[0] > COLD_SEC)
    use_cpu_now[0] = cold and COLD in ("bert", "bert+dec")
    if cold and COLD in ("bert+dec", "dec"):
        threading.Thread(target=dummy_gpu, daemon=True).start()
    f = text_to_features(text, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
    z, g = model.run_front(f, st.speaker_local_id, st.style_local_id)
    total = z.shape[2]
    n = min(total, 17)  # 150ms ≒ 13 フレーム + 先頭の無音分 (engine.decode と同じ)
    e = min(total, n + 12)
    sess = dec_cpu if (cold and dec_cpu is not None) else model.dec
    if sess is model.dec:
        with model.dec_lock:
            sess.run(None, {model.z_name: z[:, :, 0:e], model.g_name: g})
        last_gpu[0] = time.perf_counter()
    else:
        sess.run(None, {model.z_name: z[:, :, 0:e], model.g_name: g})
    if not cold and MODE == "gpu":
        last_gpu[0] = time.perf_counter()
    return (time.perf_counter() - t0) * 1000


for t in TEXTS:
    first_audio(t)  # ウォームアップ
for idle in IDLES:
    xs = []
    for t in TEXTS:
        time.sleep(idle)
        xs.append(first_audio(t))
    print(f"  idle {idle:4.1f}s: median {statistics.median(xs):5.0f}ms  max {max(xs):5.0f}ms  "
          f"all={[round(x) for x in xs]}", flush=True)
