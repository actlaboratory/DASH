# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""高倍速 (5 倍) で合成が間に合うかを見るため、テキストごとに各段の処理時間を測る。

g2p / BERT / 前段 (enc+flow) / デコーダ を分けて測り、音声の長さ (等倍) と比べる。
AIVIS 側の話速 (length_scale) を変えたときの違いも見る。
"""

import glob
import os
import sys
import time

import numpy as np

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
sys.path.insert(0, ROOT)

from aivis_core import AivisCore, SynthParams  # noqa: E402
from aivis_core import frontend  # noqa: E402
from aivis_core.nlp import bert_models  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
core = AivisCore(bert, models)
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
st = core.styles[vid]
model = core.get_model(st.model_uuid)
core.synthesize("風が強い。", SynthParams(voice_id=vid))

# BERT の時間だけ測るためにラップ
_bert_t = [0.0]
_orig = bert_models.extract_bert_feature


def timed_bert(*a, **k):
    t = time.perf_counter()
    r = _orig(*a, **k)
    _bert_t[0] += time.perf_counter() - t
    return r


bert_models.extract_bert_feature = timed_bert


TEXTS = ["ボタン", "ファイル メニュー", "吾輩は猫である。", "名前はまだ無い。", "どこで生れたかとんと見当がつかぬ。",
         "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
         "速度を上げて読み上げています。"]
cfg = core.stream
for aspeed in (1.0, 1.25, 1.5):
    tot = {"feat": 0, "bert": 0, "front": 0, "dec": 0, "audio": 0}
    print(f"--- AIVIS speed {aspeed}")
    for text in TEXTS:
        best = None
        for _ in range(3):
            _bert_t[0] = 0.0
            t0 = time.perf_counter()
            f = frontend.text_to_features(text, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
            t1 = time.perf_counter()
            z, g = model.run_front(f, st.speaker_local_id, st.style_local_id, length_scale=1 / aspeed)
            t2 = time.perf_counter()
            n = 0
            for ch in model.decode_stream(z, g, first_chunk_frames=16, chunk_frames=cfg.chunk_frames,
                                          margin_frames=cfg.margin_frames):
                n += ch.size
            t3 = time.perf_counter()
            r = dict(feat=t1 - t0 - _bert_t[0], bert=_bert_t[0], front=t2 - t1, dec=t3 - t2, audio=n / 44100,
                     frames=z.shape[2])
            if best is None or (t3 - t0) < sum(best[k] for k in ("feat", "bert", "front", "dec")):
                best = r
        for k in tot:
            tot[k] += best[k]
        ms = {k: round(best[k] * 1000) for k in ("feat", "bert", "front", "dec")}
        print(f"  {text[:16]:16s} frames={best['frames']:4d} audio={best['audio']:.2f}s {ms} "
              f"total={sum(ms.values())}ms  play@5x={best["audio"] * aspeed / 5 * 1000:.0f}ms")
    print(f"  TOTAL g2p={tot['feat']:.2f} bert={tot['bert']:.2f} front={tot['front']:.2f} dec={tot['dec']:.2f} "
          f"audio={tot['audio']:.2f}s  -> synth/audio RTF={(tot['feat'] + tot['bert'] + tot['front'] + tot['dec']) / tot['audio']:.3f}")
