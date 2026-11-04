# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""BERT fp16 / fp32 / int8(動的量子化) の速度と特徴量の差を比較する。"""
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np
import onnxruntime as ort

from aivis_core.engine import make_session_options
from aivis_core.nlp.bert_models import CharTokenizer
from aivis_core.winperf import set_process_power_throttling

set_process_power_throttling(False)
S = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine\BertModelCaches\models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\snapshots")
fp16 = glob.glob(S + r"\*\model_fp16.onnx")[0]
fp32 = glob.glob(S + r"\*\model.onnx")[0]
int8 = r"C:\Users\yuki\Downloads\aivis\core_dev\work\bert_int8.onnx"
if not os.path.exists(int8):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t = time.time()
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"quantized in {time.time()-t:.0f}s, size {os.path.getsize(int8)/1e6:.0f}MB")
tok = CharTokenizer(Path(os.path.dirname(fp16)) / "vocab.txt")
texts = ["ボタン", "ファイルメニュー", "吾輩は猫である.名前はまだ無い.", "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している." * 2]
ref = {}
for label, path in [("fp16", fp16), ("fp32", fp32), ("int8", int8)]:
    t = time.time()
    s = ort.InferenceSession(path, make_session_options(0), providers=["CPUExecutionProvider"])
    load = time.time() - t
    names = [i.name for i in s.get_inputs()]
    line = f"{label}: load={load:.1f}s "
    for text in texts:
        ids = np.array([tok.encode(text)], dtype=np.int64)
        feeds = {names[0]: ids, names[1]: np.ones_like(ids)}
        s.run(None, feeds)
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            out = s.run(None, feeds)[0].astype(np.float32)
            ts.append(time.perf_counter() - t)
        if label == "fp16":
            ref[text] = out
            d = 0.0
        else:
            d = float(np.abs(out - ref[text]).max() / np.abs(ref[text]).max())
            cos = float(np.mean(np.sum(out * ref[text], -1) / (np.linalg.norm(out, axis=-1) * np.linalg.norm(ref[text], axis=-1))))
        line += f"| len={ids.shape[1]:3d} {np.median(ts)*1000:4.0f}ms" + (f" rel={d:.3f} cos={cos:.4f}" if label != "fp16" else "")
    print(line)
    del s
