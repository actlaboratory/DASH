# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""(1) 自作トークナイザと transformers.BertJapaneseTokenizer の一致
(2) 区間デコード (のりしろ M フレーム) と一括デコードの誤差"""
import glob
import os
import sys

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
import numpy as np

from aivis_core.engine import AivisModel
from aivis_core.frontend import text_to_features
from aivis_core.nlp import bert_models
from aivis_core.nlp.japanese.g2p import g2p
from aivis_core.nlp.japanese.normalizer import normalize_text

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT_DIR = os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx"))

TEXTS = [
    "吾輩は猫である。名前はまだ無い。",
    "NVDAのバージョン2026.1をインストールしました！",
    "メールは test@example.com まで。価格は1,980円（税込）です…？",
    "「こんにちは」と彼は言った。ｱｲｳｴｵ、ＡＢＣ、①②③、𠮷野家。",
    "Windows 11 Pro の設定画面を開きます。Ctrl+Alt+Delete を押してください。",
]

# (1) tokenizer
from transformers import BertJapaneseTokenizer

ref = BertJapaneseTokenizer.from_pretrained(BERT_DIR)
bert_models.init_tokenizer_only(__import__("pathlib").Path(BERT_DIR))
mine = bert_models.load_tokenizer()
ok = True
for t in TEXTS:
    r = normalize_text(t, return_analysis_text=True)
    _, _, _, sep_text, _, _ = g2p(r.text, analysis_text=r.analysis_text)
    joined = "".join(sep_text)
    a = ref(joined, return_tensors="np")["input_ids"][0].tolist()
    b = mine.encode(joined)
    for w in sep_text:
        if ref.tokenize(w) != mine.tokenize(w):
            ok = False
            print("tokenize mismatch", w, ref.tokenize(w), mine.tokenize(w))
    print("ids equal" if a == b else "IDS DIFFER", joined)
    ok &= a == b
print("TOKENIZER OK" if ok else "TOKENIZER MISMATCH")

# (2) margin
bert_models.init(__import__("pathlib").Path(BERT_DIR))
model = AivisModel(sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0])
f = text_to_features(TEXTS[0] + TEXTS[1])
style = model.style_vector(0, 1.0)
feeds = model._front_feeds(f, 0, style, 1.0, 0.0, 0.0, 0.0)
z, g = model.front.run([model.z_name, model.g_name], feeds)
full = model.dec.run(None, {model.z_name: z, model.g_name: g})[0][0, 0]
print("frames", z.shape[2], "full peak", np.abs(full).max())
hop = model.hop_length
for M in [0, 2, 4, 8, 12, 16, 24, 32]:
    out = []
    pos, n, total = 0, 16, z.shape[2]
    while pos < total:
        end = min(total, pos + n)
        s, e = max(0, pos - M), min(total, end + M)
        w = model.dec.run(None, {model.z_name: z[:, :, s:e], model.g_name: g})[0][0, 0]
        out.append(w[(pos - s) * hop:(end - s) * hop])
        pos, n = end, 64
    out = np.concatenate(out)
    err = np.abs(out - full)
    snr = 10 * np.log10(np.sum(full**2) / max(np.sum((out - full) ** 2), 1e-20))
    print(f"margin={M:2d}  max_err={err.max():.2e}  SNR={snr:.1f}dB")
