# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""既定辞書の適用時間と、適用前後の g2p 時間・読みの違いを比較する。"""
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
from aivis_core.engine import AivisCore
from aivis_core.nlp import bert_models
from aivis_core.nlp.japanese.g2p import g2p
from aivis_core.nlp.japanese.g2p_utils import g2kata_tone
from aivis_core.nlp.japanese.normalizer import normalize_text
from aivis_core.winperf import set_process_power_throttling

set_process_power_throttling(False)
S = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine\BertModelCaches\models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\snapshots")
bert_models.init_tokenizer_only(Path(os.path.dirname(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx"))))
texts = [l for l in open(r"C:\Users\yuki\Downloads\aivis\core_dev\scripts\nvda_texts.txt", encoding="utf-8").read().splitlines() if l.strip()]
texts += ["原神のアップデートで新しいキャラが来た", "新宿駅から高田馬場駅まで", "東海林さんと小鳥遊さん"]


def run():
    t = time.perf_counter()
    out = []
    for text in texts:
        r = normalize_text(text, return_analysis_text=True)
        out.append("".join(k for k, _ in g2kata_tone(r.text, analysis_text=r.analysis_text)))
    return (time.perf_counter() - t) / len(texts), out


g2p_before, yomi_before = run()
t = time.perf_counter()
AivisCore.apply_dictionaries(glob.glob(r"C:\Users\yuki\Downloads\aivis\core_dev\ref\AivisSpeech-Engine\resources\dictionaries\*.dic"))
apply = time.perf_counter() - t
g2p_after, yomi_after = run()
print(f"apply dictionaries: {apply:.2f}s   g2p per utterance: before={g2p_before*1000:.1f}ms after={g2p_after*1000:.1f}ms")
with open(r"C:\Users\yuki\Downloads\aivis\core_dev\work\dict_diff.txt", "w", encoding="utf-8") as f:
    for text, a, b in zip(texts, yomi_before, yomi_after):
        if a != b:
            f.write(f"{text}\n  before: {a}\n  after : {b}\n")
print("diffs:", sum(a != b for a, b in zip(yomi_before, yomi_after)), "/", len(texts))
