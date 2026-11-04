# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""tsqyomi (文脈による読み分け) を使ったときの g2p の時間・読みの違い・読み込み時間・メモリを測る。

モデルは build/cache (scripts/third_party.py でダウンロードしたもの) を work/tsqyomi に集めて model_dir で読む。
"""

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

d = ROOT / "work" / "tsqyomi"
d.mkdir(parents=True, exist_ok=True)
for key, name in (("tsqyomi-model", "model.onnx"), ("tsqyomi-tokenizer", "tokenizer.json"),
                  ("tsqyomi-metadata", "metadata.json")):
    src = ROOT / "build" / "cache" / key / name
    if not (d / name).exists():
        shutil.copy2(src, d / name)


def rss_gb():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t)] + [(f"x{i}", ctypes.c_size_t) for i in range(6)]
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 2**30


from aivis_core.nlp import bert_models  # noqa: E402
from aivis_core.nlp.japanese.g2p import g2p  # noqa: E402
from aivis_core.nlp.japanese.normalizer import normalize_text  # noqa: E402

bert_dir = ROOT / "build" / "addon" / "synthDrivers" / "_aivis" / "data" / "bert"
bert_models.init_tokenizer_only(bert_dir)
from pyopenjtalk import tsqyomi  # noqa: E402

texts = [l for f in ("nvda_texts.txt", "texts.txt") for l in (ROOT / "scripts" / f).read_text(encoding="utf-8").splitlines()
         if l.strip()]
texts += ["人気のない夜道を歩く。", "この店は人気がある。", "今日は何日ですか。", "上手から役者が登場する。", "彼は上手に話す。",
          "一日中雨だった。", "四月一日に入社した。", "市場で魚を買う。", "株式市場が開いた。", "風車が回っている。"]


def run(use):
    out, times = [], []
    for t in texts:
        norm = normalize_text(t, return_analysis_text=True)
        t0 = time.perf_counter()
        phones = g2p(norm.text, analysis_text=norm.analysis_text, use_jp_extra=True, raise_yomi_error=False,
                     use_tsqyomi=use)[0]
        times.append(time.perf_counter() - t0)
        out.append(" ".join(phones))
    return out, times


run(False)  # ウォームアップ
base, tb = run(False)
m0 = rss_gb()
t0 = time.perf_counter()
tsqyomi.load_model(onnx_providers=["CPUExecutionProvider"], model_dir=d, allow_provider_fallback=False)
load = time.perf_counter() - t0
m1 = rss_gb()
run(True)
ts, tt = run(True)
med = lambda xs: sorted(xs)[len(xs) // 2] * 1000  # noqa: E731
print(f"texts={len(texts)} load={load:.2f}s memory +{(m1 - m0):.2f}GB")
print(f"g2p ms median/max: without={med(tb):.1f}/{max(tb) * 1000:.1f}  with tsqyomi={med(tt):.1f}/{max(tt) * 1000:.1f}")
diff = [(t, a, b) for t, a, b in zip(texts, base, ts) if a != b]
print(f"reading changed in {len(diff)} texts")
with open(ROOT / "work" / "tsqyomi_diff.txt", "w", encoding="utf-8") as f:
    for t, a, b in diff:
        f.write(f"{t}\n  without: {a}\n  with:    {b}\n")
