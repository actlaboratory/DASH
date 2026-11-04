# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""Sonic のピッチ変更で基本周波数 (F0) が狙いどおり変わるか確認する。

NVDA 本体の synthDrivers._sonic と sonic.dll を読み取りのみで使う (NVDA_DIR)。
"""

import os
import sys
import types
import zipfile
import importlib.util

import numpy as np

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
NVDA_DIR = os.environ["NVDA_DIR"]
sys.path.insert(0, ROOT)


g = types.ModuleType("globalVars")
g.appDir = NVDA_DIR
sys.modules["globalVars"] = g
sys.modules["logHandler"] = types.SimpleNamespace(log=__import__("logging").getLogger())
# library.zip 内の synthDrivers/_sonic.pyc を読む
sd = types.ModuleType("synthDrivers")
sd.__path__ = [os.path.join(NVDA_DIR, "library.zip", "synthDrivers")]
sys.modules["synthDrivers"] = sd
sys.path.append(os.path.join(NVDA_DIR, "library.zip"))
from synthDrivers import _sonic  # noqa: E402

_sonic.initialize()

from aivis_core import AivisCore, SynthParams  # noqa: E402
import glob  # noqa: E402

appdata = os.path.join(os.environ["APPDATA"], "AivisSpeech-Engine")
bert = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert")
models = glob.glob(os.path.join(appdata, "Models", "*.aivmx"))
core = AivisCore(bert, models)
vid = [v for v, s in core.styles.items() if "Anneli" in s.display_name][0]
pcm = np.ascontiguousarray(core.synthesize("あいうえお、こんにちは。", SynthParams(voice_id=vid)), dtype=np.int16)
SR = 44100


def f0_median(x):
    x = x.astype(np.float64) / 32768
    win, hop = 2048, 512
    out = []
    for i in range(0, len(x) - win, hop):
        f = x[i:i + win] * np.hanning(win)
        if np.sqrt(np.mean(f ** 2)) < 0.02:
            continue
        ac = np.correlate(f, f, "full")[win - 1:]
        lo, hi = SR // 800, SR // 60
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            out.append(SR / lag)
    return float(np.median(out)), len(x) / SR


def pitch_to_factor(p):  # addon/synthDrivers/_aivis/__init__.py と同じ式
    return 2.0 ** ((p - 50) / 50.0 * 0.5)


base, dur = f0_median(pcm)
print(f"original: F0={base:.1f}Hz dur={dur:.2f}s")
for p in (0, 20, 50, 80, 100):
    s = _sonic.SonicStream(SR, 1)
    fac = pitch_to_factor(p)
    s.pitch = fac
    s.writeShort(pcm.ctypes.data, pcm.size)
    s.flush()
    y = np.frombuffer(bytes(s.readShort()), dtype=np.int16)
    f, d = f0_median(y)
    print(f"pitch {p:3d}: factor={fac:.3f} F0={f:.1f}Hz ratio={f / base:.3f} dur={d:.2f}s")
