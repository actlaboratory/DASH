# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""aivis_core を使って実際に合成したあと、site-packages から読み込まれたトップレベルモジュールを列挙する。"""
import glob
import os
import sys

sys.path.insert(0, r"C:\Users\yuki\Downloads\aivis\core_dev")
from aivis_core import AivisCore, SynthParams

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
BERT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "addon", "synthDrivers", "_aivis", "data", "bert"), "model.onnx")
core = AivisCore(os.path.dirname(BERT), sorted(glob.glob(A + r"\Models\a59*.aivmx")))
core.apply_dictionaries(glob.glob(r"C:\Users\yuki\Downloads\aivis\core_dev\ref\AivisSpeech-Engine\resources\dictionaries\*.dic")[:2])
vid = next(iter(core.styles))
core.synthesize("NVDAで3.5%の値段、Hello world、2026年11月3日。", SynthParams(voice_id=vid))
sp = os.path.normcase(os.path.join(sys.prefix, "Lib", "site-packages"))
tops = set()
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if f and os.path.normcase(f).startswith(sp):
        rel = os.path.relpath(os.path.normcase(f), sp)
        tops.add(rel.split(os.sep)[0])
print(sorted(tops))
