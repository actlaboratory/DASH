# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import sys
import time

t = time.time()
import pyopenjtalk

print("import pyopenjtalk", round(time.time() - t, 2))
t = time.time()
pyopenjtalk.g2p("テスト")
print("first g2p", round(time.time() - t, 2))
t = time.time()
pyopenjtalk.run_frontend("風がこんな風に吹く")
print("homograph (sudachi load)", round(time.time() - t, 2))
t = time.time()
pyopenjtalk.run_frontend("今日は風が強い")
print("homograph 2nd", round(time.time() - t, 3))
print("sudachidict_core loaded:", "sudachidict_core" in sys.modules)
