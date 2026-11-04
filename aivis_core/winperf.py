# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""Windows の電力スロットリング (EcoQoS) の制御。

Windows 11 はフォアグラウンドでないプロセスを EcoQoS 扱いにして E コアや低クロックに寄せることがある。
NVDA は常にバックグラウンドで動くため、推論が数倍遅くなることがある。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
ProcessPowerThrottling = 4


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


def set_process_power_throttling(enabled: bool | None) -> bool:
    """enabled=False: 常にスロットリングしない / True: 常に EcoQoS / None: OS 任せ (既定)。"""
    if sys.platform != "win32":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    st = PROCESS_POWER_THROTTLING_STATE()
    st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
    if enabled is None:
        st.ControlMask = 0
        st.StateMask = 0
    else:
        st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        st.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED if enabled else 0
    return bool(
        k32.SetProcessInformation(
            k32.GetCurrentProcess(), ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st)
        )
    )
