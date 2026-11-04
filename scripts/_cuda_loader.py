# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""検証スクリプト用: CUDA パック (onnxruntime-gpu + nvidia/*/bin) を読み込む。

onnxruntime 1.21 以降は preload_dlls() があるが、1.20 (CUDA 11 / cuDNN 8 版) には無いので、
nvidia/*/bin の DLL を依存の順に自分で読み込む。cuDNN 8 は zlibwapi.dll も必要。
"""

import ctypes
import os
import sys

# 依存される側から順に
_ORDER = ["cudart", "cublasLt", "cublas", "cufft", "zlibwapi", "cudnn_ops_infer", "cudnn_cnn_infer", "cudnn_adv_infer",
          "cudnn_graph", "cudnn_engines", "cudnn_heuristic", "cudnn_ops", "cudnn_cnn", "cudnn_adv", "cudnn64"]


def load(pack: str):
    pack = os.path.abspath(pack)
    sys.path.insert(0, pack)
    import onnxruntime as ort

    bins = [os.path.join(pack, "nvidia", d, "bin") for d in sorted(os.listdir(os.path.join(pack, "nvidia")))
            if os.path.isdir(os.path.join(pack, "nvidia", d, "bin"))]
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
        return ort
    for b in bins:
        os.add_dll_directory(b)
    # cuDNN 8 は中で LoadLibrary で分割 DLL を探すので PATH にも入れる (このプロセス内だけ)
    os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    dlls = [os.path.join(b, n) for b in bins for n in os.listdir(b) if n.endswith(".dll")
            and not n.startswith(("nvblas", "cufftw", "cudart32", "cudnn_adv_train", "cudnn_cnn_train", "cudnn_ops_train"))]

    def rank(p):
        n = os.path.basename(p)
        for i, k in enumerate(_ORDER):
            if n.startswith(k):
                return i
        return len(_ORDER)

    for p in sorted(dlls, key=rank):
        try:
            ctypes.CDLL(p)
        except OSError as e:
            print("load failed", p, e)
    return ort
