# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA パックのダウンロード元 (URL, サイズ, sha256) を調べる。結果を _aivis/cuda_installer.py の SOURCES に書く。

- onnxruntime-gpu 1.20.1 (CUDA 11) : Microsoft の onnxruntime-cuda-11 フィード (Azure Artifacts)
- nvidia-*-cu11 : PyPI
"""

import json
import re
import urllib.request

AZURE = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/onnxruntime-gpu/"
PYPI = [
    ("nvidia-cuda-runtime-cu11", "11.8.89"),
    ("nvidia-cublas-cu11", "11.11.3.6"),
    ("nvidia-cufft-cu11", "10.9.0.58"),
    ("nvidia-cudnn-cu11", "8.9.5.29"),
]


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.geturl()


html = urllib.request.urlopen(AZURE, timeout=60).read().decode()
m = re.search(r'href="([^"]*onnxruntime_gpu-1\.20\.1-cp313-cp313-win_amd64\.whl[^"]*)"', html)
url = m.group(1).replace("&amp;", "&")
base, _, frag = url.partition("#")
sha = frag.split("=", 1)[1] if "sha256=" in frag else None
size, final = head_size(base)
print(json.dumps({"name": "onnxruntime-gpu 1.20.1 cu11", "url": base, "sha256": sha, "size": size}))

for name, ver in PYPI:
    data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=60))
    for f in data["urls"]:
        if f["filename"].endswith("win_amd64.whl"):
            print(json.dumps({"name": f"{name} {ver}", "url": f["url"], "sha256": f["digests"]["sha256"],
                              "size": f["size"]}))
