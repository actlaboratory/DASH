# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA 加速のダウンロード用マニフェスト (addon/synthDrivers/_aivis/cuda_manifest.json) を作る。

各ダウンロード元 (wheel / zip) について、取り出すファイルと、その中身の sha256・大きさを記録する。
sha256 は検証済みの tools/ort_cuda11 (ハーネスで動作確認したもの) から計算するので、
インストール時にダウンロード元が改ざんされていても検出できる。
"""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon" / "synthDrivers" / "_aivis"))
from remote_zip import HttpRangeFile  # noqa: E402

LOCAL = ROOT / "tools" / "ort_cuda11"
OUT = ROOT / "addon" / "synthDrivers" / "_aivis" / "cuda_manifest.json"

ORT_EXCLUDE = ("tensorrt", "/test", "/tools/", "/transformers/", "/quantization/", "__pycache__")

SOURCES = [
    {
        "name": "onnxruntime-gpu 1.20.1 (CUDA 11)",
        "url": "https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/9c975526-0258-4aac-9e34-f8b3551decdd/pypi/download/onnxruntime-gpu/1.20.1/onnxruntime_gpu-1.20.1-cp313-cp313-win_amd64.whl",
        "sha256": "748edbec9499411d7b1c507ca190ab37142eca2eaa2faf5916ebf84f96b9712c",
        "want": lambda n: n.startswith("onnxruntime/") and not n.endswith("/") and not any(x in n for x in ORT_EXCLUDE)
        and not n.endswith((".lib", ".pdb")),
    },
    {
        "name": "CUDA Runtime 11.8",
        "url": "https://files.pythonhosted.org/packages/48/ae/09f335dfbca630ae213e53d2a9294540620453fe2b37e7489ff05817a525/nvidia_cuda_runtime_cu11-11.8.89-py3-none-win_amd64.whl",
        "sha256": "f60c9fdaed8065b38de8097867240efc5556a8a710007146daeb9082334a6e63",
        "want": lambda n: n.endswith("/cudart64_110.dll"),
    },
    {
        "name": "cuBLAS 11",
        "url": "https://files.pythonhosted.org/packages/0b/1d/7a78cd36fd5e3da4021b3ac2c2c8b2651dd72345b7c3ecc0d3e051884f50/nvidia_cublas_cu11-11.11.3.6-py3-none-win_amd64.whl",
        "sha256": "6ab12b1302bef8ac1ff4414edd1c059e57f4833abef9151683fb8f4de25900be",
        "want": lambda n: n.endswith(("/cublas64_11.dll", "/cublasLt64_11.dll")),
    },
    {
        "name": "cuFFT 10",
        "url": "https://files.pythonhosted.org/packages/f8/b4/e432a74f8db0e84f734dc14d36c0e529225132bf7e239da21f55893351a6/nvidia_cufft_cu11-10.9.0.58-py3-none-win_amd64.whl",
        "sha256": "c4d316f17c745ec9c728e30409612eaf77a8404c3733cdf6c9c1569634d1ca03",
        "want": lambda n: n.endswith("/cufft64_10.dll"),
    },
    {
        "name": "cuDNN 8.9.5",
        "url": "https://files.pythonhosted.org/packages/b3/f5/97140674634a5a4f44387677bb8e5b75f72ea81ca108875f1a81854b3367/nvidia_cudnn_cu11-8.9.5.29-py3-none-win_amd64.whl",
        "sha256": "ba6358cafcbf9ab66887099f5c821dde3968d7b4580cb87c56b548947917b54d",
        "want": lambda n: n.endswith(("/cudnn64_8.dll", "/cudnn_ops_infer64_8.dll", "/cudnn_cnn_infer64_8.dll")),
    },
]
# zlibwapi.dll (cuDNN 8 の依存)。VSU と同じ配布元。小さいので常に全体をダウンロードする
ZLIB = {
    "name": "zlibwapi (zlib 1.2.3)",
    "url": "http://www.winimage.com/zLibDll/zlib123dllx64.zip",
    "sha256": "fd324c6923aa4f45a60413665e0b68bb34a7779d0861849e02d2711ff8efb9a4",
    "member": "dll_x64/zlibwapi.dll",
    "dest": "nvidia/cudnn/bin/zlibwapi.dll",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = {"version": "cuda11.8-cudnn8.9.5-ort1.20.1", "sources": []}
    total_dl = total_size = 0
    for s in SOURCES:
        f = HttpRangeFile(s["url"])
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
        files = []
        for info in z.infolist():
            if not s["want"](info.filename):
                continue
            local = LOCAL / info.filename
            assert local.is_file(), local
            assert local.stat().st_size == info.file_size, (local, info.file_size)
            files.append({"member": info.filename, "dest": info.filename, "size": info.file_size,
                          "compressed": info.compress_size, "sha256": sha256_file(local)})
            total_dl += info.compress_size
            total_size += info.file_size
        assert files, s["name"]
        out["sources"].append({"name": s["name"], "url": s["url"], "sha256": s["sha256"], "size": f.size,
                               "files": files})
        print(f"{s['name']}: {len(files)} files, {sum(x['compressed'] for x in files) / 2**20:.0f}MB to download")
    zl = LOCAL / ZLIB["dest"]
    out["zlib"] = {**ZLIB, "file_sha256": sha256_file(zl), "file_size": zl.stat().st_size}
    total_dl += 145513
    total_size += zl.stat().st_size
    out["download_bytes"] = total_dl
    out["install_bytes"] = total_size
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: download {total_dl / 2**20:.0f}MB, install {total_size / 2**20:.0f}MB")


if __name__ == "__main__":
    main()
