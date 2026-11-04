# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""DASH 用 CUDA パック (別アドオン aivisCudaPack) をビルドする。

NVIDIA GPU の人だけが追加で入れるアドオン。中身は onnxruntime-gpu と、それが使う CUDA / cuDNN の DLL だけ。
本体 (DASH) はこのアドオンが入っていれば、同梱の onnxruntime (DirectML 版) の代わりにこちらを使う。

元: tools/ort_cuda11 (pip install --target で隔離して入れたもの)
  - onnxruntime-gpu 1.20.1 の CUDA 11 版 (Microsoft の onnxruntime-cuda-11 フィード。CUDA 11 版で cp313 があるのは 1.20 系だけ)
  - CUDA 11.8 / cuBLAS 11 / cuFFT 10 / cuDNN 8.9.5 (nvidia-*-cu11) + zlibwapi.dll (cuDNN 8 の依存。winimage の zlib123dllx64)
VSU と同じ「古い ORT + cuDNN 8」の組み合わせ。cuDNN 9 (CUDA 12 版、ORT 1.20.2 / 1.24.4) ではこのモデルの
1 次元畳み込みが cuDNN の新しい API に載らず Fallback mode になり、デコーダが 3〜4 倍遅い (log.txt)。
CUDA 11.8 は RTX 50 シリーズ (Blackwell) には対応していない。
必要な DLL は実測で確認 (scripts/bench_cuda.py の CUDA_PACK=)。学習用 (*_train)・RNN 用 (adv)・TensorRT は不要。
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "ort_cuda11"
STAGE = ROOT / "build" / "aivisCudaPack"
DIST = ROOT / "dist"
VERSION = "0.1.0"

DLLS = {
    "cublas": ["cublas64_11.dll", "cublasLt64_11.dll"],
    "cufft": ["cufft64_10.dll"],
    "cuda_runtime": ["cudart64_110.dll"],
    "cudnn": ["cudnn64_8.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll", "zlibwapi.dll"],
}
LICENSES = {
    "cublas": "nvidia_cublas_cu11-*.dist-info",
    "cufft": "nvidia_cufft_cu11-*.dist-info",
    "cuda_runtime": "nvidia_cuda_runtime_cu11-*.dist-info",
    "cudnn": "nvidia_cudnn_cu11-*.dist-info",
}

MANIFEST = f'''name = aivisCudaPack
summary = "DASH CUDA pack"
description = """DASH を NVIDIA GPU (CUDA) で動かすための追加パックです。
onnxruntime-gpu と CUDA / cuDNN のライブラリが入っています。DASH の設定で「GPU を使う」をオンにしてください。"""
author = "yamahubuki, ACT Laboratory"
url = None
version = {VERSION}
minimumNVDAVersion = 2026.1
lastTestedNVDAVersion = 2026.1
updateChannel = None
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    cuda = STAGE / "cuda"
    shutil.copytree(
        SRC / "onnxruntime", cuda / "onnxruntime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "onnxruntime_providers_tensorrt.dll", "test", "tools",
                                      "transformers", "quantization", "*.lib", "*.pdb"),
    )
    # onnxruntime の preload_dlls() が CUDA のバージョンを調べるのに dist-info を見る
    for di in SRC.glob("onnxruntime_gpu-*.dist-info"):
        shutil.copytree(di, cuda / di.name)
    for pkg, names in DLLS.items():
        dst = cuda / "nvidia" / pkg / "bin"
        dst.mkdir(parents=True)
        for n in names:
            shutil.copy2(SRC / "nvidia" / pkg / "bin" / n, dst / n)
        for di in SRC.glob(LICENSES[pkg]):
            for lic in di.iterdir():
                if lic.is_file() and lic.name.lower().startswith("license"):
                    shutil.copy2(lic, cuda / "nvidia" / pkg / lic.name)
            lic_dir = di / "licenses"
            if lic_dir.is_dir():
                shutil.copytree(lic_dir, cuda / "nvidia" / pkg / "licenses")
    (STAGE / "manifest.ini").write_text(MANIFEST, encoding="utf-8")
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged: {STAGE}  ({size / 2**20:.0f} MB)")
    if args.stage_only:
        return
    DIST.mkdir(exist_ok=True)
    out = DIST / f"aivisCudaPack-{VERSION}.nvda-addon"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    print(f"built: {out}  ({out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
