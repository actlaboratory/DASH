# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""アドオンに同梱するサードパーティのファイルを、配布元からダウンロードする (VSU の sconstruct と同じ考え方)。

  tools\\py313\\python.exe -I scripts\\third_party.py --update-pins   # 版を変えたとき: URL と sha256 を調べて固定する
  (ふだんは build_addon.py が fetch() を呼ぶ。ダウンロードしたものは build/cache/ に置き、2 回目からは使い回す)

- どのファイルも sha256 を third_party_pins.json に固定し、ダウンロード後に照合する (改ざん・破損・版ずれの検出)
- PyPI の wheel は PyPI が公開している sha256、Hugging Face の LFS ファイルは LFS の oid (= sha256)、
  AivisHub のモデルは API の checksum を使う。それ以外は --update-pins のときに一度ダウンロードして計算する
- ライセンスは THIRD_PARTY の "license" に記録し、build_addon.py がアドオンの doc/ja/third_party.md に一覧を書く
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "build" / "cache"
PINS = Path(__file__).resolve().parent / "third_party_pins.json"
sys.path.insert(0, str(ROOT / "addon" / "synthDrivers" / "_aivis"))
from remote_zip import HttpRangeFile, RangeNotSupported  # noqa: E402

UA = {"User-Agent": "aivis-stream-build"}
HF_BERT = "https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx/resolve"
BERT_REV = "d701ec67708287b20d2063270f6b535e6eed09ab"  # AivisSpeech Engine と同じリビジョン (動作確認済み)
ANNELI_UUID = "a59cb814-0083-4369-8542-f51a29e72af7"
AIVISSPEECH_REV = "ab707f2c32e7d252bce2b4027aa7a49dd6862f1e"

# Python の依存ライブラリ (実際に import されるもの: scripts/list_deps.py)。cp313 / win_amd64 の wheel
WHEELS = {
    "numpy": ("numpy", "2.5.4", "BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0-1.0)"),
    "onnxruntime-directml": ("onnxruntime-directml", "1.23.0", "MIT (DirectML.dll は Microsoft DirectML ライセンス)"),
    "onnx": ("onnx", "1.23.2", "Apache-2.0"),
    "ml_dtypes": ("ml_dtypes", "0.6.0", "Apache-2.0"),
    "protobuf": ("protobuf", "7.36.2", "BSD-3-Clause"),
    "pyopenjtalk-plus": ("pyopenjtalk-plus", "0.4.1.post10", "MIT (Open JTalk / HTS Engine / naist-jdic: 修正 BSD)"),
    "sudachipy": ("sudachipy", "0.7.0", "Apache-2.0"),
    "sudachidict_core": ("sudachidict_core", "20261015", "Apache-2.0"),
    "pydantic": ("pydantic", "2.14.2", "MIT"),
    "pydantic_core": ("pydantic_core", "2.50.2", "MIT"),
    "annotated_types": ("annotated_types", "0.8.0", "MIT"),
    "typing_extensions": ("typing_extensions", "4.16.0", "PSF-2.0"),
    "typing_inspection": ("typing_inspection", "0.4.4", "MIT"),
    "jaconv": ("jaconv", "0.5.0", "MIT"),
    "num2words": ("num2words", "0.5.14", "LGPL-2.1"),
    # tsqyomi (pyopenjtalk-plus の文脈による読み分け) のトークナイザー
    "tokenizers": ("tokenizers", "0.23.3", "Apache-2.0"),
}
TSQYOMI_REPO = "https://huggingface.co/tsukumijima/tsqyomi-models/resolve"
TSQYOMI_REV = "917a89d2c93420d1c43213c1e0232b25f7a6aa5f"  # pyopenjtalk-plus 0.4.1.post10 の _MODEL_REVISION と同じ

# それ以外 (key: url, ライセンス, 説明)。zip_members があるものは、ZIP の中のそのファイルだけを取り出す
OTHERS = {
    # e2k は AivisSpeech Engine と同じ tsukumijima さんのフォーク (PyPI の 0.6.2 とは中身が違う)
    "e2k": {
        "url": "https://github.com/tsukumijima/e2k/archive/9920754e15d8c96404bef0bd81d424352fb7a48b.zip",
        "license": "Unlicense (パブリックドメイン)",
        "desc": "英単語のカタカナ読み (tsukumijima/e2k @ 9920754)",
    },
    # NVDA (py2exe) に無い標準ライブラリを補う (zoneinfo など)
    "python-embed": {
        "url": "https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-amd64.zip",
        "license": "PSF-2.0",
        "desc": "Python 3.13.9 の標準ライブラリ (python313.zip, _zoneinfo.pyd, _sqlite3.pyd, sqlite3.dll)",
    },
    "directml-license": {
        "url": "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
        "zip_members": ["LICENSE.txt", "ThirdPartyNotices.txt"],
        "license": "Microsoft DirectML ライセンス",
        "desc": "onnxruntime-directml に含まれる DirectML.dll 1.15.4 のライセンス文書 (NuGet パッケージから取り出す)",
    },
    # pyopenjtalk-plus のバイナリに組み込まれているが、wheel にライセンス文書が入っていないもの (BSD: バイナリ配布時は
    # ドキュメントに著作権表示と条文を載せる必要がある)。pyopenjtalk-plus のサブモジュールの配布元から取る
    "notice-open_jtalk": {
        "url": "https://raw.githubusercontent.com/tsukumijima/open_jtalk/80f8c72f41e6fda2f60efde199101f794be7aaab/src/COPYING",
        "license": "Modified BSD",
        "desc": "Open JTalk (pyopenjtalk-plus に組み込み) のライセンス文書",
        "filename": "COPYING-open_jtalk.txt",
    },
    "notice-hts_engine": {
        "url": "https://raw.githubusercontent.com/syoyo/hts_engine_API/2821c7a0b02441af42b642365296a35a5c9f0784/src/COPYING",
        "license": "Modified BSD",
        "desc": "hts_engine API (pyopenjtalk-plus に組み込み) のライセンス文書",
        "filename": "COPYING-hts_engine_API.txt",
    },
    "notice-mecab": {
        "url": "https://raw.githubusercontent.com/tsukumijima/open_jtalk/80f8c72f41e6fda2f60efde199101f794be7aaab/src/mecab/COPYING",
        "license": "BSD",
        "desc": "MeCab (Open JTalk に内蔵) のライセンス文書",
        "filename": "COPYING-mecab.txt",
    },
    # fp32 版を同梱する。fp16 版をメモリ上で fp32 に変換するより、読み込みが約 2.4 秒速く (4.9→2.5 秒)、
    # 読み込み中のメモリの最大が 4.4GB → 1.5GB、読み込み後も 2.6GB → 1.4GB と少ない (scripts/bench_bert_load.py)。
    # 代わりにアドオンは約 580MB 大きくなる
    "bert-model": {
        "url": f"{HF_BERT}/{BERT_REV}/model.onnx",
        "sha256": "c5c880ef4bd0d3308ec6503a8728efae920bc5c5a984de4f76fc3d0ad518a2ec",  # LFS oid
        "license": "CC BY-SA 4.0",
        "desc": "BERT (tsukumijima/deberta-v2-large-japanese-char-wwm-onnx, ku-nlp/deberta-v2-large-japanese-char-wwm の ONNX 版)",
    },
    # tsqyomi: 文脈による読み分けモデル (pyopenjtalk-plus が使う v6 だけ。Hugging Face からではなく同梱して model_dir で読む)
    "tsqyomi-model": {
        "url": f"{TSQYOMI_REPO}/{TSQYOMI_REV}/v6/model.onnx",
        "sha256": "602e630d3368fe905ace80022f235b21ca1461fcad28dfe40d073a19cfade22f",  # LFS oid
        "license": "MIT",
        "desc": "tsqyomi (文脈による読み分けモデル, tsukumijima/tsqyomi-models v6)",
        "filename": "model.onnx",
    },
    "tsqyomi-tokenizer": {
        "url": f"{TSQYOMI_REPO}/{TSQYOMI_REV}/v6/tokenizer.json",
        "license": "MIT",
        "desc": "tsqyomi のトークナイザー",
        "filename": "tokenizer.json",
    },
    "tsqyomi-metadata": {
        "url": f"{TSQYOMI_REPO}/{TSQYOMI_REV}/v6/metadata.json",
        "license": "MIT",
        "desc": "tsqyomi のメタデータ",
        "filename": "metadata.json",
    },
    "tsqyomi-license": {
        "url": f"{TSQYOMI_REPO}/{TSQYOMI_REV}/LICENSE.txt",
        "license": "MIT",
        "desc": "tsqyomi のライセンス文書",
        "filename": "LICENSE.txt",
    },
    "bert-vocab": {
        "url": f"{HF_BERT}/{BERT_REV}/vocab.txt",
        "license": "CC BY-SA 4.0",
        "desc": "BERT の語彙",
    },
    "anneli": {
        "url": f"https://api.aivis-project.com/v1/aivm-models/{ANNELI_UUID}/download?model_type=AIVMX",
        "license": "Aivis Common Model License (ACML) 1.0",
        "desc": "音声合成モデル「まお」(旧名 Anneli, Aivis Project)。AivisSpeech の標準モデル",
        "filename": f"{ANNELI_UUID}.aivmx",
    },
}


def aivisspeech_dictionaries() -> dict:
    """AivisSpeech Engine の既定辞書 (任意: --with-aivisspeech-dictionaries)。ファイル名は ref/ の一覧から。"""
    names = sorted(p.name for p in (ROOT / "ref" / "AivisSpeech-Engine" / "resources" / "dictionaries").glob("*.dic"))
    base = f"https://raw.githubusercontent.com/Aivis-Project/AivisSpeech-Engine/{AIVISSPEECH_REV}/resources/dictionaries"
    return {
        f"dic-{n}": {"url": f"{base}/{n}", "license": "AivisSpeech Engine (LGPL-3.0) 同梱の辞書。出典ごとのライセンスは未確認",
                     "desc": f"AivisSpeech Engine の既定辞書 {n}"}
        for n in names
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
            done += len(b)
            if total > 50 << 20 and done % (100 << 20) < (1 << 20):
                print(f"    {done / 2**20:.0f} / {total / 2**20:.0f} MB", flush=True)
    tmp.replace(dest)


def _extract_members(url: str, members: list[str], dest_dir: Path) -> None:
    """ZIP の中の members だけを取り出す (Range 対応なら必要な部分だけダウンロードする)。"""
    try:
        z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20))
    except RangeNotSupported:
        whole = dest_dir / "_whole.zip"
        _download(url, whole)
        z = zipfile.ZipFile(whole)
    with z:
        for m in members:
            (dest_dir / Path(m).name).write_bytes(z.read(m))
    (dest_dir / "_whole.zip").unlink(missing_ok=True)


def _pypi_wheel(name: str, version: str) -> dict:
    data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60))
    cands = [f for f in data["urls"] if f["packagetype"] == "bdist_wheel"
             and (f["filename"].endswith("-cp313-cp313-win_amd64.whl") or f["filename"].endswith("-none-any.whl")
                  or f["filename"].endswith("-abi3-win_amd64.whl"))]
    if not cands:
        raise SystemExit(f"no cp313 win_amd64 wheel for {name}=={version}")
    cands.sort(key=lambda f: ("cp313-win_amd64" not in f["filename"], "abi3" not in f["filename"]))
    f = cands[0]
    return {"url": f["url"], "sha256": f["digests"]["sha256"], "filename": f["filename"], "size": f["size"]}


def all_sources(with_dictionaries: bool = False) -> dict:
    srcs = {k: dict(v) for k, v in OTHERS.items()}
    if with_dictionaries:
        srcs.update(aivisspeech_dictionaries())
    return srcs


def update_pins(with_dictionaries: bool) -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8")) if PINS.exists() else {}
    for key, (name, ver, _lic) in WHEELS.items():
        pins[key] = _pypi_wheel(name, ver)
        print(f"{key}: {pins[key]['filename']}")
    for key, spec in all_sources(with_dictionaries).items():
        entry = {"url": spec["url"]}
        if key == "anneli":
            meta = json.load(urllib.request.urlopen(
                urllib.request.Request(f"https://api.aivis-project.com/v1/aivm-models/{ANNELI_UUID}", headers=UA), timeout=60))
            f = next(f for f in meta["model_files"] if f["model_type"] == "AIVMX")
            entry.update(sha256=f["checksum"], size=f["file_size"], version=f["version"])
        elif "sha256" in spec:
            entry["sha256"] = spec["sha256"]
        # sha256 が公開されていないものは、ここで一度ダウンロードして計算する (キャッシュにも残る)
        if "sha256" not in entry or "zip_members" in spec:
            pins[key] = entry
            path = fetch(key, spec, pins, verify=False)
            if "zip_members" in spec:
                entry["members"] = {m: _sha256(path / Path(m).name) for m in spec["zip_members"]}
            else:
                entry["sha256"] = _sha256(path)
        pins[key] = entry
        print(f"{key}: {entry.get('sha256') or entry.get('members')}")
    PINS.write_text(json.dumps(pins, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {PINS}")


def fetch(key: str, spec: dict | None = None, pins: dict | None = None, verify: bool = True) -> Path:
    """key のファイルをダウンロード (キャッシュ済みならそれを使う) して、照合したパスを返す。

    zip_members のものはディレクトリを返す。
    """
    pins = pins if pins is not None else json.loads(PINS.read_text(encoding="utf-8"))
    if key not in pins:
        raise SystemExit(f"{key} の sha256 が固定されていません。scripts/third_party.py --update-pins を実行してください")
    pin = pins[key]
    spec = spec or (all_sources(True).get(key) or {})
    if "zip_members" in spec:
        d = CACHE / key
        ok = d.is_dir() and verify and all(
            (d / Path(m).name).is_file() and _sha256(d / Path(m).name) == h for m, h in pin.get("members", {}).items())
        if not ok:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            print(f"  downloading {key} (必要なファイルだけ) ...", flush=True)
            _extract_members(pin["url"], spec["zip_members"], d)
            if verify:
                for m, h in pin["members"].items():
                    if _sha256(d / Path(m).name) != h:
                        raise SystemExit(f"sha256 mismatch: {key}/{m}")
        return d
    filename = pin.get("filename") or spec.get("filename") or pin["url"].split("?")[0].rsplit("/", 1)[-1]
    path = CACHE / key / filename
    if path.is_file() and (not verify or _sha256(path) == pin["sha256"]):
        return path
    print(f"  downloading {key}: {pin['url'][:100]}", flush=True)
    _download(pin["url"], path)
    if verify and _sha256(path) != pin["sha256"]:
        bad = path.with_name(path.name + ".bad")
        path.replace(bad)
        raise SystemExit(f"sha256 mismatch: {key} ({bad})")
    return path


def license_table(with_dictionaries: bool) -> list[tuple[str, str, str]]:
    """(名前, ライセンス, 説明・出典) の一覧。"""
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    rows = [(f"{name} {ver}", lic, pins.get(key, {}).get("url", "")) for key, (name, ver, lic) in WHEELS.items()]
    for key, spec in all_sources(with_dictionaries).items():
        rows.append((spec["desc"], spec["license"], spec["url"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-pins", action="store_true")
    ap.add_argument("--with-aivisspeech-dictionaries", action="store_true")
    a = ap.parse_args()
    if a.update_pins:
        update_pins(a.with_aivisspeech_dictionaries)
    else:
        for k in list(WHEELS) + list(all_sources(a.with_aivisspeech_dictionaries)):
            print(k, fetch(k))
