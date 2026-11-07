# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""NVDA アドオン (.nvda-addon) を組み立てる。

  tools\\py313\\python.exe -I scripts\\build_addon.py [--stage-only] [--with-aivisspeech-dictionaries]

同梱するサードパーティのファイルは、すべて配布元からダウンロードする (scripts/third_party.py、VSU と同じ考え方)。
sha256 は scripts/third_party_pins.json に固定してあり、ダウンロードしたものは build/cache/ に残して使い回す。

  synthDrivers/_aivis/
    aivis_core/        音声合成コア (このリポジトリ)
    deps/              Python の依存ライブラリ (PyPI の wheel を展開)、e2k (GitHub)
    deps/_stdlib/      NVDA に無い標準ライブラリ (python.org の埋め込み版から)
    data/bert/         BERT (Hugging Face, CC BY-SA 4.0)
    data/models/       音声合成モデル「まお」(旧 Anneli, AivisHub, ACML 1.0)
    dictionaries/      AivisSpeech Engine の既定辞書 (--with-aivisspeech-dictionaries のときだけ)
  doc/ja/third_party.html  同梱物とライセンスの一覧
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# GitHub Actions の Windows では標準出力が cp1252 になり、日本語の表示で UnicodeEncodeError になる。UTF-8 にそろえる
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
STAGE = ROOT / "build" / "addon"
OUTPUT = ROOT / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import third_party as tp  # noqa: E402

# 実行に不要で大きいもの (wheel から展開するときに除く)
EXCLUDE = [
    "*/__pycache__/*", "*.pyc", "*/tests/*", "*/test/*",
    "onnx/backend/test/*", "onnx/reference/*", "onnx/defs/*.md",
    "onnxruntime/transformers/*", "onnxruntime/tools/*", "onnxruntime/quantization/*", "onnxruntime/datasets/*",
    "numpy/_core/tests/*", "numpy/f2py/*", "numpy/distutils/*", "numpy/testing/_private/extbuild.py",
    "*.pyi", "*.h", "*.lib", "*.data/scripts/*",
]  # fmt: skip
STDLIB_FILES = ["python313.zip", "_zoneinfo.pyd", "_sqlite3.pyd", "sqlite3.dll", "LICENSE.txt"]


def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE)


def extract_wheel(whl: Path, dest: Path) -> None:
    with zipfile.ZipFile(whl) as z:
        for info in z.infolist():
            if info.is_dir() or _excluded(info.filename):
                continue
            out = dest / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)


def load_build_vars():
    """リポジトリ直下の buildVars.py を読む (tools/bumpup.py がスナップショットのビルドで書き換えることがあるので毎回読む)。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("buildVars", ROOT / "buildVars.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_manifest() -> str:
    """manifest.ini.tpl と buildVars.py から manifest.ini を作る (VSU の sconstruct と同じ)。"""
    bv = load_build_vars()
    info = dict(bv.addon_info)
    info["addon_version"] = bv.ADDON_VERSION
    return (ROOT / "manifest.ini.tpl").read_text(encoding="utf-8").format(**info)


def build_translations() -> None:
    """addon/locale/<lang>/LC_MESSAGES/nvda.po → nvda.mo と、訳したアドオンの説明 locale/<lang>/manifest.ini
    (VSU の sconstruct と manifest-translated.ini.tpl と同じ)。訳の無い文言があれば警告する。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import i18n

    if i18n.check():
        print("WARNING: some messages are not translated (run scripts/i18n.py update and fill in the .po)")
    info = load_build_vars().addon_info
    for po in i18n.po_files():
        lang_dir = STAGE / "locale" / po.parent.parent.name
        n = i18n.compile_mo(po, lang_dir / "LC_MESSAGES" / "nvda.mo")
        (lang_dir / "LC_MESSAGES" / "nvda.po").unlink()  # 配布物には .mo だけを入れる
        summary = i18n.translate(po, info["addon_summary"])
        description = i18n.translate(po, info["addon_description"])
        (lang_dir / "manifest.ini").write_text(
            f'summary = "{summary}"\ndescription = """{description}"""\n', encoding="utf-8")
        print(f"translation {po.parent.parent.name}: {n} messages")


def _varint(f) -> int:
    shift = result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        result |= (b[0] & 0x7F) << shift
        if not b[0] & 0x80:
            return result
        shift += 7


def aivmx_manifest(path: Path) -> dict:
    """AIVMX (ONNX) の metadata_props から aivm_manifest を読む。

    ビルド環境 (GitHub Actions) に onnx / onnxruntime を入れずに済むよう、ONNX の ModelProto の最上位だけを
    protobuf の形式どおりに読む (metadata_props はフィールド 14、各要素の key が 1・value が 2)。
    """
    import json

    with open(path, "rb") as f:
        size = path.stat().st_size
        while f.tell() < size:
            key = _varint(f)
            field, wire = key >> 3, key & 7
            if wire == 0:
                _varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                n = _varint(f)
                if field != 14:
                    f.seek(n, 1)
                    continue
                entry = io.BytesIO(f.read(n))
                kv = {}
                while entry.tell() < n:
                    k = _varint(entry)
                    ln = _varint(entry)
                    kv[k >> 3] = entry.read(ln).decode("utf-8")
                if kv.get(1) == "aivm_manifest":
                    return json.loads(kv[2])
            else:
                raise ValueError(f"unexpected wire type {wire} in {path}")
    raise ValueError(f"aivm_manifest not found in {path}")


def md_to_html(md: Path, html: Path, title: str = "DASH", lang: str = "ja") -> None:
    import markdown

    body = markdown.markdown(md.read_text(encoding="utf-8"), extensions=["extra"])
    html.write_text(
        f'<!DOCTYPE html>\n<html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title></head>\n'
        f"<body>\n{body}\n</body></html>\n",
        encoding="utf-8",
    )


def write_third_party_doc(path: Path, with_dictionaries: bool, notices: list[tuple[str, str]] | None = None) -> None:
    rows = tp.license_table(with_dictionaries)
    lines = [
        "# 同梱しているもの・ライセンス",
        "",
        "このアドオンには次のサードパーティのソフトウェア・データが含まれています。"
        "ライセンス文書は各ファイルの近く (`synthDrivers/_aivis/` 以下) に同梱しています。",
        "",
        "| 名前 | ライセンス | 入手元 |",
        "|---|---|---|",
    ]
    for name, lic, url in rows:
        lines.append(f"| {name} | {lic} | {url.split('?')[0]} |")
    lines += [
        "",
        "## このアドオン自身のコード",
        "",
        "- このアドオンは GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) で配布しています"
        " (アドオンのフォルダの `LICENSE.txt`)。",
        "- `synthDrivers/_aivis/aivis_core/nlp/` の日本語処理は Style-Bert-VITS2 (tsukumijima フォーク) から取り込んだもので、"
        "GNU AGPL v3.0 (AGPL-3.0-only) のままです (`aivis_core/LICENSE_style_bert_vits2`)。"
        "著作権者: tsukumijima, litagin02 (Style-Bert-VITS2, https://github.com/tsukumijima/Style-Bert-VITS2)、"
        "Bert-VITS2 の作者たち (https://github.com/fishaudio/Bert-VITS2)。各ファイルの先頭に出典と改変の有無を書いています。"
        "`mora_list.py` は元は VOICEVOX ENGINE 由来 (LGPL v3.0, `aivis_core/LGPL_LICENSE`)、"
        "その中のモーラ表は Open JTalk (修正 BSD, Copyright (c) 2008-2014 Nagoya Institute of Technology) 由来です。",
        "",
        "## BERT (CC BY-SA 4.0)",
        "",
        "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx "
        "(https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx) は、"
        "京都大学 黒橋・褚・村脇研究室の ku-nlp/deberta-v2-large-japanese-char-wwm "
        "(https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) を ONNX に変換したものです。"
        "改変せずにそのまま同梱しています。ライセンス: https://creativecommons.org/licenses/by-sa/4.0/",
        "",
        "## 音声合成モデル",
        "",
        "同梱のモデルのライセンス文書 (ACML 1.0) は `synthDrivers/_aivis/data/models/` にあります。",
    ]
    # バイナリ配布の条件として、ドキュメントに著作権表示と条文を載せる必要があるもの (BSD・CC BY)
    if notices:
        lines += ["", "## ライセンス条文", ""]
        for title, text in notices:
            lines += [f"### {title}", "", "```", text.strip("\n"), "```", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--output", help="出力する .nvda-addon のパス (既定: output/<ADDON_NAME>-<ADDON_VERSION>.nvda-addon)")
    ap.add_argument("--with-aivisspeech-dictionaries", action="store_true",
                    help="AivisSpeech Engine の既定辞書 (226MB) も同梱する。辞書ごとの出典・ライセンスは未確認")
    args = ap.parse_args()
    with_dic = args.with_aivisspeech_dictionaries

    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(ROOT / "addon", STAGE, ignore=shutil.ignore_patterns("__pycache__"))
    pkg = STAGE / "synthDrivers" / "_aivis"
    shutil.copytree(ROOT / "aivis_core", pkg / "aivis_core", ignore=shutil.ignore_patterns("__pycache__"))

    print("downloading / verifying third-party files ...")
    deps = pkg / "deps"
    deps.mkdir()
    for key in tp.WHEELS:
        extract_wheel(tp.fetch(key), deps)
    # e2k (GitHub のアーカイブの src/e2k)
    with zipfile.ZipFile(tp.fetch("e2k")) as z:
        for info in z.infolist():
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) > 1 else ""
            if info.is_dir() or not (rel.startswith("src/e2k/") or rel == "LICENSE") or rel.endswith(".gitkeep"):
                continue
            out = deps / ("e2k/LICENSE" if rel == "LICENSE" else rel[len("src/"):])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
    # DirectML.dll のライセンス文書
    dml = tp.fetch("directml-license")
    for name in ("LICENSE.txt", "ThirdPartyNotices.txt"):
        shutil.copy2(dml / name, deps / "onnxruntime" / f"DirectML_{name}")
    # pyopenjtalk-plus に組み込まれている Open JTalk / hts_engine API / MeCab のライセンス文書 (wheel に無い)
    notices: list[tuple[str, str]] = []
    for key, title in (("notice-open_jtalk", "Open JTalk"), ("notice-hts_engine", "hts_engine API"),
                       ("notice-mecab", "MeCab (Open JTalk に内蔵)")):  # fmt: skip
        p = tp.fetch(key)
        shutil.copy2(p, deps / "pyopenjtalk" / p.name)
        notices.append((title, p.read_text(encoding="utf-8")))
    # wheel に含まれているもの: 辞書 (naist-jdic) と HTS 音声「メイ」(このアドオンでは使わないが wheel に含まれる)
    notices.append(("NAIST Japanese Dictionary (pyopenjtalk/dictionary)",
                    (deps / "pyopenjtalk" / "dictionary" / "COPYING").read_text(encoding="utf-8")))
    notices.append(("HTS Voice \"Mei\" (pyopenjtalk/htsvoice, このアドオンでは使用していません)",
                    (deps / "pyopenjtalk" / "htsvoice" / "LICENSE_mei_normal.htsvoice").read_text(encoding="utf-8")))

    # NVDA (py2exe) は NVDA 自身が使う標準ライブラリしか持っていない (例: zoneinfo が無く pydantic が落ちる)。
    # 同じ 3.13 の標準ライブラリ一式を同梱し、sys.path の末尾に足す (NVDA 側にあるものはそちらが優先される)
    stdlib = deps / "_stdlib"
    stdlib.mkdir()
    with zipfile.ZipFile(tp.fetch("python-embed")) as z:
        for name in STDLIB_FILES:
            (stdlib / name).write_bytes(z.read(name))

    # tsqyomi (文脈による読み分けモデル, MIT, Aivis Project)。pyopenjtalk-plus が使う v6 を同梱し、model_dir で読む
    tsq = pkg / "data" / "tsqyomi"
    tsq.mkdir(parents=True)
    for key in ("tsqyomi-model", "tsqyomi-tokenizer", "tsqyomi-metadata", "tsqyomi-license"):
        p = tp.fetch(key)
        shutil.copy2(p, tsq / p.name)
    notices.append(("tsqyomi (synthDrivers/_aivis/data/tsqyomi)", (tsq / "LICENSE.txt").read_text(encoding="utf-8")))

    # BERT と音声合成モデル (動作に最低限必要なもの)
    bert = pkg / "data" / "bert"
    bert.mkdir(parents=True)
    shutil.copy2(tp.fetch("bert-model"), bert / "model.onnx")
    shutil.copy2(tp.fetch("bert-vocab"), bert / "vocab.txt")
    (bert / "LICENSE.md").write_text(
        "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx (revision " + tp.BERT_REV + ")\n"
        "https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx\n\n"
        "Original model: ku-nlp/deberta-v2-large-japanese-char-wwm\n"
        "https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm\n\n"
        "License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)\n"
        "https://creativecommons.org/licenses/by-sa/4.0/\n\n"
        "The files are redistributed without modification.\n",
        encoding="utf-8",
    )
    models = pkg / "data" / "models"
    models.mkdir(parents=True)
    model = tp.fetch("anneli")
    shutil.copy2(model, models / model.name)
    # ACML は「配布するときはライセンス文書を添付」が条件。モデルに埋め込まれた文書をそのまま書き出す
    manifest = aivmx_manifest(model)
    (models / f"{model.stem}_LICENSE.md").write_text(
        f"# {manifest['name']} (version {manifest['version']}, {', '.join(manifest.get('creators', []))})\n\n"
        + manifest["license"], encoding="utf-8")

    if with_dic:
        (pkg / "dictionaries").mkdir()
        for key in tp.aivisspeech_dictionaries():
            p = tp.fetch(key)
            shutil.copy2(p, pkg / "dictionaries" / p.name)
        shutil.copy2(ROOT / "ref" / "AivisSpeech-Engine" / "LICENSE", pkg / "dictionaries" / "LICENSE_AivisSpeech-Engine")

    shutil.copy2(ROOT / "LICENSE", STAGE / "LICENSE.txt")  # このアドオンのライセンス (AGPL-3.0)
    (STAGE / "manifest.ini").write_text(make_manifest(), encoding="utf-8")
    build_translations()
    md_to_html(STAGE / "doc" / "ja" / "readme.md", STAGE / "doc" / "ja" / "readme.html")
    # 英語版 (日本語以外の NVDA はこれを開く。同梱物の一覧 third_party.html は日本語版だけで、doc/ja を参照する)
    md_to_html(STAGE / "doc" / "en" / "readme.md", STAGE / "doc" / "en" / "readme.html", lang="en")
    write_third_party_doc(STAGE / "doc" / "ja" / "third_party.md", with_dic, notices)
    md_to_html(STAGE / "doc" / "ja" / "third_party.md", STAGE / "doc" / "ja" / "third_party.html", "同梱物とライセンス")

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged: {STAGE}  ({total / 1e6:.0f} MB)")
    if args.stage_only:
        return

    bv = load_build_vars()
    # VSU (sconstruct) と同じく「output/<ADDON_NAME>-<ADDON_VERSION>.nvda-addon」
    out = Path(args.output) if args.output else OUTPUT / f"{bv.ADDON_NAME}-{bv.ADDON_VERSION}.nvda-addon"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    print(f"built: {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
