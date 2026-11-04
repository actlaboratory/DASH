# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""Python ファイルの先頭にライセンスのヘッダー (SPDX と著作権表示) を入れる。

- このプロジェクトで書いたファイル: AGPL-3.0-or-later、Copyright yamahubuki, ACT Laboratory
- Style-Bert-VITS2 から取り込んだファイル: Style-Bert-VITS2 は AGPL-3.0-only (pyproject.toml) なので
  AGPL-3.0-only のまま。元の著作権者 (git の履歴の作者) と出典を書き、改変したファイルには
  AGPL-3.0 第 5 条 a に従って「改変したこと」と日付を書く (ref/sbv2 の元ファイルと比べて判定)
- 一部を流用して書き直したファイル: AGPL-3.0-only、自分たちと元の著作権者の両方を書く

すでにヘッダーがあるファイルは、このスクリプトが付けたヘッダー部分だけを書き直す (元からあるコメントは残す)。

  tools\\py313\\python.exe -I scripts\\add_license_headers.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = "yamahubuki, ACT Laboratory"
SBV2_URL = "https://github.com/tsukumijima/Style-Bert-VITS2"
SBV2_COMMIT = "b937361"
SBV2_SRC = ROOT / "ref" / "sbv2" / "style_bert_vits2"

# 取り込んだファイル: (元のファイル, 元の著作権表示)。著作権者は ref/sbv2 の git log の作者から
# (nlp/__init__.py と nlp/japanese/__init__.py は中身が空なので自作扱い)
SBV2_AUTHORS = "tsukumijima, litagin02 (Style-Bert-VITS2)"
VENDORED = {
    "aivis_core/nlp/symbols.py": (
        "nlp/symbols.py", f"2023-2026 Bert-VITS2 contributors (fishaudio/Bert-VITS2), {SBV2_AUTHORS}"),
    "aivis_core/nlp/nanairo_emoji.py": ("nlp/nanairo_emoji.py", "2026 tsukumijima (Style-Bert-VITS2)"),
    "aivis_core/nlp/japanese/g2p.py": ("nlp/japanese/g2p.py", f"2024-2026 {SBV2_AUTHORS}"),
    "aivis_core/nlp/japanese/g2p_utils.py": ("nlp/japanese/g2p_utils.py", f"2024-2026 {SBV2_AUTHORS}"),
    "aivis_core/nlp/japanese/itaiji_map.py": ("nlp/japanese/itaiji_map.py", "2026 tsukumijima (Style-Bert-VITS2)"),
    "aivis_core/nlp/japanese/katakana_map.py": (
        "nlp/japanese/katakana_map.py", "2024-2026 tsukumijima (Style-Bert-VITS2)"),
    "aivis_core/nlp/japanese/mora_list.py": ("nlp/japanese/mora_list.py", f"2024-2026 {SBV2_AUTHORS}"),
    "aivis_core/nlp/japanese/normalizer.py": ("nlp/japanese/normalizer.py", f"2024-2026 {SBV2_AUTHORS}"),
}  # fmt: skip
EXTRA_NOTE = {
    "aivis_core/nlp/japanese/mora_list.py": [
        "# Style-Bert-VITS2 took this file from VOICEVOX ENGINE (LGPL-3.0), and the mora table from Open JTalk",
        "# (Modified BSD, Copyright (c) 2008-2014 Nagoya Institute of Technology; see the notice below).",
    ],
}
# 一部を流用して書き直したもの: 元にしたファイル
DERIVED = {
    "aivis_core/nlp/bert_models.py": "nlp/bert_models.py",
    "aivis_core/frontend.py": "nlp/__init__.py, models/infer_onnx.py",
}
# VSU (同じ作者の NVDA アドオン, GPL-2.0-or-later) から流用したもの: VSU での元のファイル名
FROM_VSU = {
    "addon/globalPlugins/DASH/updater.py": "updater.py",
    "addon/globalPlugins/DASH/compat.py": "compat.py",
    "addon/globalPlugins/DASH/constants.py": "constants.py",
    "addon/globalPlugins/DASH/updaterStrings.py": "updaterStrings.py",
}
# このスクリプトが付けたヘッダーの行 (書き直すときに取り除く)
OUR_PREFIXES = (
    "# SPDX-License-Identifier:", "# Copyright (C) 2026 ", "# This file is taken from", "# licensed under the GNU",
    "# (relative imports, logging)", "# Style-Bert-VITS2 in turn", "# Based in part on", "# Copyright (C) 20",
    "# Portions Copyright", "# Modified by ", "# Unmodified", "# Style-Bert-VITS2 took this file",
    "# (Modified BSD, Copyright", "# Taken from VSU",
)  # fmt: skip


def _body(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith(OUR_PREFIXES):
        i += 1
    return lines[i:]


def header(rel: str, body: list[str]) -> list[str]:
    if rel in FROM_VSU:
        return [
            "# SPDX-License-Identifier: AGPL-3.0-or-later",
            f"# Copyright (C) 2023-2026 {OURS}",
            f"# Taken from VSU (globalPlugins/VSU/{FROM_VSU[rel]}).",
        ]
    if rel in VENDORED:
        src, holders = VENDORED[rel]
        orig = (SBV2_SRC / src).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        modified = [line.rstrip() for line in body] != [line.rstrip() for line in orig]
        lines = [
            "# SPDX-License-Identifier: AGPL-3.0-only",
            f"# Copyright (C) {holders}",
            f"# This file is taken from Style-Bert-VITS2 ({SBV2_URL}, commit {SBV2_COMMIT}, style_bert_vits2/{src}),",
            "# licensed under the GNU Affero General Public License v3.0.",
        ]
        lines += EXTRA_NOTE.get(rel, [])
        if modified:
            lines.append(f"# Modified by {OURS}, 2026, for DASH (import paths, logging, pydantic -> dataclass).")
        else:
            lines.append("# Unmodified from the original.")
        return lines
    if rel in DERIVED:
        return [
            "# SPDX-License-Identifier: AGPL-3.0-only",
            f"# Copyright (C) 2026 {OURS}",
            f"# Portions Copyright (C) 2024-2026 {SBV2_AUTHORS}",
            f"# Based in part on Style-Bert-VITS2 ({SBV2_URL}, commit {SBV2_COMMIT}, style_bert_vits2/{DERIVED[rel]}),",
            "# licensed under the GNU Affero General Public License v3.0.",
        ]
    return ["# SPDX-License-Identifier: AGPL-3.0-or-later", f"# Copyright (C) 2026 {OURS}"]


def main() -> None:
    files = sorted(
        p for d in ("aivis_core", "addon", "scripts") for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts
    )
    changed = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        bom = text.startswith("﻿")
        if bom:
            text = text[1:]
        nl = "\r\n" if "\r\n" in text[:2000] else "\n"
        body = _body(text)
        new = nl.join(header(rel, body) + body)
        out = (("﻿" if bom else "") + new).encode("utf-8")
        if out != raw:
            p.write_bytes(out)
            changed += 1
            kind = "vendored" if rel in VENDORED else "derived " if rel in DERIVED else "own     "
            print(f"{kind} {rel}")
    print(f"updated {changed} files")


if __name__ == "__main__":
    main()
