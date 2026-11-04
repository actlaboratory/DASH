# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AIVMX (ONNX) のメタデータを、重い依存 (onnx / numpy / onnxruntime) なしで読む。

onnx.load は重み (約 250MB) まで読み込んで解析するので、モデルが増えると声の一覧を作るだけで遅くなる
(1 モデル 210〜250ms → このモジュールで 2〜37ms, 内容は同一)。
ModelProto の最上位のフィールドを protobuf の形式どおりにたどり、metadata_props (フィールド 14、
各要素は key=1 / value=2) 以外の長いフィールド (graph など) は読まずに飛ばす。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any


def _read_varint(f) -> int:
    shift = result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("unexpected end of file")
        result |= (b[0] & 0x7F) << shift
        if not b[0] & 0x80:
            return result
        shift += 7


def read_onnx_metadata(path: Path) -> dict[str, str]:
    """ONNX の metadata_props を {key: value} で返す。"""
    props: dict[str, str] = {}
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        while f.tell() < size:
            key = _read_varint(f)
            field, wire = key >> 3, key & 7
            if wire == 0:
                _read_varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                n = _read_varint(f)
                if field != 14:
                    f.seek(n, 1)
                    continue
                entry = io.BytesIO(f.read(n))
                kv: dict[int, str] = {}
                while entry.tell() < n:
                    k = _read_varint(entry)
                    ln = _read_varint(entry)
                    kv[k >> 3] = entry.read(ln).decode("utf-8")
                if 1 in kv:
                    props[kv[1]] = kv.get(2, "")
            else:
                raise ValueError(f"unexpected protobuf wire type {wire} in {path}")
    return props


def read_aivm_manifest(path: Path) -> dict[str, Any]:
    """AIVMX の AIVM マニフェスト (名前・UUID・話者・スタイル・ライセンスなど)。"""
    return json.loads(read_onnx_metadata(path)["aivm_manifest"])
