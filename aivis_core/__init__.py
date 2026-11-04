# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AIVIS Core: AivisSpeech (Style-Bert-VITS2 / AIVMX) のエンジン非依存・ストリーミング対応音声合成コア。

engine (numpy / onnxruntime を使う) は、AivisCore などに初めて触れたときに読み込む (PEP 562)。
aivis_core.aivmx (モデルのメタデータを読むだけ) を、重い依存なしで使えるようにするため。
"""

__all__ = ["AivisCore", "AivisModel", "StreamConfig", "SynthParams", "StyleInfo"]


def __getattr__(name):
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
