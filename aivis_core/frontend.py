# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 yamahubuki, ACT Laboratory
# Portions Copyright (C) 2024-2026 tsukumijima, litagin02 (Style-Bert-VITS2)
# Based in part on Style-Bert-VITS2 (https://github.com/tsukumijima/Style-Bert-VITS2, commit b937361, style_bert_vits2/nlp/__init__.py, models/infer_onnx.py),
# licensed under the GNU Affero General Public License v3.0.
"""テキスト → (音素 ID, トーン, 言語 ID, BERT 特徴量)。

style_bert_vits2.models.infer_onnx.get_text_onnx() / nlp.clean_text_with_given_phone_tone() の
日本語 (JP-Extra, Nanairo 非対応) 経路だけを切り出したもの。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .nlp import bert_models
from .nlp.japanese.g2p import g2p
from .nlp.japanese.normalizer import normalize_text
from .nlp.nanairo_emoji import (
    contains_nanairo_emoji_symbols,
    normalize_nanairo_emoji_text,
    strip_nanairo_emoji_symbols,
)
from .nlp.symbols import LANGUAGE_ID_MAP, LANGUAGE_TONE_START_MAP, SYMBOLS

_SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

# g2p では対応しているが現行モデルにはない音素のフォールバック
_PHONE_CONVERSION_MAP = {
    "kw": ("k", "u", "w"),
    "gw": ("g", "u", "w"),
    "fy": ("hy",),
}


@dataclass
class TextFeatures:
    phones: np.ndarray  # (T,) int64
    tones: np.ndarray  # (T,) int64
    languages: np.ndarray  # (T,) int64
    bert: np.ndarray  # (1024, T) float32
    norm_text: str


def _convert_unsupported_phones(phone: list[str], tone: list[int], word2ph: list[int]) -> None:
    for idx in reversed([i for i, p in enumerate(phone) if p in _PHONE_CONVERSION_MAP]):
        converted = _PHONE_CONVERSION_MAP[phone[idx]]
        phone[idx : idx + 1] = list(converted)
        tone[idx : idx + 1] = [tone[idx]] * len(converted)
        count = 0
        for ci, n in enumerate(word2ph):
            if count + n > idx:
                word2ph[ci] += len(converted) - 1
                break
            count += n


def _intersperse(lst: list[int], item: int) -> list[int]:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def text_to_features(text: str, add_blank: bool = True, use_jp_extra: bool = True) -> TextFeatures | None:
    """テキストから推論用特徴量を作る。読み上げる音が無い場合は None。"""
    text = normalize_nanairo_emoji_text(text)
    if contains_nanairo_emoji_symbols(text):
        text = strip_nanairo_emoji_symbols(text)
    result = normalize_text(text.strip(), return_analysis_text=True)
    norm_text = result.text
    if not norm_text:
        return None
    # tsqyomi (文脈による読み分け) が読み込まれていれば使う (AivisSpeech Engine と同じ。engine.AivisCore.load_tsqyomi)
    from pyopenjtalk import tsqyomi

    phone, tone, word2ph, sep_text, _, _ = g2p(
        norm_text,
        analysis_text=result.analysis_text,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
        use_tsqyomi=tsqyomi.is_model_loaded(),
    )
    _convert_unsupported_phones(phone, tone, word2ph)
    # 音素が「_」だけ (= 発音すべき内容がない) なら合成しない
    if all(p in ("_",) for p in phone):
        return None

    phone_ids = [_SYMBOL_TO_ID[p] for p in phone]
    tone_ids = [t + LANGUAGE_TONE_START_MAP["JP"] for t in tone]
    lang_ids = [LANGUAGE_ID_MAP["JP"]] * len(phone_ids)
    if add_blank:
        phone_ids = _intersperse(phone_ids, 0)
        tone_ids = _intersperse(tone_ids, 0)
        lang_ids = _intersperse(lang_ids, 0)
        word2ph = [n * 2 for n in word2ph]
        word2ph[0] += 1

    bert = bert_models.extract_bert_feature("".join(sep_text), word2ph)
    assert bert.shape[-1] == len(phone_ids), (bert.shape, len(phone_ids))
    return TextFeatures(
        phones=np.array(phone_ids, dtype=np.int64),
        tones=np.array(tone_ids, dtype=np.int64),
        languages=np.array(lang_ids, dtype=np.int64),
        bert=bert,
        norm_text=norm_text,
    )


# 文の区切り。句点・感嘆符・疑問符・改行の直後で区切る (記号の連続はまとめて前の文に付ける)
_SENTENCE_SPLIT = re.compile(r"(?<=[。．！？!?\n])(?![。．！？!?」』）)\n])")
# 長すぎる文の区切り候補 (読点)
_CLAUSE_SPLIT = re.compile(r"(?<=[、，,])")


def split_sentences(text: str, max_len: int = 0, first_max_len: int = 0) -> list[str]:
    """テキストを文単位に分割する。

    max_len > 0 のとき、それより長い文は読点で更に区切る。
    first_max_len > 0 のとき、先頭の文だけはより短く (読点で) 区切って再生開始を早める。
    """
    parts = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    out: list[str] = []
    for s in parts:
        limit = first_max_len if (not out and first_max_len > 0) else max_len
        if limit > 0 and len(s) > limit:
            buf = ""
            for clause in _CLAUSE_SPLIT.split(s):
                if buf and limit > 0 and len(buf) + len(clause) > limit:
                    out.append(buf)
                    buf = ""
                    limit = max_len if max_len > 0 else 0
                buf += clause
            if buf:
                out.append(buf)
        else:
            out.append(s)
    return out
