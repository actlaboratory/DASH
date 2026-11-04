# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 yamahubuki, ACT Laboratory
# Portions Copyright (C) 2024-2026 tsukumijima, litagin02 (Style-Bert-VITS2)
# Based in part on Style-Bert-VITS2 (https://github.com/tsukumijima/Style-Bert-VITS2, commit b937361, style_bert_vits2/nlp/bert_models.py),
# licensed under the GNU Affero General Public License v3.0.
"""transformers に依存しない、日本語 DeBERTa (char-wwm) 用の最小トークナイザと BERT 推論。

元の Style-Bert-VITS2 は transformers.BertJapaneseTokenizer
(word_tokenizer_type="basic", subword_tokenizer_type="character") を使っている。
正規化済みテキストでは「1 文字 = 1 トークン」になるので、vocab.txt の引き当てだけで同じ ID 列になる。
"""

from __future__ import annotations

import enum
import threading
import unicodedata
from pathlib import Path

import numpy as np


class Languages(str, enum.Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


class CharTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.unk_id = self.vocab["[UNK]"]

    def tokenize(self, text: str) -> list[str]:
        # BasicTokenizer + CharacterTokenizer(normalize_text=True) 相当:
        # NFKC 正規化し、制御文字と空白を除いて 1 文字ずつに分ける
        text = unicodedata.normalize("NFKC", text)
        return [
            c
            for c in text
            if not c.isspace() and unicodedata.category(c) not in ("Cc", "Cf")
        ]

    def encode(self, text: str) -> list[int]:
        return [self.cls_id] + [self.vocab.get(c, self.unk_id) for c in self.tokenize(text)] + [self.sep_id]


_tokenizer: CharTokenizer | None = None
_bert_session = None
_bert_lock = threading.Lock()


def init(bert_dir: Path, session_options=None, providers=None) -> None:
    """BERT モデルのディレクトリ (vocab.txt と fp32 の model.onnx を含む) を指定して初期化する。

    fp32 版を使う。CPU では fp16 版の演算が遅く (Cast だらけになる)、fp32 版の方が約 3 倍速い
    (122ms → 40ms, scripts/bench_bert.py)。特徴量は同一 (cos=1.0000)。
    """
    import onnxruntime as ort

    global _tokenizer, _bert_session
    _tokenizer = CharTokenizer(bert_dir / "vocab.txt")
    providers = providers or ["CPUExecutionProvider"]
    _bert_session = ort.InferenceSession(str(bert_dir / "model.onnx"), session_options, providers=providers)


def init_tokenizer_only(bert_dir: Path) -> None:
    global _tokenizer
    _tokenizer = CharTokenizer(bert_dir / "vocab.txt")


def load_tokenizer(language: Languages = Languages.JP) -> CharTokenizer:
    """g2p.py から呼ばれる互換 API。"""
    assert _tokenizer is not None, "bert_models.init() has not been called"
    return _tokenizer


def extract_bert_feature(text: str, word2ph: list[int]) -> np.ndarray:
    """正規化済みテキストから音素単位の BERT 特徴量 (1024, 音素数) を返す。"""
    assert _tokenizer is not None and _bert_session is not None
    ids = np.array([_tokenizer.encode(text)], dtype=np.int64)
    assert len(word2ph) == ids.shape[1], (len(word2ph), ids.shape[1], text)
    feeds = {
        _bert_session.get_inputs()[0].name: ids,
        _bert_session.get_inputs()[1].name: np.ones_like(ids),
    }
    with _bert_lock:
        res = _bert_session.run(None, feeds)[0]
    if res.ndim == 3:
        res = res[0]
    res = res.astype(np.float32)
    # 各文字の特徴量をその文字に割り当てられた音素数だけ繰り返す
    feature = np.repeat(res, word2ph, axis=0)
    return feature.T
