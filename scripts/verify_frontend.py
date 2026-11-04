# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""自作フロントエンドと本家 style_bert_vits2 (ref/sbv2) の get_text_onnx() の出力を比較する。"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\ref\sbv2")
import numpy as np

A = os.path.expandvars(r"%APPDATA%\AivisSpeech-Engine")
# 本家は AivisSpeech と同じ fp16 版 (model_fp16.onnx しか読まない)、DASH は同梱の fp32 版。特徴量は同じになるはず
REF_BERT_DIR = os.path.dirname(glob.glob(
    A + r"\BertModelCaches\models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\snapshots\*\model_fp16.onnx")[0])
BERT_DIR = os.path.join(ROOT, "build", "addon", "synthDrivers", "_aivis", "data", "bert")

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer_onnx import get_text_onnx
from style_bert_vits2.nlp import onnx_bert_models

from aivis_core.engine import read_aivmx_header
from aivis_core.frontend import text_to_features
from aivis_core.nlp import bert_models

providers = ["CPUExecutionProvider"]
# 本家: ローカルの BERT ディレクトリを直接ロード
onnx_bert_models.load_model(Languages.JP, pretrained_model_name_or_path=REF_BERT_DIR, onnx_providers=providers)
onnx_bert_models.load_tokenizer(Languages.JP, pretrained_model_name_or_path=REF_BERT_DIR)
bert_models.init(Path(BERT_DIR))

_, hps_dict, _ = read_aivmx_header(Path(sorted(glob.glob(A + r"\Models\a59cb814*.aivmx"))[0]))
hps = HyperParameters.model_validate(hps_dict)

texts = [l for l in open(ROOT + r"\scripts\texts.txt", encoding="utf-8").read().splitlines() if l.strip()]
bad = 0
for t in texts:
    _, ja_bert, _, phone, tone, lang = get_text_onnx(t, Languages.JP, hps, providers)
    f = text_to_features(t)
    same = (
        np.array_equal(phone, f.phones)
        and np.array_equal(tone, f.tones)
        and np.array_equal(lang, f.languages)
        and ja_bert.shape == f.bert.shape
    )
    err = float(np.abs(ja_bert - f.bert).max()) if ja_bert.shape == f.bert.shape else float("nan")
    rel = err / float(np.abs(ja_bert).max()) if ja_bert.shape == f.bert.shape else float("nan")
    if not same or rel > 1e-2:
        bad += 1
        print("DIFF", json.dumps(t, ensure_ascii=True), phone.shape, f.phones.shape, err)
    else:
        print("ok", len(phone), f"ids_equal={same} bert_maxabs={float(np.abs(ja_bert).max()):.1f} err={err:.1e} rel={rel:.1e}")
print("FRONTEND OK" if bad == 0 else f"FRONTEND MISMATCH {bad}/{len(texts)}")
