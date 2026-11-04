# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""モデル・BERT・辞書の置き場所を探す。

探す順番:
  1. NVDA ユーザー設定フォルダ\\aivis\\ (models\\*.aivmx, bert\\, dictionaries\\*.dic, user.dic)
  2. AivisSpeech Engine のデータ (%APPDATA%\\AivisSpeech-Engine) を読み取り専用で流用 (モデル・ユーザー辞書)
  3. アドオン同梱 (data\\bert\\, data\\models\\*.aivmx, dictionaries\\*.dic)
BERT は 1 → 3 だけ (fp32 版の model.onnx を同梱しているので、AivisSpeech の BERT は使わない)。
モデルは全部を使う。同じモデル (UUID) が複数あれば先に見つかったものを使う。
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def user_data_dir() -> Path:
	try:
		import globalVars

		base = Path(globalVars.appArgs.configPath)
	except Exception:
		base = Path(os.environ.get("APPDATA", ".")) / "nvda"
	return base / "aivis"


def aivisspeech_dir() -> Path:
	return Path(os.environ.get("APPDATA", "")) / "AivisSpeech-Engine"


def _bert_ok(d: Path) -> bool:
	return (d / "vocab.txt").is_file() and (d / "model.onnx").is_file()


def find_bert_dir() -> Path | None:
	"""BERT (fp32 の model.onnx と vocab.txt) のフォルダ。ユーザー設定フォルダ\\aivis\\bert → 同梱の data\\bert。"""
	for d in (user_data_dir() / "bert", PKG_DIR / "data" / "bert"):
		if _bert_ok(d):
			return d
	return None


def find_tsqyomi_dir() -> Path | None:
	"""tsqyomi (文脈による読み分けモデル) のフォルダ。ユーザー設定フォルダ\\aivis\\tsqyomi → 同梱の data\\tsqyomi。"""
	for d in (user_data_dir() / "tsqyomi", PKG_DIR / "data" / "tsqyomi"):
		if all((d / n).is_file() for n in ("model.onnx", "tokenizer.json", "metadata.json")):
			return d
	return None


def find_models() -> list[Path]:
	out: list[Path] = []
	for d in (user_data_dir() / "models", aivisspeech_dir() / "Models", PKG_DIR / "data" / "models"):
		if d.is_dir():
			out += sorted(d.glob("*.aivmx"))
	return out


def find_dictionaries() -> tuple[list[Path], list[Path]]:
	"""(既定辞書, ユーザー辞書) を返す。

	既定辞書: AivisSpeech Engine の既定辞書 (NVDA メニューからダウンロードした aivis\\aivisspeech_dictionaries、
	無ければビルド時に同梱した dictionaries) のあとに、利用者が自分で置いた aivis\\dictionaries\\*.dic を続ける。
	ユーザー辞書: AivisSpeech の user.dic と aivis\\user.dic (読みが後段で補正されないよう保護して適用する)。
	"""
	defaults: list[Path] = []
	for d in (user_data_dir() / "aivisspeech_dictionaries", PKG_DIR / "dictionaries"):
		if d.is_dir():
			defaults = sorted(d.glob("*.dic"))
			if defaults:
				break
	own = user_data_dir() / "dictionaries"
	if own.is_dir():
		defaults += sorted(own.glob("*.dic"))
	users = [p for p in (aivisspeech_dir() / "user.dic", user_data_dir() / "user.dic") if p.is_file()]
	return defaults, users


def find_cuda_pack() -> Path | None:
	"""CUDA 加速 (onnxruntime-gpu + CUDA/cuDNN の DLL) の場所。

	ユーザー設定フォルダ\\aivis\\cuda (トレイメニューからダウンロードしたもの)。
	以前の配布形式の別アドオン aivisCudaPack (addons\\aivisCudaPack\\cuda) も探す。
	"""
	addons_dir = PKG_DIR.parents[2]  # addons\\<アドオン>\\synthDrivers\\_aivis → addons
	for d in (user_data_dir() / "cuda", addons_dir / "aivisCudaPack" / "cuda"):
		if (d / "onnxruntime" / "__init__.py").is_file() and (d / "nvidia").is_dir():
			return d
	return None


def has_minimum_data() -> bool:
	return find_bert_dir() is not None and len(find_models()) > 0
