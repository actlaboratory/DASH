# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AivisSpeech Engine の既定辞書 (OpenJTalk 形式の .dic, 38 個) をダウンロードしてインストールする。NVDA には依存しない。

辞書は AivisSpeech Engine のリポジトリ (LGPL-3.0) に含まれているが、辞書ごとの出典・ライセンスが確認できないので
アドオンには同梱せず、利用者が AivisSpeech Engine の GitHub (固定のコミット) から直接ダウンロードする。
一覧と sha256 は dictionaries_manifest.json (scripts/make_dict_manifest.py が生成)。

保存先: <ユーザー設定フォルダ>\\aivis\\aivisspeech_dictionaries
  (利用者が自分で辞書を置く aivis\\dictionaries とは別。入れ直し・削除で自分の辞書を消さないため)
いったん aivisspeech_dictionaries_pending に置き、すぐに切り替える (activate_pending)。
pyopenjtalk は読み込んだ辞書のファイルを開いたままにしないので、使用中でも入れ直し・削除ができる
(work/test_dict_install.py で確認)。万一切り替えられなかった場合は、NVDA の次回起動時に反映する。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .cuda_installer import COMPLETE_MARK, Cancelled, VerifyError, _download, _rmtree  # noqa: F401

MANIFEST = Path(__file__).resolve().parent / "dictionaries_manifest.json"
DIR_NAME = "aivisspeech_dictionaries"
REMOVE_MARK = "aivisspeech_dictionaries_remove"


def load_manifest() -> dict:
	return json.loads(MANIFEST.read_text(encoding="utf-8"))


def installed_dir(base: Path) -> Path:
	return base / DIR_NAME


def installed_version(base: Path) -> str | None:
	for d in (base / f"{DIR_NAME}_pending", base / DIR_NAME):
		mark = d / COMPLETE_MARK
		if mark.is_file():
			return mark.read_text(encoding="utf-8").strip() or "?"
	return None


def activate_pending(base: Path) -> bool:
	"""削除の予約・新しいインストールを反映する。辞書を読み込む前 (NVDA の起動時) と、インストール直後に呼ぶ。

	切り替えられたら True。既存の辞書が使用中で置き換えられなければ False (次回起動時に反映)。
	"""
	cur, pending = base / DIR_NAME, base / f"{DIR_NAME}_pending"
	if (base / REMOVE_MARK).exists():
		_rmtree(cur)
		_rmtree(pending)
		if not cur.exists():
			(base / REMOVE_MARK).unlink(missing_ok=True)
	if (pending / COMPLETE_MARK).is_file():
		_rmtree(cur)
		if cur.exists():
			return False
		pending.rename(cur)
		return True
	if pending.exists():
		_rmtree(pending)  # 途中で止まったインストールの残り
	return False


def request_remove(base: Path) -> None:
	base.mkdir(parents=True, exist_ok=True)
	(base / REMOVE_MARK).write_text("", encoding="utf-8")
	_rmtree(base / f"{DIR_NAME}_pending")


def install(
	base: Path,
	progress: Callable[[int, int, str], None] | None = None,
	cancelled: Callable[[], bool] | None = None,
	tmp_dir: Path | None = None,
) -> Path:
	"""base\\aivisspeech_dictionaries_pending にダウンロードし、そのパスを返す。

	progress(done_bytes, total_bytes, message) で進捗を通知し、cancelled() が True なら中断する (Cancelled 例外)。
	"""
	man = load_manifest()
	pending = base / f"{DIR_NAME}_pending"
	_rmtree(pending)
	pending.mkdir(parents=True)
	(base / REMOVE_MARK).unlink(missing_ok=True)
	total = int(man["download_bytes"])
	done = [0]

	def tick(n: int, msg: str) -> None:
		if cancelled is not None and cancelled():
			raise Cancelled()
		done[0] += n
		if progress is not None:
			progress(min(done[0], total), total, msg)

	tmp_root = Path(tempfile.mkdtemp(prefix="dash_dic_", dir=str(tmp_dir) if tmp_dir else None))
	try:
		for f in man["files"]:
			tick(0, f["name"])
			tmp = tmp_root / f["name"]
			# sha256 が一致しなければ VerifyError (改ざん・破損の検出)
			_download(f["url"], tmp, f["sha256"], lambda k, name=f["name"]: tick(k, name))
			shutil.move(str(tmp), str(pending / f["name"]))
		(pending / COMPLETE_MARK).write_text(man["version"], encoding="utf-8")
		return pending
	except BaseException:
		_rmtree(pending)
		raise
	finally:
		_rmtree(tmp_root)
