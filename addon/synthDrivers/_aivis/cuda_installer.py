# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""CUDA 加速 (onnxruntime-gpu + CUDA / cuDNN の DLL) をダウンロードしてインストールする。NVDA には依存しない。

VSU の CUDA インストーラと同じく、配布元 (Microsoft の onnxruntime フィード、PyPI の NVIDIA の wheel、
winimage の zlib) から直接ダウンロードする。中身は cuda_manifest.json (scripts/make_cuda_manifest.py が生成):
  - wheel は ZIP なので、HTTP の Range 要求で必要な DLL だけを取り出す (1423MB → 1128MB)。
    Range に対応していないサーバーなら全体をダウンロードしてから取り出す
  - 取り出した各ファイルの sha256 をマニフェストと照合する (改ざん・破損の検出)

インストール先: <ユーザー設定フォルダ>\\aivis\\cuda_pending に置き、NVDA の次回起動時に cuda へ切り替える
(activate_pending())。読み込み済みの DLL は置き換えられないため。削除も次回起動時に行う。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .remote_zip import USER_AGENT, HttpRangeFile, RangeNotSupported

try:  # NVDA の中ではアドオンの翻訳を使う (ビルド用のスクリプトから使うときは訳さない)
	import addonHandler

	addonHandler.initTranslation()
except BaseException:

	def _(x):
		return x


MANIFEST = Path(__file__).resolve().parent / "cuda_manifest.json"
COMPLETE_MARK = ".complete"
REMOVE_MARK = "cuda_remove"


class Cancelled(Exception):
	pass


class VerifyError(Exception):
	pass


def load_manifest() -> dict:
	return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _rmtree(path: Path) -> None:
	def on_error(func, p, exc_info):
		try:
			os.chmod(p, stat.S_IWRITE)
			func(p)
		except Exception:
			pass

	if path.exists():
		shutil.rmtree(path, onerror=on_error)


def has_nvidia_driver() -> bool:
	"""NVIDIA のドライバ (nvcuda.dll) が入っているか。"""
	sysroot = os.environ.get("SystemRoot", r"C:\Windows")
	return (Path(sysroot) / "System32" / "nvcuda.dll").is_file()


def activate_pending(base: Path) -> None:
	"""NVDA 起動時 (CUDA の DLL を読み込む前) に呼ぶ。削除予約・新しいインストールを反映する。"""
	cuda, pending = base / "cuda", base / "cuda_pending"
	if (base / REMOVE_MARK).exists():
		_rmtree(cuda)
		_rmtree(pending)
		if not cuda.exists():
			(base / REMOVE_MARK).unlink(missing_ok=True)
	if (pending / COMPLETE_MARK).is_file():
		_rmtree(cuda)
		if not cuda.exists():
			pending.rename(cuda)
	elif pending.exists():
		_rmtree(pending)  # 途中で止まったインストールの残り


def installed_version(base: Path) -> str | None:
	for d in (base / "cuda_pending", base / "cuda"):
		mark = d / COMPLETE_MARK
		if mark.is_file():
			return mark.read_text(encoding="utf-8").strip() or "?"
	return None


def request_remove(base: Path) -> None:
	base.mkdir(parents=True, exist_ok=True)
	(base / REMOVE_MARK).write_text("", encoding="utf-8")
	_rmtree(base / "cuda_pending")


def install(
	base: Path,
	progress: Callable[[int, int, str], None] | None = None,
	cancelled: Callable[[], bool] | None = None,
	tmp_dir: Path | None = None,
	force_full_download: bool = False,
) -> Path:
	"""base\\cuda_pending にインストールし、そのパスを返す。

	progress(done_bytes, total_bytes, message): ダウンロード量で進捗を通知する (別スレッドから呼ばれる)。
	cancelled(): True を返すと中断する (Cancelled 例外)。
	"""
	man = load_manifest()
	pending = base / "cuda_pending"
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

	tmp_root = Path(tempfile.mkdtemp(prefix="aivis_cuda_", dir=str(tmp_dir) if tmp_dir else None))
	try:
		for src in man["sources"]:
			label = src["name"]
			tick(0, label)
			want = {f["member"]: f for f in src["files"]}
			z = None
			if not force_full_download:
				try:
					rf = HttpRangeFile(src["url"], on_read=lambda k, label=label: tick(k, label))
					z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=8 << 20))
				except RangeNotSupported:
					z = None
			if z is None:
				# 全体をダウンロード (この場合は wheel 全体の sha256 も照合)。進捗は必要分の割合に換算する
				need = sum(f["compressed"] for f in src["files"])
				scale = need / max(1, int(src["size"]))
				path = tmp_root / "download.whl"
				_download(src["url"], path, src["sha256"], lambda k, label=label: tick(int(k * scale), label))
				z = zipfile.ZipFile(path)
			with z:
				for member, f in want.items():
					_extract_verified(z, member, pending / f["dest"], f["sha256"], f["size"], cancelled)
			_rmtree(tmp_root / "download.whl")
		# zlibwapi.dll
		zl = man["zlib"]
		tick(0, zl["name"])
		path = tmp_root / "zlib.zip"
		_download(zl["url"], path, zl["sha256"], lambda k: tick(k, zl["name"]))
		with zipfile.ZipFile(path) as z:
			_extract_verified(z, zl["member"], pending / zl["dest"], zl["file_sha256"], zl["file_size"], cancelled)
		(pending / COMPLETE_MARK).write_text(man["version"], encoding="utf-8")
		return pending
	except BaseException:
		_rmtree(pending)
		raise
	finally:
		_rmtree(tmp_root)


def _download(url: str, path: Path, sha256: str, on_read: Callable[[int], None]) -> None:
	h = hashlib.sha256()
	req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
	with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as out:
		while True:
			b = r.read(1 << 20)
			if not b:
				break
			out.write(b)
			h.update(b)
			on_read(len(b))
	if h.hexdigest() != sha256:
		raise VerifyError(_("The downloaded file is corrupted: {url}").format(url=url))


def _extract_verified(z: zipfile.ZipFile, member: str, dest: Path, sha256: str, size: int, cancelled) -> None:
	dest.parent.mkdir(parents=True, exist_ok=True)
	h = hashlib.sha256()
	n = 0
	tmp = dest.with_name(dest.name + ".part")
	with z.open(member) as src, open(tmp, "wb") as out:
		while True:
			if cancelled is not None and cancelled():
				raise Cancelled()
			b = src.read(1 << 20)
			if not b:
				break
			out.write(b)
			h.update(b)
			n += len(b)
	if n != size or h.hexdigest() != sha256:
		tmp.unlink(missing_ok=True)
		raise VerifyError(_("The extracted file does not match: {member}").format(member=member))
	tmp.replace(dest)
