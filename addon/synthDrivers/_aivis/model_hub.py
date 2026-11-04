# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AivisHub (https://hub.aivis-project.com/) から音声合成モデル (AIVMX) を探してダウンロードする。NVDA には依存しない。

AivisHub API (https://api.aivis-project.com/v1):
  GET /aivm-models/search?sort=download|like|recent&keyword=&page=&limit=(最大 30)
  GET /aivm-models/{uuid}/download?model_type=AIVMX  (署名付き URL へリダイレクト)
検索結果の model_files にライセンスの種類・本文、版、大きさ、checksum (sha256) が入っている。
ダウンロードしたファイルは sha256 を照合してから保存する。

保存先: <ユーザー設定フォルダ>\\aivis\\models\\<uuid>.aivmx (paths.find_models が探す場所)
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .aivis_core.aivmx import read_aivm_manifest
from .cuda_installer import Cancelled, VerifyError, _download  # noqa: F401
from .remote_zip import USER_AGENT

API = "https://api.aivis-project.com/v1"
PAGE_SIZE = 30  # API の上限


def N_(text: str) -> str:
	"""翻訳の目印 (gettext_noop)。このモジュールは NVDA に依存しないのでここでは訳さず、
	画面に出すときに globalPlugins/DASH が _() で訳す。"""
	return text


SORTS = {"download": N_("Most popular (downloads)"), "recent": N_("Newest"), "like": N_("Most liked")}
LICENSE_LABELS = {
	"ACML 1.0": N_("ACML 1.0 (commercial and non-commercial use)"),
	"ACML-NC 1.0": N_("ACML-NC 1.0 (non-commercial use only)"),
	"CC0": N_("CC0 (public domain)"),
	"Custom": N_("Custom license (be sure to read the full text)"),
	"Internal": N_("Internal use"),
}
CATEGORY_LABELS = {
	"ExistingCharacter": N_("Existing character"),
	"OriginalCharacter": N_("Original character"),
	"LivingPerson": N_("Real person"),
	"DeceasedPerson": N_("Deceased person"),
	"FictionalPerson": N_("Fictional person"),
	"Other": N_("Other"),
}
TIMBRE_LABELS = {
	"YoungMale": N_("Boy"),
	"YoungFemale": N_("Girl"),
	"YouthfulMale": N_("Young man"),
	"YouthfulFemale": N_("Young woman"),
	"AdultMale": N_("Adult man"),
	"AdultFemale": N_("Adult woman"),
	"MiddleAgedMale": N_("Middle-aged man"),
	"MiddleAgedFemale": N_("Middle-aged woman"),
	"ElderlyMale": N_("Elderly man"),
	"ElderlyFemale": N_("Elderly woman"),
	"Neutral": N_("Androgynous"),
	"Baby": N_("Infant"),
	"Mechanical": N_("Mechanical"),
	"Other": N_("Other"),
}
# DASH が扱えるモデルの構造
SUPPORTED_ARCHITECTURES = ("Style-Bert-VITS2 (JP-Extra)",)


def _get_json(url: str) -> Any:
	req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
	with urllib.request.urlopen(req, timeout=60) as r:
		return json.loads(r.read().decode("utf-8"))


def search(sort: str = "download", keyword: str | None = None) -> list[dict[str, Any]]:
	"""公開されているモデルを全ページ分取得し、DASH で使えるもの (AIVMX があり、構造が対応しているもの) を返す。"""
	out: list[dict[str, Any]] = []
	page = 1
	while True:
		q = {"sort": sort, "page": page, "limit": PAGE_SIZE}
		if keyword:
			q["keyword"] = keyword
		d = _get_json(f"{API}/aivm-models/search?{urllib.parse.urlencode(q)}")
		models = d.get("aivm_models") or []
		for m in models:
			info = summarize(m)
			if info is not None:
				out.append(info)
		if not models or page * PAGE_SIZE >= int(d.get("total") or 0):
			return out
		page += 1


def summarize(m: dict[str, Any]) -> dict[str, Any] | None:
	"""検索結果の 1 件を、表示とダウンロードに使う形にまとめる。DASH で使えないものは None。"""
	f = next((f for f in m.get("model_files") or [] if f.get("model_type") == "AIVMX"), None)
	if f is None or f.get("model_architecture") not in SUPPORTED_ARCHITECTURES:
		return None
	user = m.get("user") or {}
	return {
		"uuid": m["aivm_model_uuid"],
		"name": m.get("name") or f.get("name") or "",
		"author": user.get("name") or user.get("handle") or "",
		"description": (m.get("description") or "").strip(),
		"category": CATEGORY_LABELS.get(m.get("category"), m.get("category") or ""),
		"timbre": TIMBRE_LABELS.get(m.get("voice_timbre"), m.get("voice_timbre") or ""),
		"downloads": int(m.get("total_download_count") or 0),
		"likes": int(m.get("like_count") or 0),
		"speakers": [(s.get("name", ""), [st.get("name", "") for st in s.get("styles") or []]) for s in m.get("speakers") or []],
		# 試聴用の音声 (スタイルごと。m4a): (話者, スタイル, URL, 読み上げ文)
		"samples": [
			(s.get("name", ""), st.get("name", ""), vs["audio_url"], vs.get("transcript") or "")
			for s in m.get("speakers") or []
			for st in s.get("styles") or []
			for vs in (st.get("voice_samples") or [])[:1]
			if vs.get("audio_url")
		],
		"license_type": f.get("license_type") or "",
		"license_label": LICENSE_LABELS.get(f.get("license_type"), f.get("license_type") or ""),
		"license_text": (f.get("license_text") or "").strip(),
		"version": f.get("version") or "",
		"size": int(f.get("file_size") or 0),
		"sha256": (f.get("checksum") or "").lower(),
		"url": f"{API}/aivm-models/{m['aivm_model_uuid']}/download?model_type=AIVMX",
	}


def fetch_sample(url: str, cache: dict) -> tuple[bytes, int, int]:
	"""試聴用の音声をダウンロードし、16 ビット PCM にして (データ, サンプリング周波数, チャンネル数) を返す。

	m4a (AAC) は NVDA では再生できないので Media Foundation で変換する (mf_decode)。一度取得したものは cache に置く。
	"""
	if url in cache:
		return cache[url]
	from .mf_decode import decode_to_pcm16

	fd, tmp = tempfile.mkstemp(prefix="dash_sample_", suffix=os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".m4a")
	os.close(fd)
	try:
		req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
		with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
			f.write(r.read())
		result = decode_to_pcm16(tmp)
	finally:
		try:
			os.unlink(tmp)
		except OSError:
			pass
	cache[url] = result
	return result


def model_path(models_dir: Path, uuid: str) -> Path:
	return Path(models_dir) / f"{uuid}.aivmx"


def download(
	info: dict[str, Any],
	models_dir: Path,
	progress: Callable[[int, int], None] | None = None,
	cancelled: Callable[[], bool] | None = None,
) -> Path:
	"""モデルをダウンロードして sha256 を照合し、models_dir\\<uuid>.aivmx に保存する (既にあれば置き換える)。"""
	models_dir = Path(models_dir)
	models_dir.mkdir(parents=True, exist_ok=True)
	total = int(info["size"])
	done = [0]

	def on_read(n: int) -> None:
		if cancelled is not None and cancelled():
			raise Cancelled()
		done[0] += n
		if progress is not None:
			progress(min(done[0], total), total)

	# 同じドライブに一時ファイルを作り、照合できてから置き換える (途中で止まっても壊れたファイルが残らない)
	fd, tmp = tempfile.mkstemp(prefix=".download_", suffix=".aivmx", dir=str(models_dir))
	os.close(fd)
	tmp = Path(tmp)
	try:
		_download(info["url"], tmp, info["sha256"], on_read)
		dest = model_path(models_dir, info["uuid"])
		tmp.replace(dest)
		return dest
	finally:
		tmp.unlink(missing_ok=True)


def downloaded_models(models_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
	"""models_dir にある .aivmx と、そのマニフェスト (名前・作者など) の一覧。"""
	out = []
	d = Path(models_dir)
	if d.is_dir():
		for p in sorted(d.glob("*.aivmx")):
			try:
				out.append((p, read_aivm_manifest(p)))
			except Exception:
				out.append((p, {"name": p.stem, "uuid": p.stem}))
	return out


def installed_uuids(model_paths) -> set[str]:
	"""DASH が今使えるモデル (同梱・AivisSpeech・ダウンロード済み) の UUID。"""
	out = set()
	for p in model_paths:
		try:
			out.add(read_aivm_manifest(p)["uuid"])
		except Exception:
			pass
	return out
