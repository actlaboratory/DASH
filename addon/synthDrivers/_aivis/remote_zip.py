# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""HTTP の Range 要求で、リモートの ZIP (wheel) から必要なファイルだけを取り出すための読み取り専用ファイル。

zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url))) のように使う。ZIP は末尾の目次 (central directory) と
各ファイルの位置が分かるので、全体 (cuDNN の wheel なら 712MB) を落とさなくても必要な DLL の分だけで済む。
サーバーが Range に対応していない場合は range_ok が False になる (呼び出し側で全体のダウンロードに切り替える)。
"""

from __future__ import annotations

import io
import urllib.request

USER_AGENT = "DASH-NVDA-addon"


class RangeNotSupported(Exception):
	pass


class HttpRangeFile(io.RawIOBase):
	def __init__(self, url: str, timeout: float = 60.0, on_read=None):
		self.url = url
		self.timeout = timeout
		self.on_read = on_read  # 読んだバイト数を通知 (進捗表示・キャンセル用。例外を投げれば中断)
		self.pos = 0
		self.requests = 0
		self.fetched = 0
		# 末尾 1 バイトを要求して、Range に対応しているかと全体の大きさを調べる
		req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
		with urllib.request.urlopen(req, timeout=timeout) as r:
			self.final_url = r.geturl()
			cr = r.headers.get("Content-Range")
			self.range_ok = r.status == 206 and cr is not None
			if self.range_ok:
				self.size = int(cr.rsplit("/", 1)[1])
			else:
				self.size = int(r.headers.get("Content-Length", 0))
		if not self.range_ok:
			raise RangeNotSupported(url)

	def readable(self) -> bool:
		return True

	def seekable(self) -> bool:
		return True

	def tell(self) -> int:
		return self.pos

	def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
		if whence == io.SEEK_SET:
			self.pos = offset
		elif whence == io.SEEK_CUR:
			self.pos += offset
		elif whence == io.SEEK_END:
			self.pos = self.size + offset
		return self.pos

	def readinto(self, b) -> int:
		n = min(len(b), self.size - self.pos)
		if n <= 0:
			return 0
		# リダイレクト後の URL (署名付き URL など) を使う
		req = urllib.request.Request(
			self.final_url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}", "User-Agent": USER_AGENT}
		)
		with urllib.request.urlopen(req, timeout=self.timeout) as r:
			if r.status != 206:
				raise RangeNotSupported(self.url)
			got = 0
			mv = memoryview(b)
			while got < n:
				k = r.readinto(mv[got:n])
				if not k:
					break
				got += k
				if self.on_read is not None:
					self.on_read(k)
		self.requests += 1
		self.fetched += got
		self.pos += got
		return got
