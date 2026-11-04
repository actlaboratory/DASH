# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""アドオンの翻訳 (gettext) を管理する。ソースの文言は英語で、各言語の訳は addon/locale/<lang>/LC_MESSAGES/nvda.po。

  python scripts/i18n.py update   ソースから翻訳の対象を集め、各言語の .po を同期する
                                  (訳は残す・新しい文言は空欄で追加・使われなくなった文言は消す)
  python scripts/i18n.py check    訳が無い文言があれば一覧を出して失敗する (終了コード 1)

翻訳の対象: addon/ の .py (同梱の deps と aivis_core は除く) と buildVars.py の、_("...") と N_("...") の文字列。
N_ は訳さずに目印だけ付ける (synthDrivers/_aivis/model_hub.py。画面に出すときに _() で訳す)。
直前の行の「# Translators: ...」は翻訳者向けの説明として .po に書く。

ビルド (scripts/build_addon.py) は compile_mo() で .po から .mo を作る。開発環境に GNU gettext が無くても動くように、
.po の読み書きと .mo の書き出しはここで行う。
"""

from __future__ import annotations

import ast
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON = ROOT / "addon"
LOCALE = ADDON / "locale"
KEYWORDS = ("_", "N_")
SKIP_DIRS = {"deps", "aivis_core", "__pycache__"}


def source_files() -> list[Path]:
	files = [ROOT / "buildVars.py"]
	for p in sorted(ADDON.rglob("*.py")):
		if not SKIP_DIRS.intersection(p.relative_to(ADDON).parts):
			files.append(p)
	return files


def extract() -> dict[str, dict]:
	"""msgid → {"refs": [...], "comments": [...]} (出てきた順)。"""
	out: dict[str, dict] = {}
	for path in source_files():
		src = path.read_text(encoding="utf-8")
		lines = src.splitlines()
		for node in ast.walk(ast.parse(src)):
			if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in KEYWORDS):
				continue
			if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
				continue
			msgid = node.args[0].value
			e = out.setdefault(msgid, {"refs": [], "comments": []})
			e["refs"].append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
			# 直前の連続したコメント行のうち「Translators:」で始まるもの
			i = node.lineno - 2
			block = []
			while i >= 0 and lines[i].strip().startswith("#"):
				block.insert(0, lines[i].strip().lstrip("#").strip())
				i -= 1
			for c in block:
				if c.startswith("Translators:") and c not in e["comments"]:
					e["comments"].append(c)
	return dict(sorted(out.items(), key=lambda kv: kv[1]["refs"][0]))


# ---- .po ----

def _unescape(s: str) -> str:
	out, i = [], 0
	while i < len(s):
		c = s[i]
		if c == "\\" and i + 1 < len(s):
			n = s[i + 1]
			out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}.get(n, "\\" + n))
			i += 2
		else:
			out.append(c)
			i += 1
	return "".join(out)


def _escape(s: str) -> str:
	return s.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def read_po(path: Path) -> tuple[str, dict[str, str]]:
	"""(ヘッダー, {msgid: msgstr})。"""
	entries: dict[str, str] = {}
	header = ""
	cur = {"msgid": None, "msgstr": None}
	field = None

	def flush():
		nonlocal header
		if cur["msgid"] is None:
			return
		if cur["msgid"] == "":
			header = cur["msgstr"] or ""
		else:
			entries[cur["msgid"]] = cur["msgstr"] or ""
		cur["msgid"] = cur["msgstr"] = None

	for raw in path.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line or line.startswith("#"):
			if not line:
				flush()
				field = None
			continue
		if line.startswith("msgid "):
			flush()
			field = "msgid"
			cur["msgid"] = _unescape(line[6:].strip()[1:-1])
		elif line.startswith("msgstr "):
			field = "msgstr"
			cur["msgstr"] = _unescape(line[7:].strip()[1:-1])
		elif line.startswith('"') and field:
			cur[field] += _unescape(line[1:-1])
	flush()
	return header, entries


def _po_string(keyword: str, s: str) -> str:
	if "\n" not in s.rstrip("\n") and len(s) < 70:
		return f'{keyword} "{_escape(s)}"\n'
	parts = s.split("\n")
	chunks = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
	return f'{keyword} ""\n' + "".join(f'"{_escape(c)}"\n' for c in chunks)


def write_po(path: Path, header: str, msgs: dict[str, dict], translations: dict[str, str]) -> None:
	if not header.strip():
		raise ValueError("empty header")
	out = ["# DASH (Direct Aivis Streaming Handler) translation.\n",
	       "# This file is distributed under the same license as the DASH package (AGPL-3.0-or-later).\n",
	       "#\n", 'msgid ""\n', 'msgstr ""\n'] + [f'"{_escape(line)}\\n"\n' for line in header.splitlines()]
	for msgid, e in msgs.items():
		out.append("\n")
		for c in e["comments"]:
			out.append(f"#. {c}\n")
		out.append("#: " + " ".join(e["refs"]) + "\n")
		if "{" in msgid:
			out.append("#, python-brace-format\n")
		elif "%s" in msgid:
			out.append("#, python-format\n")
		out.append(_po_string("msgid", msgid))
		out.append(_po_string("msgstr", translations.get(msgid, "")))
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(out), encoding="utf-8", newline="\n")


DEFAULT_HEADER = (
	"Project-Id-Version: DASH\n"
	"Report-Msgid-Bugs-To: https://github.com/actlaboratory/DASH/issues\n"
	"Language: {lang}\n"
	"MIME-Version: 1.0\n"
	"Content-Type: text/plain; charset=UTF-8\n"
	"Content-Transfer-Encoding: 8bit\n"
	"Plural-Forms: nplurals=1; plural=0;\n"
)


def po_files() -> list[Path]:
	return sorted(LOCALE.glob("*/LC_MESSAGES/nvda.po"))


def update(seed: dict[str, dict[str, str]] | None = None) -> None:
	"""各言語の .po をソースに合わせる。seed = {lang: {msgid: msgstr}} は空欄を埋めるのに使う (最初の移行用)。"""
	msgs = extract()
	# .po がまだ無い言語も、addon/locale/<lang>/LC_MESSAGES/ があれば作る
	langs = {d.parent.name: d / "nvda.po" for d in sorted(LOCALE.glob("*/LC_MESSAGES")) if d.is_dir()}
	for lang in (seed or {}):
		langs.setdefault(lang, LOCALE / lang / "LC_MESSAGES" / "nvda.po")
	for lang, path in langs.items():
		header, tr = read_po(path) if path.exists() else (DEFAULT_HEADER.format(lang=lang), {})
		for k, v in ((seed or {}).get(lang) or {}).items():
			if not tr.get(k):
				tr[k] = v
		dropped = sorted(set(tr) - set(msgs))
		write_po(path, header, msgs, tr)
		missing = [m for m in msgs if not tr.get(m)]
		print(f"{path.relative_to(ROOT)}: {len(msgs)} messages, {len(missing)} untranslated, {len(dropped)} removed")


def check() -> int:
	msgs = extract()
	bad = 0
	for path in po_files():
		_h, tr = read_po(path)
		missing = [m for m in msgs if not tr.get(m)]
		stale = [m for m in tr if m not in msgs]
		for m in missing:
			print(f"{path.relative_to(ROOT)}: untranslated: {m!r}")
		for m in stale:
			print(f"{path.relative_to(ROOT)}: not in source (run update): {m!r}")
		for m, s in tr.items():
			if m in msgs and s and _placeholders(m) != _placeholders(s):
				print(f"{path.relative_to(ROOT)}: placeholders differ: {m!r} -> {s!r}")
				bad += 1
		bad += len(missing) + len(stale)
	return 1 if bad else 0


def _placeholders(s: str) -> list[str]:
	import re

	return sorted(re.findall(r"\{[^{}]*\}|%[sd]", s))


# ---- .mo ----

def compile_mo(po: Path, mo: Path) -> int:
	""".po → .mo (GNU gettext 形式、ハッシュ表なし)。訳の空の項目は入れない。書いた項目数 (ヘッダー除く) を返す。"""
	header, tr = read_po(po)
	items = {"": header}
	items.update({k: v for k, v in tr.items() if v})
	keys = sorted(items, key=lambda k: k.encode("utf-8"))
	ids = [k.encode("utf-8") for k in keys]
	strs = [items[k].encode("utf-8") for k in keys]
	n = len(keys)
	start = 7 * 4 + n * 16
	orig, trans, data = [], [], b""
	off = start
	for b in ids:
		orig.append((len(b), off))
		data += b + b"\0"
		off += len(b) + 1
	for b in strs:
		trans.append((len(b), off))
		data += b + b"\0"
		off += len(b) + 1
	out = struct.pack("<7I", 0x950412DE, 0, n, 28, 28 + n * 8, 0, start)
	out += b"".join(struct.pack("<2I", *x) for x in orig)
	out += b"".join(struct.pack("<2I", *x) for x in trans)
	mo.parent.mkdir(parents=True, exist_ok=True)
	mo.write_bytes(out + data)
	return n - 1


def translate(po: Path, msgid: str) -> str:
	"""ビルド時に manifest の説明などを訳す。訳が無ければ英語のまま。"""
	return read_po(po)[1].get(msgid) or msgid


if __name__ == "__main__":
	cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
	if cmd == "update":
		update()
	elif cmd == "check":
		sys.exit(check())
	else:
		sys.exit(__doc__)
