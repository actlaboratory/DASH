# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AivisSpeech Engine の既定辞書のダウンロード用マニフェスト (addon/synthDrivers/_aivis/dictionaries_manifest.json) を作る。

辞書は出典・ライセンスが辞書ごとに確認できていないのでアドオンには同梱せず、利用者が NVDA メニューから
AivisSpeech Engine の GitHub (固定のコミット) から直接ダウンロードする (_aivis/dict_installer.py)。
URL と sha256 は scripts/third_party_pins.json (third_party.py --update-pins --with-aivisspeech-dictionaries) から取る。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import third_party as tp  # noqa: E402

OUT = ROOT / "addon" / "synthDrivers" / "_aivis" / "dictionaries_manifest.json"


def main() -> None:
    pins = json.loads(tp.PINS.read_text(encoding="utf-8"))
    files = []
    for key in tp.aivisspeech_dictionaries():
        pin = pins[key]
        path = tp.fetch(key)  # 大きさを知るためにダウンロード (キャッシュ済みならそれを使う) して照合
        files.append({"name": path.name, "url": pin["url"], "sha256": pin["sha256"], "size": path.stat().st_size})
    out = {
        "version": f"AivisSpeech-Engine@{tp.AIVISSPEECH_REV[:7]}",
        "source": f"https://github.com/Aivis-Project/AivisSpeech-Engine/tree/{tp.AIVISSPEECH_REV}/resources/dictionaries",
        "files": files,
        "download_bytes": sum(f["size"] for f in files),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(files)} files, {out['download_bytes'] / 2**20:.0f}MB")


if __name__ == "__main__":
    main()
