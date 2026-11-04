# DASH (開発用)

AivisSpeech の音声合成モデル (AIVMX, Style-Bert-VITS2 JP-Extra) を、ストリーミング再生で NVDA から使うための
音声エンジン (NVDA アドオン) です。AivisSpeech Engine (HTTP サーバー) は使わず、音声合成のコアだけを NVDA の中で動かします。

利用者向けの説明は [addon/doc/ja/readme.md](addon/doc/ja/readme.md)、開発の経緯は [log.txt](log.txt) にあります。

## 構成

| パス | 内容 |
|---|---|
| `aivis_core/` | 音声合成コア (NVDA 非依存)。前段 / デコーダへのグラフ分割、ストリーミング合成、日本語処理 |
| `addon/` | NVDA アドオンのソース (synthDrivers/dash.py, synthDrivers/_aivis/, globalPlugins/DASH/) |
| `buildVars.py`, `manifest.ini.tpl`, `version.json` | 名前・バージョン (VSU と同じ形。manifest.ini はビルド時に作る) |
| `sconstruct` | `scons` でビルドする入口 (中身は `scripts/build_addon.py` を呼ぶ) |
| `scripts/build_addon.py` | アドオンのビルド (ダウンロードと組み立て) |
| `.github/`, `tools/build.py`, `tools/bumpup.py`, `tools/download_gettext.py` | VSU と共通の GitHub Actions とビルドスクリプト (下記) |
| `scripts/third_party.py`, `scripts/third_party_pins.json` | 同梱するサードパーティのファイルのダウンロード元・sha256・ライセンス |
| `scripts/make_cuda_manifest.py` | CUDA 加速 (実行時にダウンロード) の一覧 `addon/synthDrivers/_aivis/cuda_manifest.json` を作る |
| `scripts/make_dict_manifest.py` | AivisSpeech の辞書 (実行時にダウンロード) の一覧 `addon/synthDrivers/_aivis/dictionaries_manifest.json` を作る |
| `scripts/nvda_harness.py` | NVDA を模したテスト (本物の NVDA の標準ライブラリと Sonic を使える) |
| `addon/locale/<言語>/LC_MESSAGES/nvda.po`, `scripts/i18n.py` | 画面の文言の翻訳 (下記) |
| `scripts/` のその他 | 計測・検証用のスクリプト |

## 開発環境の用意

OS には何もインストールせず、このフォルダの中で完結させます。

1. Python 3.13.9 (64 ビット) の埋め込み版 (`python-3.13.9-embed-amd64.zip`) を `tools\py313` に展開する
2. `tools\py313\python313._pth` の `#import site` のコメントを外し、`Lib/site-packages` の行を足す
3. `get-pip.py` で pip を入れ、`tools\py313\python.exe -m pip install -r requirements-dev.txt`

## ビルド

VSU と同じく `scons` でビルドする (`pip install -r requirements.txt` で scons と markdown を入れた Python で):

```
scons
```

- `output\Direct Aivis Streaming Handler-<version>.nvda-addon` ができる (約 1.6GB)。名前とバージョンは `buildVars.py`。
- バージョンを上げるときは VSU と同じく `python tools\bumpup.py` (buildVars.py と version.json を書き換える)。
- 開発環境の Python で直接 `tools\py313\python.exe -I scripts\build_addon.py` を実行しても同じものができる。
- 同梱するサードパーティのファイルは、すべて配布元から直接ダウンロードする (VSU と同じ考え方)。
  ダウンロードしたものは `build\cache\` に残り、2 回目からは使い回す。どれも sha256 を照合する。

| 同梱するもの | 入手元 | ライセンス |
|---|---|---|
| Python の依存ライブラリ (numpy, onnxruntime-directml, onnx, pyopenjtalk-plus, SudachiPy ほか) | PyPI の wheel | MIT / BSD / Apache-2.0 / PSF / LGPL-2.1 (num2words) |
| DirectML.dll のライセンス文書 | NuGet (Microsoft.AI.DirectML 1.15.4) | Microsoft DirectML ライセンス |
| e2k (英単語のカタカナ読み) | GitHub tsukumijima/e2k @ 9920754 | Unlicense |
| NVDA に無い標準ライブラリ | python.org (Python 3.13.9 埋め込み版) | PSF-2.0 |
| BERT (deberta-v2-large-japanese-char-wwm, fp32 ONNX) | Hugging Face (リビジョン d701ec6) | CC BY-SA 4.0 |
| 音声合成モデル「まお」(旧名 Anneli) | AivisHub | ACML 1.0 |
| 文脈による読み分けモデル tsqyomi (v6) と Tokenizers | Hugging Face (tsukumijima/tsqyomi-models @ 917a89d) / PyPI | MIT / Apache-2.0 |

一覧はアドオンの `doc/ja/third_party.html` にも書き出す。

オプション (`scripts/build_addon.py`):

- `--stage-only`: zip を作らず `build\addon\` に並べるだけ (テスト用)
- `--with-aivisspeech-dictionaries` (`scons dictionaries=1`): AivisSpeech Engine の既定辞書 38 個 (226MB) も同梱する。
  AivisSpeech Engine のリポジトリ (LGPL-3.0) に含まれているが、辞書ごとの出典とライセンスが確認できていないため既定では入れない。
  無くても動く (pyopenjtalk-plus 内蔵の辞書を使う)。配布版では、利用者が NVDA メニュー →「DASH」→
  「AivisSpeech の辞書をインストール」で AivisSpeech Engine の GitHub (コミット固定) から直接ダウンロードする
  (`_aivis/dict_installer.py`、一覧は `scripts/make_dict_manifest.py` で作る)。

版を上げるとき: `scripts/third_party.py` の版や URL を書き換えて
`tools\py313\python.exe -I scripts\third_party.py --update-pins` を実行すると、`third_party_pins.json` が更新される。

## GitHub Actions

VSU と同じワークフローでビルド・リリースする。**`.github/` と `tools/build.py`・`tools/bumpup.py`・`tools/download_gettext.py` は
VSU からのコピーで、編集しない** (共通部分が更新されたら丸ごとコピーし直す)。
例外として、VSU との違いはワークフローの Python の指定 2 行だけ:
`.github/workflows/pullRequest.yml` と `testAndBuild.yml` の `architecture: x86 → x64`、`python-version: 3.8 → "3.12"`
(NVDA の 64 ビット化に伴う。1.6GB のアドオンを tools/build.py がメモリに読んでハッシュを計算するため 32 ビットでは足りない。
3.13 では tools/build.py の `from audioop import add` が動かない (audioop が削除された) ので 3.12)。

- DASH 独自の点検 `.github/workflows/checks.yml` だけは VSU にない別ファイル (VSU からのコピーには手を加えない)。
  push とプルリクエストのたびに、本物の NVDA やモデルが無くても動く点検を Ubuntu + Python 3.13 で実行する:
  全 .py の構文、翻訳 (`scripts/i18n.py check`)、進捗ダイアログの再入の試験 (`scripts/test_progress_dialog.py`)。
- master への push → スナップショット (`DASH-latestcommit` のリリース、バージョンはコミット時刻から)
- `x.y.z` のタグ → 正式リリース (下書き)。タグは buildVars.py の ADDON_VERSION と一致させる
- プルリクエスト → `scons` でビルドできるかの確認
- 成果物 `DASH-*.zip`・`DASH-*_info.json` の名前はリポジトリ名で探されるので、**GitHub のリポジトリ名は `DASH`** にする
- ビルドのたびに BERT・モデルなど約 1.7GB を配布元からダウンロードする (sha256 で照合)

## テスト

```
set NVDA_DIR=<NVDA のフォルダ>
tools\py313\python.exe -I scripts\build_addon.py --stage-only
tools\py313\python.exe -I scripts\nvda_harness.py
```

- `NVDA_DIR` を指定すると、本物の NVDA の library.zip (標準ライブラリ) と Sonic を使う。
- `HARNESS_GPU=1` で GPU を使う設定にする。`HARNESS_QUICK=1` で高速時のシナリオだけを回す。
  `HARNESS_SWITCH=1` で別のモデルの声への切り替えを繰り返し、切り替え後の最初の音までの時間を測る。
- `APPDATA` を空のフォルダに向けると、AivisSpeech のデータが無い環境 (同梱物だけ) を試せる。

## 翻訳

画面の文言は NVDA のアドオンの標準どおり gettext で、ソースには英語で書く (`_("...")`)。
訳は `addon/locale/<言語>/LC_MESSAGES/nvda.po` (今は日本語 `ja` だけ。NVDA が日本語なら日本語、それ以外は英語で表示)。

- 文言を足したり変えたりしたら `tools\py313\python.exe -I scripts\i18n.py update` で .po を同期し、空欄の訳を埋める。
  `scripts\i18n.py check` は訳の無い文言や、`{name}` などの差し込みが訳と合わない項目があると失敗する。
- 翻訳者向けの説明は、文言の直前の行に `# Translators: ...` と書くと .po に入る。
- `synthDrivers/_aivis/model_hub.py` は NVDA に依存しないので `N_("...")` で目印だけ付け、画面に出すときに
  `globalPlugins/DASH` が `_()` で訳す。
- 言語を増やすときは `addon/locale/<言語>/LC_MESSAGES/` を作って `update` を実行し、できた .po を訳す。
- ビルドは .po から .mo と、訳したアドオンの説明 `locale/<言語>/manifest.ini` を作る (`buildVars.py` の説明を .po で訳す)。
- 利用者向けの説明書は `addon/doc/ja/readme.md` (日本語) と `addon/doc/en/readme.md` (英語)。日本語以外の NVDA は
  英語版を開く (NVDA は doc/<言語> → doc/<地域なしの言語> → doc/en の順に探す)。内容を変えたら両方を直す。
  同梱物の一覧 `third_party.html` は日本語版 (doc/ja) だけで、英語版からはそこへリンクしている。

## CUDA 加速

アドオンには含めず、利用者が NVDA メニュー →「DASH」→「CUDA 加速をインストール」で配布元からダウンロードする。
一覧 (`cuda_manifest.json`) は、動作確認した `tools\ort_cuda11` (onnxruntime-gpu 1.20.1 + CUDA 11.8 + cuDNN 8.9) から
`scripts/make_cuda_manifest.py` で作る。

## ライセンス

このプロジェクトのコードは **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)** で配布する
([LICENSE](LICENSE))。各ファイルの先頭に SPDX のヘッダーがある (`scripts/add_license_headers.py` で付ける)。

- `aivis_core/nlp/` のほとんどは Style-Bert-VITS2 (tsukumijima フォーク) から取り込んだコード
  (`aivis_core/LICENSE_style_bert_vits2`)。Style-Bert-VITS2 は「AGPL v3.0」としか書いていない (or later の記載が無い) ので、
  取り込んだファイルと一部を流用したファイル (`aivis_core/nlp/bert_models.py`, `aivis_core/frontend.py`) は
  **AGPL-3.0-only** のままにしている (Style-Bert-VITS2 の pyproject.toml も `AGPL-3.0-only`)。
  したがってアドオン全体としては AGPL-3.0 の条件で配布される。
- 取り込んだファイルのヘッダーには、元の著作権者 (git の履歴の作者: tsukumijima, litagin02。`symbols.py` は
  Bert-VITS2 (fishaudio) の作者たちも)、出典 (コミット b937361 の元のファイル名)、改変の有無を書いている。
  改変したファイル (g2p.py, g2p_utils.py, normalizer.py, nanairo_emoji.py) には AGPL-3.0 第 5 条 a に従い
  改変したことと日付を書いている。
- `mora_list.py` は Style-Bert-VITS2 が VOICEVOX ENGINE (LGPL-3.0, `aivis_core/LGPL_LICENSE`) から取り込んだもので、
  中のモーラ表は Open JTalk (修正 BSD, 名古屋工業大学) 由来。元の著作権表示はファイル内にそのまま残している。
- NVDA は GPL-2.0-or-later (github.com/nvaccess/nvda の copying.txt。NVDA 2026.1 に同梱のものはまだ「version 2」)。
  アドオンは NVDA の派生物として GPL version 2 or later で配布できる必要があるが、GPL-2.0-or-later は GPL-3.0 として
  扱え、GPL-3.0 と AGPL-3.0 は組み合わせられる (AGPL-3.0 第 13 条) ので、AGPL-3.0(-or-later) で配布する。
- 同梱するサードパーティのファイルは、それぞれのライセンスに従う (上の「ビルド」の表、アドオンの `doc/ja/third_party.html`)。
