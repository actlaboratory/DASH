# DASH: Direct Aivis Streaming Handler

## 概要

DASH (Direct Aivis Streaming Handler) は、AivisSpeech の音声合成モデル (AIVMX) を NVDA の音声合成エンジンとして使えるようにするアドオンです。
AivisSpeech Engine (HTTP サーバー) は使わず、音声合成のコア部分だけを NVDA の中で直接動かします。
動作に必要な BERT モデルと音声合成モデル「まお」を同梱しているので、AivisSpeech を別途インストールする必要はありません。

- **ストリーミング再生**: 音声合成モデルを「テキスト → 潜在表現」と「潜在表現 → 波形」に分割し、
  波形化を少しずつ行って、できた部分から再生します。文全体の合成を待たないので、
  短い発話なら 0.1〜0.2 秒程度で読み上げが始まります (CPU, Core i7-13700F で計測)。
- 長い文章は文・読点単位に区切り、先頭の文から順に合成します。
- AivisSpeech Engine と同じテキスト処理 (正規化・読み・アクセント・BERT) を使います。
- 文脈による読み分け (tsqyomi) に対応しています。「人気のない夜道」(ひとけ)、「上手から登場」(かみて)、
  「市場で魚を買う」(いちば) のように、同じ漢字でも文脈に合った読みを選びます。

### AivisSpeech について

AivisSpeech は、Aivis Project が開発している無料の日本語音声合成ソフトウェアです。
Style-Bert-VITS2 をもとにした音声合成モデルで、感情豊かで自然な読み上げができます。
音声合成モデルは AivisHub で多数公開されており、好きな声を追加して使えます。
DASH は、この AivisSpeech の音声合成モデルとテキスト処理を NVDA の中で直接使えるようにしたものです。

- AivisSpeech 公式サイト: https://aivis-project.com/
- AivisHub (音声合成モデルの公開サイト): https://hub.aivis-project.com/

## システム要件

- NVDA 2026.1 以降 (64 ビット版)
- Windows 10 / Windows 11 (64 ビット)
- メモリ: 4GB 以上の空き (BERT モデル・音声合成モデル・読み分けモデルで約 2.3GB を使います)
- ディスク: 約 2.2GB (CUDA 加速を入れる場合はさらに約 2GB)
- CPU: 4 コア以上を推奨。CPU だけで動作します
- GPU (任意): DirectX 12 対応 GPU (DirectML)、または NVIDIA の GPU (CUDA 加速)

## インストール

アドオンパッケージ (`.nvda-addon` ファイル) を開いてインストールし、NVDA を再起動してください。
アドオンストアの「外部ソースからインストール」ボタンからもインストールできます。
NVDA 設定ダイアログの「音声」カテゴリーで「変更」ボタンを押し、「音声エンジンの選択」ダイアログで「DASH (AivisSpeech)」を選びます。

NVDA の起動時に、モデルの読み込みに 10 秒ほどかかります。読み込みが終わるまでの読み上げは少し遅れます。

## 音声について

標準では、アドオンに同梱している音声合成モデル「まお」(Aivis Project) を使います。
声 (話者とスタイルの組み合わせ) は、NVDA の音声設定の「音声」で選べます。

### 音声合成モデルの追加

AivisHub (https://hub.aivis-project.com/) で公開されている音声合成モデルを、NVDA メニューから追加できます。

1. NVDA メニュー →「DASH」→「音声モデルをダウンロード」を選びます。
2. 「並び順」(人気順・新着順・いいねの多い順) と「キーワード」で探し、「モデル」の一覧から選びます。
   「詳細とライセンス」に、作者、声の種類、話者とスタイル、説明、ライセンスの種類と本文が表示されます。
   - 試聴: 「モデル」の一覧でスペースキーを押すと、そのモデルの試聴用の音声 (最初のスタイル) を再生します。
     もう一度押すと止まります。別のモデルに移ると止まります。
   - 「スタイルの試聴」の一覧では、話者・スタイルごとの試聴用の音声を、スペースキーかエンターで再生できます。
     「試聴」ボタンでも再生できます。
   - 試聴用の音声は AivisHub で公開されているもので、NVDA の出力デバイスから再生します。
3. 「ダウンロード」を押し、ライセンスへの同意を確認すると、ダウンロードします (1 つ約 240MB)。
   内容が正しいか (SHA-256) を確認してから保存します。
4. 完了すると、すぐに NVDA の音声設定の「音声」で選べるようになります (再起動は不要です)。

- 保存先: NVDA のユーザー設定フォルダー (通常 `%APPDATA%\nvda`) の `aivis\models`
- 削除: NVDA メニュー →「DASH」→「音声モデルを削除」(`aivis\models` にあるモデルが対象です)
- 声を切り替えた直後の 1 回目は、モデルの読み込みに 2〜3 秒ほどかかります (メモリを抑えるため、読み込むモデルは 1 つだけです)。

ほかに、次の方法でもモデルを追加できます。

- AivisSpeech をインストールして使う: AivisSpeech に追加したモデル
  (`%APPDATA%\AivisSpeech-Engine\Models`) を読み取り専用で流用します。AivisSpeech のユーザー辞書も読み込みます。
- 自分で配置する: `aivis\models` フォルダーに `.aivmx` ファイルを置きます (NVDA の再起動後に使えるようになります)。

### 音声合成モデルの利用規約

音声合成モデルには、それぞれ利用規約があります。必ずモデルごとの規約に従ってください。

同梱の「まお」は Aivis Common Model License (ACML) 1.0 です
(全文: アドオンのフォルダーの `synthDrivers\_aivis\data\models\` にある `*_LICENSE.md`)。
営利・非営利を問わず使えますが、次のような使い方は禁止されています。

- 話者本人・公式関係者であるかのように誤解させる利用 (ディープフェイクなど)
- 話者のイメージや尊厳を傷つける利用、実在の人物・団体などへの批判・攻撃・嫌がらせ・差別
- 人を騙す目的の虚偽情報の公開、誇大・虚偽のマーケティング
- 特定の政治・宗教・思想・陰謀論への賛同・反対を呼びかける活動
- 反社会的・犯罪目的の利用

追加したモデルの規約はモデルごとに異なります (非営利のみのモデルもあります)。
AivisHub のモデルのページや、AivisSpeech の「モデル管理」で確認してください。

## 合成パラメータ

NVDA の音声設定から次のパラメータを調整できます。

| パラメータ | 説明 | 規定値での動作 |
|---|---|---|
| 音声 | 話者とスタイルの組み合わせ | 最初に見つかった声 (同梱のモデルだけなら「まお (ノーマル)」) |
| 速さ | 読み上げ速度 | 50 で等速、75 で約 2.2 倍、100 で 5 倍。0 で 0.5 倍 |
| 高さ | 声の高さ | 50 で変化なし。0 で半オクターブ下、100 で半オクターブ上 |
| 抑揚 | 感情表現の強さ (AivisSpeech の「感情表現の強さ」) | 50 で標準 |
| 音量 | 音量 | 100 で標準 |
| GPU (DirectML/CUDA) を使う | GPU で推論する | オフ (CPU) |

- 速さ: 声の速さは変えずに合成し、NVDA 同梱の Sonic で後から速くします (速くしても聞き取りやすさを保つため)。
  CPU で動かしている場合、4 倍を超える分 (速さ 94 以上) だけは、合成が間に合うように AIVIS 側の話速も少し上げます
  (5 倍 = AIVIS 1.25 倍 × Sonic 4 倍)。CUDA 加速を使っている場合は、5 倍まで AIVIS 側の話速は変えません。
- 高さ: Sonic で変えるので、速さは変わりません。大文字を読むときの高さの変化 (NVDA の「大文字のピッチ変更率」) にも対応しています。
- GPU を使う: CUDA 加速 (下記) をインストールしていれば CUDA、していなければ DirectML を使います。
  DirectML は短い発話ではむしろ遅くなることがあるため、既定はオフです。

## NVDA メニューの項目

DASH をインストールすると、NVDA メニューに「DASH」の項目が追加されます。

- **起動時のアップデートチェックを無効化 / 有効化**: NVDA 起動時の自動アップデートチェックの ON/OFF
- **アップデートを確認**: 手動でのアップデートの確認
- **音声モデルをダウンロード**: AivisHub の音声合成モデルを選んでダウンロードし、声を追加します (上記「音声合成モデルの追加」)
- **音声モデルを削除**: ダウンロードした (または自分で置いた) 音声合成モデルを削除します
- **AivisSpeech の辞書をインストール**: AivisSpeech Engine の辞書を GitHub からダウンロード・インストールします (下記「辞書」)
- **AivisSpeech の辞書を削除**: ダウンロードした AivisSpeech の辞書を削除します
- **CUDA 加速をインストール**: NVIDIA の GPU 向けの CUDA 加速ライブラリをダウンロード・インストールします
- **CUDA 加速を削除**: ダウンロードした CUDA 加速ライブラリを削除します (NVDA の再起動後に削除されます)

## CUDA 加速について

NVIDIA の GPU があれば、CUDA で音声合成できます。CPU の負荷が数分の一になり、
連続読みや高速読み上げ、キーを続けて押したときの読み上げ開始が速くなります。

### CUDA インストール手順

1. NVDA メニュー →「DASH」→「CUDA 加速をインストール」を選びます。
2. 確認ダイアログで「はい」を選びます。(ダウンロードサイズ: 約 1.1GB)
3. 次のライブラリが配布元から直接ダウンロード・インストールされます。
   必要なファイルだけを取り出し、それぞれの内容が正しいか (SHA-256) を確認してから保存します。
   - onnxruntime-gpu 1.20.1 (Microsoft)
   - CUDA 11.8 ランタイム、cuBLAS、cuFFT、cuDNN 8.9 (NVIDIA, PyPI)
   - zlibwapi.dll (cuDNN 8 の依存ライブラリ, winimage)
4. 完了したら NVDA を再起動します。「GPU を使う」は自動でオンになります。

### 注意事項

- CUDA 加速には NVIDIA の GPU (GeForce RTX シリーズ等) と、NVIDIA のドライバが必要です。
- インストールには安定したインターネット接続と、約 2GB の空き容量が必要です。
- 保存先は NVDA のユーザー設定フォルダーの `aivis\cuda` です。アドオンを更新しても入れ直す必要はありません。
- 古い onnxruntime と cuDNN 8 を使っています。新しい版 (cuDNN 9) ではこのモデルの計算が遅くなるためです。
  そのため RTX 50 シリーズには対応していません。
- しばらく何も読まなかった直後の 1 発話目は、GPU が省電力状態から戻るため CPU より少し遅いことがあります。

## 辞書

- 読み・アクセントは pyopenjtalk-plus に内蔵の辞書を使います。
- AivisSpeech で登録したユーザー辞書 (`%APPDATA%\AivisSpeech-Engine\user.dic`) があれば自動で読み込みます。
- NVDA のユーザー設定フォルダーの `aivis\dictionaries\` に OpenJTalk 形式の辞書 (.dic) を置くと、それも読み込みます。

### AivisSpeech の辞書

AivisSpeech Engine に付属している辞書 (38 個。地名・人名・駅名・専門用語などの読みとアクセント) を追加すると、
AivisSpeech と同じように読めるようになります。

1. NVDA メニュー →「DASH」→「AivisSpeech の辞書をインストール」を選びます。
2. 確認ダイアログで「はい」を選ぶと、AivisSpeech Engine の GitHub から直接ダウンロードします (約 225MB)。
   それぞれの内容が正しいか (SHA-256) を確認してから保存します。
3. 完了すると、次の読み上げから使われます (NVDA の再起動は不要です)。

- 保存先: NVDA のユーザー設定フォルダーの `aivis\aivisspeech_dictionaries` (自分で置いた `aivis\dictionaries` とは別です)
- 削除: NVDA メニュー →「DASH」→「AivisSpeech の辞書を削除」
- これらの辞書は AivisSpeech Engine (LGPL-3.0) のリポジトリに含まれているものです。辞書ごとの出典・ライセンスが
  確認できないため、アドオンには同梱せず、利用者がダウンロードする形にしています。
  各辞書の出典・ライセンスは AivisSpeech Engine に準じます。
- 辞書を入れても、読み上げの開始までの時間はほとんど変わりません (1 発話あたり 2 ミリ秒程度)。

## 英語の読み上げについて

AivisSpeech Engine と同じく、英単語は自動でカタカナの読みに変換してから読み上げます (e2k)。
辞書にない単語は推定した読みになるため、不自然な読みになることがあります。

## メモリについて

CPU 推論を速くするため、BERT モデルは fp32 版 (約 1.3GB) を同梱しています (fp16 版より CPU で約 3 倍速い)。
音声合成モデルは選択中の 1 つだけを読み込みます。読み分けモデル (tsqyomi) が約 0.3GB を使い、
合わせて約 2.3GB のメモリを使います。

## 今後に向けて

- CPU で 3.5 倍以上の速さにすると、「ファイル」「編集」のようなごく短い項目を続けて読むときに、
  項目の間にわずかな間 (10〜90 ミリ秒) が空くことがあります。
- RTX 50 シリーズでの CUDA 加速には対応できていません。

## バージョン履歴

- Ver 0.1.0 (2026-11)
	- 初期バージョン

## 連絡先

ソフトウェアページ: https://actlab.org/software/DASH

メールでのお問い合わせ: support@actlab.org

## 著作権

本アドオンは GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) の条件に基づき使用することができます
(アドオンのフォルダーの `LICENSE.txt`)。
Copyright (c) 2026 yamahubuki, ACT Laboratory

日本語処理の一部 (正規化・読み・アクセント) は Style-Bert-VITS2 のコードを使っており、
その部分は GNU Affero General Public License v3.0 に基づき使用しています (各ファイルの先頭に出典を記載)。
Copyright (c) 2024-2026 tsukumijima, litagin02 (Style-Bert-VITS2)
Copyright (c) Bert-VITS2 contributors (fishaudio/Bert-VITS2)

モーラ表の一部は VOICEVOX ENGINE (GNU Lesser General Public License v3.0) に由来します。

BERT モデル (tsukumijima/deberta-v2-large-japanese-char-wwm-onnx) は、
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) に基づき、無改変で同梱しています。
元のモデル: ku-nlp/deberta-v2-large-japanese-char-wwm (京都大学 黒橋・褚・村脇研究室)

音声合成モデル「まお」(Aivis Project) は、Aivis Common Model License (ACML) 1.0 に基づき同梱しています。

DirectML.dll は、Microsoft Software License Terms (Microsoft DirectX Machine Learning) に基づき同梱しています。

ONNX Runtime は MIT License に基づき同梱しています。Copyright (c) Microsoft Corporation

読み分けモデル tsqyomi (tsukumijima/tsqyomi-models) は、MIT License に基づき同梱しています。
Copyright (c) 2026 Aivis Project

Hugging Face Tokenizers は Apache License 2.0 に基づき同梱しています。

pyopenjtalk-plus は MIT License に基づき同梱しています。
Copyright (c) 2018 Ryuichi Yamamoto

pyopenjtalk-plus に含まれる次のソフトウェア・データは、それぞれのライセンスに基づき同梱しています。

- Open JTalk: Modified BSD License
  Copyright (c) 2008-2016 Nagoya Institute of Technology, Department of Computer Science
- hts_engine API: Modified BSD License
  Copyright (c) 2001-2014 Nagoya Institute of Technology, Department of Computer Science;
  2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering
- MeCab: BSD License
  Copyright (c) 2001-2008 Taku Kudo; Copyright (c) 2004-2008 Nippon Telegraph and Telephone Corporation
- NAIST Japanese Dictionary (読みの辞書): BSD License
  Copyright (c) 2009 Nara Institute of Science and Technology, Japan
- HTS Voice "Mei" (このアドオンでは使用していません): Creative Commons Attribution 3.0
  Copyright (c) 2009-2013 Nagoya Institute of Technology, Department of Computer Science

これらのライセンス条文の全文と、その他の同梱ライブラリ (NumPy, ONNX, SudachiPy, pydantic, e2k など) の
ライセンスの一覧は、[third_party.html](third_party.html) にあります。
各ライブラリのライセンス文書は、アドオンのフォルダー (`synthDrivers\_aivis\deps\`) にも同梱しています。
