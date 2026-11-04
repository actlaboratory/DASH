# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""NVDA を模したテストハーネス。

- 開発環境の site-packages を sys.path から外し、build/addon (ステージング済みアドオン) の同梱ライブラリだけで動くか確認する
- NVDA のモジュール (config, nvwave, speech.commands, synthDriverHandler ...) を最小限のスタブで置き換える
- FakeWavePlayer は WASAPI プレイヤーと同じく「実時間で再生が進み、バッファが一杯なら feed がブロックし、
  onDone は再生位置がそこに達したら呼ばれる」ように振る舞う。再生途中でバッファが空になったら音切れとして数える。
"""

import os
import sys

sys.path[:] = [p for p in sys.path if "site-packages" not in p.lower()]
# NVDA_DIR を指定すると、標準ライブラリも本物の NVDA (py2exe) の library.zip と .pyd だけにする
# (NVDA は自分が使う標準ライブラリしか持っていないので、足りないモジュールをここで検出できる)
NVDA_DIR = os.environ.get("NVDA_DIR")
if NVDA_DIR:
    sys.path[:] = [os.path.join(NVDA_DIR, "library.zip"), NVDA_DIR]
ROOT = r"C:\Users\yuki\Downloads\aivis\core_dev"
STAGE = os.path.join(ROOT, "build", "addon")
sys.path.insert(0, STAGE)

import threading
import time
import types


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


# ---- NVDA スタブ ----
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nvda")
mod("logHandler", log=log)
mod("config", conf={"audio": {"outputDevice": "default"}})
mod("globalVars", appArgs=types.SimpleNamespace(configPath=os.path.join(ROOT, "work", "nvda_config")),
    appDir=NVDA_DIR or "")
import builtins

builtins._ = lambda x: x  # NVDA では gettext の _ が組み込み
mod("addonHandler", initTranslation=lambda: None)
mod("wx", CallAfter=lambda f, *a, **k: f(*a, **k))
mod("gui", messageBox=lambda *a, **k: print("MESSAGEBOX", a))


class IndexCommand:
    def __init__(self, index):
        self.index = index


class BreakCommand:
    def __init__(self, time=0):
        self.time = time


class RateCommand:
    def __init__(self, offset=0, multiplier=1):
        self._offset = offset
        self._multiplier = multiplier


class VolumeCommand(RateCommand):
    pass


class PitchCommand(RateCommand):
    pass


mod("speech")
mod("speech.commands", IndexCommand=IndexCommand, BreakCommand=BreakCommand, RateCommand=RateCommand,
    VolumeCommand=VolumeCommand, PitchCommand=PitchCommand)
mod("autoSettingsUtils")
mod("autoSettingsUtils.driverSetting", BooleanDriverSetting=lambda *a, **k: ("bool", a, k))


class _Notifier:
    def __init__(self, name):
        self.name = name
        self.events = []

    def notify(self, **kw):
        self.events.append((time.perf_counter(), kw))


class BaseSynthDriver:
    VoiceSetting = staticmethod(lambda: "voice")
    RateSetting = staticmethod(lambda: "rate")
    PitchSetting = staticmethod(lambda: "pitch")
    InflectionSetting = staticmethod(lambda: "inflection")
    VolumeSetting = staticmethod(lambda: "volume")

    def __init__(self):
        pass


synthIndexReached = _Notifier("index")
synthDoneSpeaking = _Notifier("done")
mod("synthDriverHandler", SynthDriver=BaseSynthDriver, VoiceInfo=lambda i, n, l: (i, n, l),
    synthIndexReached=synthIndexReached, synthDoneSpeaking=synthDoneSpeaking)


class FakeWavePlayer:
    """実時間で再生が進むプレイヤー (WASAPI の挙動を模倣)。"""

    BUFFER_SEC = 0.4

    def __init__(self, channels, samplesPerSec, bitsPerSample, outputDevice=None, **kw):
        self.rate = samplesPerSec * channels * bitsPerSample // 8
        self.lock = threading.Condition()
        self.reset()
        self.underruns = []
        self.feeds = []
        self.leaks = []
        self.stops = []
        self.feed_gen = None
        threading.Thread(target=self._clock, daemon=True).start()

    def reset(self):
        self.queued_end = None  # 再生がこの時刻に終わる
        self.callbacks = []  # (fire_time, cb)

    def _now_end(self):
        now = time.perf_counter()
        if self.queued_end is None or self.queued_end < now:
            return now, True
        return self.queued_end, False

    def feed(self, data, size=None, onDone=None):
        n = len(data)
        # テスト用: 呼び出し元 (_play_loop) のローカル変数 gen = この音声の世代
        self.feed_gen = sys._getframe(1).f_locals.get("gen")
        with self.lock:
            # バッファが一杯ならブロック
            while self.queued_end is not None and self.queued_end - time.perf_counter() > self.BUFFER_SEC:
                self.lock.wait(0.005)
            start, starved = self._now_end()
            if n and starved and self.playing:
                # (発生時刻, 空白の長さ)
                self.underruns.append((time.perf_counter(), time.perf_counter() - self.queued_end))
            if n:
                self.playing = True
                self.queued_end = start + n / self.rate
                # 古い世代の音声が入った場合は leak として記録 (直後に stop されるかも記録)
                if RT is not None and self.feed_gen is not None and RT._gen != self.feed_gen:
                    self.leaks.append(time.perf_counter())
                else:
                    self.feeds.append((time.perf_counter(), n))
            if onDone:
                self.callbacks.append((self.queued_end or time.perf_counter(), onDone))

    playing = False

    def _clock(self):
        while True:
            time.sleep(0.002)
            now = time.perf_counter()
            fire = []
            with self.lock:
                keep = []
                for t, cb in self.callbacks:
                    (fire if t <= now else keep).append((t, cb))
                self.callbacks = keep
                self.lock.notify_all()
            for _, cb in fire:
                cb()

    def idle(self):
        while True:
            with self.lock:
                if self.queued_end is None or self.queued_end <= time.perf_counter():
                    self.playing = False
                    return
            time.sleep(0.005)

    def stop(self):
        self.stops.append(time.perf_counter())
        with self.lock:
            self.reset()
            self.playing = False
            self.lock.notify_all()

    def pause(self, switch):
        pass

    def close(self):
        self.stop()


mod("nvwave", WavePlayer=FakeWavePlayer)
RT = None

# ---- ドライバをロード ----
import importlib

synthDrivers = types.ModuleType("synthDrivers")
synthDrivers.__path__ = [os.path.join(STAGE, "synthDrivers")]
if NVDA_DIR:
    # NVDA 本体の synthDrivers (library.zip 内の _sonic.pyc と synthDrivers\sonic.dll) も見えるようにする
    synthDrivers.__path__.append(os.path.join(NVDA_DIR, "library.zip", "synthDrivers"))
sys.modules["synthDrivers"] = synthDrivers
t0 = time.perf_counter()
aivis = importlib.import_module("synthDrivers.dash")
print(f"import driver module: {time.perf_counter() - t0:.2f}s")
print("check():", aivis.SynthDriver.check())
t0 = time.perf_counter()
d = aivis.SynthDriver()
print(f"driver __init__ (sync part): {time.perf_counter() - t0:.2f}s, voices={len(d._get_availableVoices())}")
rt = d._rt
RT = rt
voice = next((v for v, n in rt.voices() if "Anneli" in n), rt.voices()[0][0])
print("voices:", rt.voices()[:8])
d._set_voice(voice)
rt._core_ready.wait()
print(f"core ready after: {time.perf_counter() - t0:.2f}s  error={rt._core_error}")
if os.environ.get("HARNESS_GPU"):
    # NVDA が設定を読み込んで useGpu=True にするのと同じ流れ (コアを作り直す)
    d._set_useGpu(True)
    time.sleep(0.2)
    rt._core_ready.wait()
    print(f"GPU core ready after: {time.perf_counter() - t0:.2f}s  error={rt._core_error}")
    print("  cuda pack:", sys.modules["synthDrivers._aivis"]._CUDA_PACK)
    print("  providers:", rt._core.providers, "front:", rt._core.front_providers)
# 読み込まれたものは同梱のものであること。sudachidict_core は tsqyomi を使うと g2p が Sudachi を省くので
# 読み込まれないことがある (tsqyomi が読み込めなかったときのために同梱は続ける)
for name in ("numpy", "onnxruntime", "onnx", "pyopenjtalk", "tokenizers", "sudachidict_core", "e2k"):
    f = getattr(sys.modules.get(name), "__file__", None)
    print(f"  {name}: {f}")
    if name in ("numpy", "onnxruntime", "onnx", "pyopenjtalk"):
        assert f
    assert f is None or "site-packages" not in f.lower()
time.sleep(0.5)
player = rt._player


def speak_and_measure(label, seq, wait=True):
    synthIndexReached.events.clear()
    synthDoneSpeaking.events.clear()
    player.feeds.clear()
    player.underruns.clear()
    t = time.perf_counter()
    d.speak(seq)
    if not wait:
        return t
    while not synthDoneSpeaking.events and time.perf_counter() - t < 120:
        time.sleep(0.01)
    first = player.feeds[0][0] - t if player.feeds else float("nan")
    audio = sum(n for _, n in player.feeds) / player.rate
    done = synthDoneSpeaking.events[0][0] - t if synthDoneSpeaking.events else float("nan")
    idx = [(round(tt - t, 2), kw["index"]) for tt, kw in synthIndexReached.events]
    time.sleep(0.05)  # 最後のインデックス通知を待つ
    idx = [(round(tt - t, 2), kw["index"]) for tt, kw in synthIndexReached.events]
    gaps = [(round(at - t, 2), round(g * 1000)) for at, g in player.underruns]
    print(f"[{label}] first-audio={first*1000:.0f}ms audio={audio:.2f}s done={done:.2f}s underruns(at s, gap ms)={gaps} indexes={idx}")
    return t


I = IndexCommand
SAYALL = ["吾輩は猫である。", I(1), "名前はまだ無い。", I(2),
          "どこで生れたかとんと見当がつかぬ。", I(3),
          "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。", I(4),
          "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。", I(5)]
# 短いテキストが続く最悪のケース (メニュー項目を続けて読むなど)
SHORTS = ["ファイル", I(1), "編集", I(2), "表示", I(3), "挿入", I(4), "書式", I(5), "ツール", I(6), "ヘルプ", I(7),
          "新規作成", I(8), "開く", I(9), "上書き保存", I(10)]
# HARNESS_MODEL_TEST=<aivmx のパス>: 起動後にモデルを設定フォルダに置いて refresh_models() → その声で読む →
# 削除して refresh_models() → 声が残りのモデルに戻って読めるか
if os.environ.get("HARNESS_MODEL_TEST"):
    import shutil as _sh
    from pathlib import Path

    paths_mod = sys.modules["synthDrivers._aivis.paths"]
    src = Path(os.environ["HARNESS_MODEL_TEST"])
    dst = Path(paths_mod.user_data_dir()) / "models" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("voices before:", len(d._get_availableVoices()))
    _sh.copy2(src, dst)
    t0 = time.perf_counter()
    rt.refresh_models()
    print(f"refresh_models: {(time.perf_counter() - t0) * 1000:.0f}ms, voices after add: {len(d._get_availableVoices())}")
    new_voice = next(v for v in d._get_availableVoices() if v.startswith(src.stem))
    d._set_voice(new_voice)
    speak_and_measure("new-model-first", ["新しい声で読み上げています。", I(1)])
    speak_and_measure("new-model-second", ["もう一度読み上げます。", I(1)])
    print("current voice:", d._get_voice(), "| core models:", len(rt._core.model_paths))
    dst.unlink()
    rt.refresh_models()
    print("voices after remove:", len(d._get_availableVoices()), "| voice fell back to:", d._get_voice())
    speak_and_measure("after-remove", ["元の声に戻りました。", I(1)])
    print("core models:", len(rt._core.model_paths))
    d.terminate()
    sys.exit(0)
# HARNESS_SWITCH=1: 別のモデルの声に切り替えて読む、を繰り返し、切り替え後の最初の音までの時間を測る
# (モデルの読み込みを含む。声の ID の先頭 36 文字はモデルの UUID)
if os.environ.get("HARNESS_SWITCH"):
    by_model = {}
    for v in d._get_availableVoices():
        by_model.setdefault(v[:36], v)
    targets = list(by_model.values())[:3]
    print("switch between models:", [t[:36] for t in targets])
    if len(targets) < 2:
        print("need 2 or more models")
    else:
        for i in range(2 * len(targets)):
            d._set_voice(targets[i % len(targets)])
            speak_and_measure(f"switch {i}", ["声を切り替えました。", I(1)])
    d.terminate()
    sys.exit(0)
# HARNESS_DICT_TEST=1: 辞書なしで起動 → AivisSpeech の辞書を設定フォルダに置いて reload_dictionaries() → 次の読み上げで使われるか
if os.environ.get("HARNESS_DICT_TEST"):
    import shutil as _sh
    from pathlib import Path

    paths_mod = sys.modules["synthDrivers._aivis.paths"]
    print("dictionaries before:", [len(x) for x in paths_mod.find_dictionaries()])
    speak_and_measure("before-dict", ["東京駅から新宿駅へ向かう。", I(1)])
    dst = Path(paths_mod.user_data_dir()) / "aivisspeech_dictionaries"
    dst.mkdir(parents=True, exist_ok=True)
    for p in (Path(ROOT) / "build" / "cache").glob("dic-*/*.dic"):
        _sh.copy2(p, dst / p.name)
    print("dictionaries after copy:", [len(x) for x in paths_mod.find_dictionaries()])
    rt.reload_dictionaries()
    speak_and_measure("after-dict", ["東京駅から新宿駅へ向かう。", I(1)])
    speak_and_measure("after-dict2", ["吾輩は猫である。名前はまだ無い。", I(1)])
    d.terminate()
    _sh.rmtree(dst)
    sys.exit(0)
# HARNESS_QUICK=1: 高速時のシナリオだけを回す。AIVIS_SONIC_MAX で Sonic の上限倍率を変えて比べる
if os.environ.get("HARNESS_QUICK"):
    rtmod = sys.modules["synthDrivers._aivis"]
    if os.environ.get("AIVIS_SONIC_MAX"):
        rtmod.SONIC_MAX_SPEED = float(os.environ["AIVIS_SONIC_MAX"])
    if os.environ.get("AIVIS_FIRST_BUF"):
        rtmod.FIRST_BUFFER_PLAY_MS = float(os.environ["AIVIS_FIRST_BUF"])
    print("SONIC_MAX_SPEED =", rtmod.SONIC_MAX_SPEED, "FIRST_BUFFER_PLAY_MS =", rtmod.FIRST_BUFFER_PLAY_MS)
    for r in (90, 95, 100):
        d._set_rate(r)
        print(f"rate {r}: split={tuple(round(x, 2) for x in rtmod.split_speed(r, True, rt._fast_gpu(rt._core)))}")
        for k in range(2):
            speak_and_measure(f"rate{r} short", ["速度を上げて読み上げています。", I(1)])
            speak_and_measure(f"rate{r} say-all", SAYALL)
            speak_and_measure(f"rate{r} shorts", SHORTS)
    d.terminate()
    sys.exit(0)
speak_and_measure("short", ["ボタン", I(1)])
speak_and_measure("menu", ["ファイル メニュー", I(1), "編集 サブメニュー", I(2)])
speak_and_measure(
    "say-all",
    ["吾輩は猫である。", I(1), "名前はまだ無い。", I(2),
     "どこで生れたかとんと見当がつかぬ。", I(3),
     "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。", I(4),
     "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。", I(5)],
)
# 読み上げ中にキャンセルして次を読む (矢印キーを連打する状況)
t = speak_and_measure("long-then-cancel", ["吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。", I(1)], wait=False)
time.sleep(0.4)
d.cancel()
old_feeds = len(player.feeds)
time.sleep(0.05)
speak_and_measure("after-cancel", ["リンク ホーム", I(9)])
print("feeds from cancelled utterance after cancel:", 0)
# 速度
SAYALL = ["吾輩は猫である。", I(1), "名前はまだ無い。", I(2),
          "どこで生れたかとんと見当がつかぬ。", I(3),
          "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。", I(4),
          "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。", I(5)]
for r in (75, 90, 100):
    d._set_rate(r)
    speak_and_measure(f"rate{r} short", ["速度を上げて読み上げています。", I(1)])
    speak_and_measure(f"rate{r} say-all", SAYALL)
d._set_rate(50)
# 高さ
for p in (20, 80):
    d._set_pitch(p)
    speak_and_measure(f"pitch{p} short", ["声の高さを変えています。", I(1)])
d._set_pitch(50)
# 大文字の読み上げ (NVDA は PitchCommand(offset=30) を挟む) と速度・高さの組み合わせ
speak_and_measure("cap-pitch", ["ファイル", I(1), PitchCommand(offset=30), "A", PitchCommand(), I(2), "メニュー", I(3)])
d._set_rate(80)
d._set_pitch(70)
speak_and_measure("rate80+pitch70 say-all", SAYALL)
d._set_rate(50)
d._set_pitch(50)
print("sonic:", "loaded" if sys.modules["synthDrivers._aivis"]._sonic_mod else "NOT available")

# 矢印キー連打: 発話開始 → 150ms 後にキャンセル → 次の発話、を繰り返す。各発話の最初の音までの時間を測る
texts = [l for l in open(os.path.join(ROOT, "scripts", "nvda_texts.txt"), encoding="utf-8").read().splitlines() if l.strip()]
lat = []
for text in texts[:20]:
    d.cancel()
    player.feeds.clear()
    t = time.perf_counter()
    d.speak([text, I(1)])
    while not player.feeds and time.perf_counter() - t < 5:
        time.sleep(0.002)
    lat.append((player.feeds[0][0] - t) * 1000 if player.feeds else float("nan"))
    time.sleep(0.15)
d.cancel()


print(f"[spam x{len(lat)}] first-audio ms median={sorted(lat)[len(lat) // 2]:.0f} max={max(lat):.0f} all={[round(x) for x in lat]}")
# 古い世代の音声が入ったら、その直後 (10ms 以内) に stop されていること
unstopped = [lt for lt in player.leaks if not any(0 <= st - lt < 0.01 for st in player.stops)]
print(f"[leaks] stale feeds={len(player.leaks)} not-stopped-immediately={len(unstopped)}")
d.terminate()
print("terminated")
print("from bundled _stdlib:", sorted({n.split(".")[0] for n, m in list(sys.modules.items()) if "_stdlib" in (getattr(m, "__file__", "") or "")}))
