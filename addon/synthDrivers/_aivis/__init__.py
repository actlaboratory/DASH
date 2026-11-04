# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""DASH の実行部分 (NVDA 依存)。

スレッド構成 (VSU と同じ考え方):
  SynthThread : 読み上げ要求を受け取り、aivis_core でストリーミング合成して PlayQueue に PCM 区間を積む
  PlayThread  : PlayQueue から取り出して nvwave.WavePlayer に流す。インデックスは再生位置に合わせて通知する
cancel() は世代番号を進め、両キューを空にしてプレイヤーを止める。合成側は世代番号を見て途中で打ち切る。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from functools import partial

# 同梱の依存ライブラリ (cp313 / win_amd64)。NVDA 本体が同名モジュールを持っていればそちらが優先される
_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
# NVDA は自分が使う標準ライブラリしか同梱していないので、足りない分を補う (NVDA にあるものが優先)
_STDLIB = os.path.join(_DEPS, "_stdlib")
for _p in (_DEPS, os.path.join(_STDLIB, "python313.zip"), _STDLIB):
	if os.path.exists(_p) and _p not in sys.path:
		sys.path.append(_p)

import config
import nvwave
from logHandler import log
from speech.commands import BreakCommand, IndexCommand, PitchCommand, RateCommand, VolumeCommand

from . import paths

try:
	import addonHandler

	addonHandler.initTranslation()
except BaseException:

	def _(x):
		return x


# CUDA 加速 (トレイメニューの「DASH」→「CUDA 加速をインストール」でダウンロードしたもの) があれば、
# 同梱の onnxruntime (DirectML 版) の代わりにその onnxruntime-gpu を使う
# (同じプロセスに 2 つの onnxruntime は読み込めないので、先頭に置いて優先させる)。
# ダウンロードは cuda_pending に置かれるので、DLL を読み込む前のここで cuda に切り替える (削除の予約も反映)
try:
	from . import cuda_installer

	cuda_installer.activate_pending(paths.user_data_dir())
except Exception:
	log.error("DASH: failed to activate the downloaded CUDA files", exc_info=True)
# NVDA メニューからダウンロードした AivisSpeech の辞書も、読み込む前のここで入れ替え・削除の予約を反映する
try:
	from . import dict_installer

	dict_installer.activate_pending(paths.user_data_dir())
except Exception:
	log.error("DASH: failed to activate the downloaded dictionaries", exc_info=True)
_CUDA_PACK = paths.find_cuda_pack()
if _CUDA_PACK is not None and str(_CUDA_PACK) not in sys.path:
	sys.path.insert(0, str(_CUDA_PACK))
_cuda_dlls_loaded = False
# CUDA パックの DLL を読み込む順 (依存される側から)。cuDNN 8 は分割 DLL を名前で LoadLibrary するので、
# 先にフルパスで読み込んでおけば読み込み済みのものが使われる (PATH を変える必要がない)
_CUDA_DLL_ORDER = ("cudart", "cublasLt", "cublas", "cufft", "zlibwapi", "cudnn_ops_infer", "cudnn_cnn_infer",
                   "cudnn_adv_infer", "cudnn_graph", "cudnn_engines", "cudnn_heuristic", "cudnn_ops", "cudnn_cnn",
                   "cudnn_adv", "cudnn64")


# 推論に使わない DLL (32 ビット版、学習用など)。手動で置いたフォルダに混ざっていても読まない
_CUDA_DLL_SKIP = ("cudart32", "nvblas", "cufftw", "cudnn_adv_train", "cudnn_cnn_train", "cudnn_ops_train")


def _preload_cuda_dlls(pack) -> None:
	"""CUDA / cuDNN の DLL を依存の順に読み込む。読み込めないものがあれば、まとめて例外にする。"""
	import ctypes

	dlls = [p for p in pack.glob("nvidia/*/bin/*.dll") if not p.name.startswith(_CUDA_DLL_SKIP)]

	def rank(p):
		for i, k in enumerate(_CUDA_DLL_ORDER):
			if p.name.startswith(k):
				return i
		return len(_CUDA_DLL_ORDER)

	failed = []
	for p in sorted(dlls, key=rank):
		try:
			ctypes.WinDLL(str(p))
		except OSError as e:
			failed.append(f"{p.name}: {e}")
	if failed:
		raise OSError("could not load: " + "; ".join(failed))


def _load_onnxruntime(use_gpu: bool) -> None:
	"""onnxruntime を読み込む。CUDA パックを使う場合、GPU を使う設定なら CUDA / cuDNN の DLL を先に読み込む。

	- パックの onnxruntime 自体が読み込めなければ、パックを外して同梱のものを使う
	- onnxruntime を読み込んだ後で CUDA の DLL が読み込めなかった場合は、同梱のものに戻せない
	  (読み込み済みの onnxruntime.dll はプロセスから外せず、混ざって壊れる) ので、パックの onnxruntime を CPU で使う
	  (CUDA の実行環境は使えないので、AivisCore は CPU で動く)
	"""
	global _CUDA_PACK, _cuda_dlls_loaded
	if _CUDA_PACK is None:
		return
	if "onnxruntime" not in sys.modules:
		try:
			import onnxruntime  # noqa: F401
		except Exception:
			log.error("DASH: failed to load the CUDA pack; using the bundled onnxruntime", exc_info=True)
			try:
				sys.path.remove(str(_CUDA_PACK))
			except ValueError:
				pass
			for k in [k for k in sys.modules if k == "onnxruntime" or k.startswith("onnxruntime.")]:
				del sys.modules[k]
			_CUDA_PACK = None
			return
	import onnxruntime

	if use_gpu and not _cuda_dlls_loaded:
		try:
			# パック内の nvidia\*\bin から読み込む (onnxruntime 1.20 には preload_dlls() が無い)
			_preload_cuda_dlls(_CUDA_PACK)
			_cuda_dlls_loaded = True
		except Exception:
			log.error("DASH: failed to load the CUDA libraries; synthesizing on the CPU", exc_info=True)
	log.info(f"DASH: using CUDA pack {_CUDA_PACK} (providers={onnxruntime.get_available_providers()})")


SAMPLE_RATE = 44100

# NVDA の速度 0..100 → 再生速度。0 で 0.5 倍、50 で等倍、100 で 5 倍 (指数的に変化)
MIN_SPEED = 0.5
MAX_SPEED = 5.0
# CPU で合成するとき、Sonic で速くするのはこの倍率まで。超えた分は AIVIS 側の話速で速くする
# (5 倍 = Sonic 4 倍 × AIVIS 1.25 倍)。AIVIS 側を速くするとデコードするフレーム数も減るので、合成が間に合いやすくなる。
# GPU (CUDA) では合成に十分な余裕がある (デコーダ RTF 0.015) ので、AIVIS 側は常に 1.0 で Sonic だけで速くする
SONIC_MAX_SPEED = 4.0
# Sonic で速くするとき、最初に溜めてから再生を始める量 (再生時間で)
FIRST_BUFFER_PLAY_MS = 100


def rate_to_speed(rate: int) -> float:
	if rate == 50:
		return 1.0
	if rate < 50:
		return MIN_SPEED ** ((50 - rate) / 50.0)
	return MAX_SPEED ** ((rate - 50) / 50.0)


def split_speed(rate: int, sonic: bool, gpu: bool = False) -> tuple[float, float]:
	"""速度を (AIVIS 側の話速, Sonic の倍率) に分ける。"""
	speed = rate_to_speed(rate)
	if not sonic:
		return speed, 1.0
	if gpu:
		return 1.0, speed
	s = min(speed, SONIC_MAX_SPEED)
	return speed / s, s


# NVDA の高さ 0..100 → 声の高さの倍率。50 で等倍、0 で半オクターブ下、100 で半オクターブ上。
# (大文字の読み上げでは既定で +30 されるので、等倍から約 1.23 倍 = 3.6 半音上がる)
PITCH_OCTAVES = 0.5


def pitch_to_factor(pitch: int) -> float:
	return 2.0 ** ((pitch - 50) / 50.0 * PITCH_OCTAVES)


def _prosody(item, base: int) -> int:
	"""Rate/Pitch/VolumeCommand をドライバの現在値 base に当てはめた値 (0..100)。

	NVDA の command.multiplier は offset 指定 (大文字の高さ変更など) のとき NVDA 側の設定値から計算するので、
	ここでは _offset / _multiplier を直接使う。"""
	offset = getattr(item, "_offset", 0) or 0
	if offset:
		value = base + offset
	else:
		value = base * getattr(item, "_multiplier", getattr(item, "multiplier", 1))
	return max(0, min(100, int(value)))


_sonic_mod = None
_sonic_failed = False


def _sonic_available() -> bool:
	"""NVDA 同梱の Sonic (synthDrivers/sonic.dll, synthDrivers._sonic) を読み込む。"""
	global _sonic_mod, _sonic_failed
	if _sonic_mod is not None:
		return True
	if _sonic_failed:
		return False
	try:
		from synthDrivers import _sonic

		_sonic.initialize()
		_sonic_mod = _sonic
		return True
	except Exception:
		log.warning("DASH: NVDA's Sonic library is not available; falling back to model speed", exc_info=True)
		_sonic_failed = True
		return False


class _Speedup:
	"""Sonic による音声の高速化 (音程は変えずに速度だけ変える。高倍速の音声向けのアルゴリズム)。"""

	def __init__(self):
		self._s = _sonic_mod.SonicStream(SAMPLE_RATE, 1)
		self._speed = 1.0
		self._pitch = 1.0

	def set_speed(self, speed: float):
		if speed != self._speed:
			self._s.speed = speed
			self._speed = speed

	def set_pitch(self, pitch: float):
		if pitch != self._pitch:
			self._s.pitch = pitch
			self._pitch = pitch

	def process(self, pcm) -> bytes:
		self._s.writeShort(pcm.ctypes.data, pcm.size)
		if self._s.samplesAvailable == 0:
			return b""
		return bytes(self._s.readShort())

	def flush(self) -> bytes:
		self._s.flush()
		if self._s.samplesAvailable == 0:
			return b""
		return bytes(self._s.readShort())


class Runtime:
	def __init__(self, on_index, on_done):
		self._on_index = on_index
		self._on_done = on_done
		self._gen = 0
		self._queued_until = 0.0
		self.rate = 50
		self.pitch = 50
		self.volume = 100
		self.inflection = 50
		self.use_gpu = False
		self.voice = ""
		self._core = None
		self._core_error: BaseException | None = None
		self._core_ready = threading.Event()

		bert_dir = paths.find_bert_dir()
		self._model_paths = paths.find_models()
		if bert_dir is None:
			raise RuntimeError(
				_("The BERT model (model.onnx) was not found. Please reinstall DASH.")
			)
		if not self._model_paths:
			raise RuntimeError(
				_("No voice models (.aivmx) were found. Place them in {path}.").format(
					path=paths.user_data_dir() / "models")
			)
		self._bert_dir = bert_dir

		# 声の一覧はメタデータだけ読んで同期的に作る (NVDA が起動直後に一覧を要求するため)
		_load_onnxruntime(False)
		self._models_dirty = False
		self._voices = self._scan_voices(self._model_paths)
		if not self._voices:
			raise RuntimeError(_("There are no voice models that can be loaded."))
		self.voice = self._voices[0][0]

		self._player = nvwave.WavePlayer(
			channels=1,
			samplesPerSec=SAMPLE_RATE,
			bitsPerSample=16,
			outputDevice=config.conf["audio"]["outputDevice"],
		)
		self._synth_q: queue.Queue = queue.Queue()
		self._play_q: queue.Queue = queue.Queue()
		self._synth_thread = threading.Thread(target=self._synth_loop, name="DASH.Synth", daemon=True)
		self._play_thread = threading.Thread(target=self._play_loop, name="DASH.Play", daemon=True)
		self._synth_thread.start()
		self._play_thread.start()
		threading.Thread(target=self._init_core, name="DASH.Init", daemon=True).start()

	# ---- コア初期化 ----
	def _init_core(self, use_gpu: bool | None = None):
		from .aivis_core import AivisCore, SynthParams

		self._core_ready.clear()
		self._core_error = None
		try:
			t = time.perf_counter()
			gpu = self.use_gpu if use_gpu is None else use_gpu
			_load_onnxruntime(gpu)
			if gpu and _CUDA_PACK is not None and not _cuda_dlls_loaded:
				gpu = False  # CUDA の DLL が読み込めなかった (ログに記録済み)。CPU で動かす
			core = AivisCore(self._bert_dir, self._model_paths, use_gpu=gpu)
			defaults, users = self._apply_dictionaries(core)
			# 文脈による読み分け (tsqyomi)。読み込めなくても読み上げはできる
			tsqyomi_dir = paths.find_tsqyomi_dir()
			tsqyomi_ok = tsqyomi_dir is not None and core.load_tsqyomi(tsqyomi_dir)
			self._core = core
			# 選択中の声のモデルを先読みし、Sudachi 辞書なども含めて一度合成しておく
			vid = self.voice if self.voice in core.styles else next(iter(core.styles))
			core.get_model(core.styles[vid].model_uuid)
			core.synthesize("風が強い。", SynthParams(voice_id=vid))
			log.info(
				f"DASH: core ready in {time.perf_counter() - t:.1f}s "
				f"(bert={self._bert_dir}, models={len(self._model_paths)}, dictionaries={len(defaults)}+{len(users)}, "
				f"tsqyomi={tsqyomi_ok}, "
				f"providers={core.providers})"
			)
		except BaseException as e:
			log.error("DASH: core initialization failed", exc_info=True)
			self._core_error = e
		finally:
			self._core_ready.set()

	def _wait_core(self):
		if not self._core_ready.wait(timeout=300):
			raise RuntimeError("DASH core did not initialize in time")
		if self._core_error is not None:
			raise RuntimeError(f"DASH core failed to initialize: {self._core_error}")
		return self._core

	# ---- 設定 ----
	def voices(self):
		return list(self._voices)

	def set_voice(self, voice: str):
		if voice not in dict(self._voices):
			voice = self._voices[0][0]
		self.voice = voice
		# モデルの切り替え (数秒) は合成スレッドで先に済ませておく
		self._synth_q.put(("preload", voice, self._gen))

	def set_use_gpu(self, value: bool):
		if value == self.use_gpu:
			return
		self.use_gpu = value
		if self._core_ready.is_set():
			threading.Thread(target=self._init_core, args=(value,), name="DASH.Init", daemon=True).start()

	@staticmethod
	def _fast_gpu(core) -> bool:
		"""CUDA で合成しているか (AIVIS 側の話速を変えずに Sonic だけで 5 倍まで出せる)。

		DirectML はデコーダが CPU と大差ない (RTF 約 0.1) ので含めない。
		"""
		p = core.providers[0]
		return (p if isinstance(p, str) else p[0]) == "CUDAExecutionProvider"

	def _params(self, core, rate: int, volume: int):
		from .aivis_core import SynthParams

		# AIVIS 側の話速は原則 1.0 (速くすると聞き取りにくくなる)。速度は合成後に Sonic で変える。
		# CPU で SONIC_MAX_SPEED を超える分と、Sonic が使えない環境ではモデル側の話速を使う
		speed, _ = split_speed(rate, _sonic_available(), self._fast_gpu(core))
		# 抑揚 0..100 → AivisSpeech の intonationScale 0..2 → スタイルの強さ (AivisSpeech Engine と同じ変換)
		iscale = self.inflection / 50.0
		weight = iscale if iscale <= 1.0 else 1.0 + (iscale - 1.0) * 9.0
		return SynthParams(voice_id=self.voice, speed=speed, style_weight=weight, volume=volume / 100.0)

	# ---- 読み上げ ----
	def speak(self, seq):
		self._synth_q.put(("speak", list(seq), self._gen))

	def cancel(self):
		self._gen += 1
		self._queued_until = 0.0
		for q in (self._synth_q, self._play_q):
			try:
				while True:
					q.get_nowait()
			except queue.Empty:
				pass
		self._player.stop()

	def pause(self, switch: bool):
		self._player.pause(switch)

	def terminate(self):
		self.cancel()
		self._synth_q.put(None)
		self._play_q.put(None)
		self._synth_thread.join(timeout=5)
		self._play_thread.join(timeout=5)
		try:
			self._player.close()
		except Exception:
			pass
		self._core = None

	@staticmethod
	def _scan_voices(model_paths) -> list[tuple[str, str]]:
		"""モデルのメタデータから (声の ID, 表示名) の一覧を作る。同じモデル (UUID) は先に見つかったものを使う。"""
		from .aivis_core.engine import list_styles, read_aivmx_header

		voices: list[tuple[str, str]] = []
		seen = set()
		for p in model_paths:
			try:
				manifest, _, _ = read_aivmx_header(p)
			except Exception:
				log.exception(f"DASH: failed to read {p}")
				continue
			if manifest["uuid"] in seen:
				continue
			seen.add(manifest["uuid"])
			for st in list_styles(manifest):
				voices.append((st.voice_id, st.display_name))
		return voices

	def refresh_models(self) -> None:
		"""音声合成モデルを追加・削除したあとに呼ぶ (NVDA メニューからダウンロード・削除した直後など)。

		声の一覧 (NVDA の音声設定の「声」) はすぐに作り直す。合成コアへの反映は、合成中に差し替えないよう
		合成スレッドが次の仕事を始める前に行う (辞書の再適用と同じく印で伝える)。
		"""
		model_paths = paths.find_models()
		voices = self._scan_voices(model_paths)
		if not voices:
			return  # 全部消えることは無い (同梱のモデルがある) が、念のため今の一覧を残す
		self._model_paths = model_paths
		self._voices = voices
		self._models_dirty = True
		if self.voice not in dict(voices):
			# 使っていた声のモデルが削除された
			self.set_voice(voices[0][0])

	@staticmethod
	def _apply_dictionaries(core):
		"""見つかった辞書を OpenJTalk に適用する (適用は 0.01 秒程度)。(既定辞書, ユーザー辞書) を返す。"""
		defaults, users = paths.find_dictionaries()
		try:
			core.apply_dictionaries(defaults, users)
		except Exception:
			log.exception("DASH: failed to apply dictionaries")
		return defaults, users

	def reload_dictionaries(self):
		"""辞書が増えた・変わったときに呼ぶ (NVDA メニューから辞書をインストールした直後など)。

		合成中に辞書を差し替えないよう、合成スレッドが次の仕事を始める前に適用する。
		cancel() は合成キューを空にするので、キューではなく印で伝える。
		"""
		self._dicts_dirty = True

	def _synth_loop(self):
		while True:
			job = self._synth_q.get()
			if job is None:
				return
			if getattr(self, "_dicts_dirty", False) and self._core is not None:
				self._dicts_dirty = False
				defaults, users = self._apply_dictionaries(self._core)
				log.info(f"DASH: dictionaries reloaded ({len(defaults)}+{len(users)})")
			if self._models_dirty and self._core is not None:
				self._models_dirty = False
				self._core.set_model_paths(self._model_paths)
				log.info(f"DASH: models reloaded ({len(self._core.model_paths)} models)")
			kind, payload, gen = job
			if gen != self._gen:
				continue
			try:
				core = self._wait_core()
				if kind == "preload":
					if payload in core.styles:
						core.get_model(core.styles[payload].model_uuid)
				elif kind == "speak":
					self._speak_seq(core, payload, gen)
			except Exception:
				log.error("DASH: synthesis failed", exc_info=True)
				if gen == self._gen:
					self._play_q.put(("done", None, gen))

	def _speak_seq(self, core, seq, gen):
		cancelled = lambda: gen != self._gen  # noqa: E731
		# 速度・音量コマンドを先に反映して、テキストごとの合成パラメータを決めておく
		plan = []
		rate, pitch, volume = self.rate, self.pitch, self.volume
		for item in seq:
			if isinstance(item, str):
				if item.strip():
					plan.append((item, rate, pitch, core.begin(item, self._params(core, rate, volume), cancelled)))
			elif isinstance(item, RateCommand):
				rate = _prosody(item, self.rate)
				plan.append((item, rate, pitch, None))
			elif isinstance(item, PitchCommand):
				pitch = _prosody(item, self.pitch)
			elif isinstance(item, VolumeCommand):
				volume = _prosody(item, self.volume)
			else:
				plan.append((item, rate, pitch, None))
		sonic = None  # 発話 (speak 1 回) ごとに作る
		for item, rate, pitch, utt in plan:
			if cancelled():
				return
			if isinstance(item, str):
				_, speed = split_speed(rate, _sonic_available(), self._fast_gpu(core))
				pfactor = pitch_to_factor(pitch)
				use_sonic = (speed != 1.0 or pfactor != 1.0) and _sonic_available()
				if use_sonic:
					if sonic is None:
						sonic = _Speedup()
					sonic.set_speed(speed)
					sonic.set_pitch(pfactor)
				# 速く再生するほど最初に溜めた音がすぐ尽きるので、再生時間で 100ms 分 (最低 150ms) 溜める
				min_first = max(core.stream.min_first_ms, FIRST_BUFFER_PLAY_MS * speed) if use_sonic else None
				# 前の音声がまだ十分に残っているなら、最初から大きな区間でデコードする (のりしろの無駄が減る)
				lead = self._queued_until - time.perf_counter()
				first_frames = core.stream.chunk_frames if lead > 0.4 else 64 if lead > 0.25 else None
				for pcm in core.decode(utt, min_first_ms=min_first, first_chunk_frames=first_frames):
					if cancelled():
						return
					data = sonic.process(pcm) if use_sonic else pcm.tobytes()
					self._put_audio(data, gen)
				# テキストの区切り (この後にインデックスが来る) で Sonic の内部に残っている分を出し切る
				if use_sonic:
					self._put_audio(sonic.flush(), gen)
			elif isinstance(item, IndexCommand):
				self._play_q.put(("index", item.index, gen))
			elif isinstance(item, BreakCommand):
				n = int(SAMPLE_RATE * item.time / 1000 / rate_to_speed(rate))
				self._put_audio(b"\0\0" * n, gen)
		self._play_q.put(("done", None, gen))

	def _put_audio(self, data: bytes, gen: int):
		if not data:
			return
		# 再生待ちの音声がいつ終わるかの見積もり (次のテキストのデコードの刻み方を決めるのに使う)
		now = time.perf_counter()
		self._queued_until = max(now, self._queued_until) + len(data) / 2 / SAMPLE_RATE
		self._play_q.put(("audio", data, gen))

	def _play_loop(self):
		fed = False  # 今回の発話でまだ音を出していないなら、インデックスは即時通知する
		while True:
			item = self._play_q.get()
			if item is None:
				return
			kind, payload, gen = item
			if gen != self._gen:
				fed = False
				continue
			try:
				if kind == "audio":
					self._player.feed(payload)
					fed = True
					if gen != self._gen:
						# 世代を確認してから feed するまでの間に cancel() された場合、
						# 古い音声が stop() の後に入ってしまうので、もう一度止める
						self._player.stop()
						fed = False
				elif kind == "index":
					if fed:
						# 長さ 0 の feed は「ここまで再生されたら呼ぶ」コールバックとして働く (WASAPI)
						self._player.feed(b"", onDone=partial(self._on_index, payload))
					else:
						self._on_index(payload)
				elif kind == "done":
					if fed:
						self._player.idle()
					fed = False
					if gen == self._gen:
						self._on_done()
			except Exception:
				log.error("DASH: playback error", exc_info=True)
