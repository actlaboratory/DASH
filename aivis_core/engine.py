# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""AIVMX (Style-Bert-VITS2 ONNX) のストリーミング音声合成コア。

AIVMX は「テキスト特徴量 → 波形」を 1 回の推論で出す ONNX グラフなので、
文全体の推論が終わるまで 1 サンプルも得られない。これが AivisSpeech の遅延の原因。

そこでロード時にグラフを 2 つに分割する:
  front : テキスト特徴量 → 潜在変数 z (192ch × フレーム, 1 フレーム = hop_length サンプル) と話者埋め込み g
  dec   : (z, g) → 波形 (HiFi-GAN 系デコーダ。畳み込みのみで構成される)
dec は畳み込みだけなので、z をフレーム方向に「のりしろ」付きで切り出してデコードすれば、
一括デコードとほぼ同一の波形を区間ごとに得られる。先頭区間を短くすれば最初の音が早く出る。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .aivmx import read_onnx_metadata
from .frontend import TextFeatures, split_sentences, text_to_features
from .nlp import bert_models

log = logging.getLogger(__name__)

DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_STYLE_WEIGHT = 1.0


def make_session_options(threads: int = 0) -> ort.SessionOptions:
    so = ort.SessionOptions()
    if threads <= 0:
        # ハイブリッド CPU で全論理コアを使うと逆に遅くなる (E コアや他プロセスとの奪い合い) ので控えめに
        threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    # スピン待ちは他のスレッド (NVDA 本体、他セッション) の CPU を奪うので無効化
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.enable_cpu_mem_arena = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return so


def default_providers(use_gpu: bool) -> list[Any]:
    available = ort.get_available_providers()
    # NOTE: AivisSpeech Engine は arena_extend_strategy=kSameAsRequested を指定しているが、
    # 前段グラフ (flow) では巨大な一時テンソルの確保・解放が繰り返されて約 4 倍遅くなる
    # (800 フレームで 1015ms → 253ms, scripts/bench_front2.py)。メモリより速度を優先して既定の戦略を使う。
    cpu = "CPUExecutionProvider"
    if use_gpu and "CUDAExecutionProvider" in available:
        # HEURISTIC / EXHAUSTIVE は同じ形の入力を繰り返すなら速いが (log.txt)、
        # 実際は区間の長さが毎回変わり、新しい形のたびに探索が走るので DEFAULT の方が速い
        # (最初の音まで 90ms 対 123ms, scripts/bench_cuda.py の ALGO=)
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), cpu]
    if use_gpu and "DmlExecutionProvider" in available:
        return [("DmlExecutionProvider", {"device_id": 0}), cpu]
    return [cpu]


@dataclass
class StyleInfo:
    voice_id: str  # "<model uuid>/<speaker local id>/<style local id>"
    model_uuid: str
    model_name: str
    speaker_name: str
    speaker_local_id: int
    style_name: str
    style_local_id: int

    @property
    def display_name(self) -> str:
        return f"{self.speaker_name}({self.style_name})"


def read_aivmx_header(path: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """AIVMX から manifest / hyper_parameters / style_vectors を読む (重み本体は読まない)。"""
    props = read_onnx_metadata(path)
    manifest = json.loads(props["aivm_manifest"])
    hps = json.loads(props["aivm_hyper_parameters"])
    style_vectors = np.load(io.BytesIO(base64.b64decode(props["aivm_style_vectors"])))
    return manifest, hps, style_vectors


def list_styles(manifest: dict[str, Any]) -> list[StyleInfo]:
    out = []
    for spk in manifest["speakers"]:
        for st in spk["styles"]:
            out.append(
                StyleInfo(
                    voice_id=f"{manifest['uuid']}/{spk['local_id']}/{st['local_id']}",
                    model_uuid=manifest["uuid"],
                    model_name=manifest["name"],
                    speaker_name=spk["name"],
                    speaker_local_id=int(spk["local_id"]),
                    style_name=st["name"],
                    style_local_id=int(st["local_id"]),
                )
            )
    return out


class AivisModel:
    """1 つの AIVMX モデル。ロード時に front / dec の 2 グラフへ分割する。"""

    def __init__(
        self,
        path: Path,
        providers: Sequence[Any] | None = None,
        threads: int = 0,
        front_providers: Sequence[Any] | None = None,
    ) -> None:
        """front_providers: 前段 (enc + flow) だけ別の実行環境にする場合に指定 (既定は providers と同じ)。"""
        import onnx
        from onnx.utils import Extractor

        self.path = Path(path)
        m = onnx.load(str(self.path))
        props = {p.key: p.value for p in m.metadata_props}
        self.manifest = json.loads(props["aivm_manifest"])
        self.hps = json.loads(props["aivm_hyper_parameters"])
        self.style_vectors = np.load(io.BytesIO(base64.b64decode(props["aivm_style_vectors"])))
        self.sampling_rate = int(self.hps["data"]["sampling_rate"])
        self.hop_length = int(self.hps["data"]["hop_length"])
        self.add_blank = bool(self.hps["data"].get("add_blank", True))
        self.styles = list_styles(self.manifest)
        # style2id はハイパーパラメータ側のスタイル名 → ID (AIVM マニフェストの local_id と一致するはず)
        self.input_names = [i.name for i in m.graph.input]
        self.is_jp_extra = "bert" in self.input_names

        # デコーダの入力 (z と g) と出力を特定する
        z_name = g_name = None
        for node in m.graph.node:
            if node.name == "/dec/conv_pre/Conv":
                z_name = node.input[0]
            elif node.name == "/dec/cond/Conv":
                g_name = node.input[0]
        self.output_name = m.graph.output[0].name

        so = make_session_options(threads)
        providers = list(providers) if providers else default_providers(False)
        front_providers = list(front_providers) if front_providers else providers
        if z_name and g_name:
            ex = Extractor(m)
            front = ex.extract_model(self.input_names, [z_name, g_name])
            dec = ex.extract_model([z_name, g_name], [self.output_name])
            del ex, m
            self.front = ort.InferenceSession(front.SerializeToString(), so, providers=front_providers)
            del front
            self.dec = ort.InferenceSession(dec.SerializeToString(), so, providers=providers)
            del dec
            self.z_name, self.g_name = z_name, g_name
            self.streamable = True
        else:
            # 想定外の構造のモデルは分割せず一括推論する
            log.warning("decoder boundary not found; streaming disabled for %s", self.path)
            data = m.SerializeToString()
            del m
            self.front = ort.InferenceSession(data, so, providers=providers)
            self.dec = None
            self.streamable = False
        # 同一セッションの並列 run は避ける (ORT 自体はスレッドセーフだが CPU を食い合うだけ)。
        # front と dec は別のロックにして、次の文の前段と今の文のデコードを並行に実行できるようにする
        self.front_lock = threading.Lock()
        self.dec_lock = threading.Lock()

    def style_vector(self, style_local_id: int, weight: float) -> np.ndarray:
        mean = self.style_vectors[0]
        vec = self.style_vectors[style_local_id]
        return (mean + (vec - mean) * weight).astype(np.float32)

    def _front_feeds(
        self,
        f: TextFeatures,
        speaker_local_id: int,
        style_vec: np.ndarray,
        length_scale: float,
        sdp_ratio: float,
        noise: float,
        noise_w: float,
    ) -> dict[str, np.ndarray]:
        T = f.phones.shape[0]
        # 入力の並び順は style_bert_vits2.models.infer_onnx.infer_onnx() と同じ
        if self.is_jp_extra:
            berts = [f.bert[None]]
        else:
            zeros = np.zeros_like(f.bert)[None]
            berts = [zeros, f.bert[None], zeros]  # zh, ja, en
        values = [
            f.phones[None, :],
            np.array([T], dtype=np.int64),
            np.array([speaker_local_id], dtype=np.int64),
            f.tones[None, :],
            f.languages[None, :],
            *berts,
            style_vec[None, :],
            np.array(length_scale, dtype=np.float32),
            np.array(sdp_ratio, dtype=np.float32),
        ]
        # 古い形式の AIVMX には noise_scale / noise_scale_w の入力が無い (モデル内の固定値を使う)。
        # 無いものは渡さない (渡すと入力の数が合わず、その声では何も読めなかった)
        names = set(self.input_names)
        if "noise_scale" in names:
            values.append(np.array(noise, dtype=np.float32))
        if "noise_scale_w" in names:
            values.append(np.array(noise_w, dtype=np.float32))
        assert len(values) == len(self.input_names), self.input_names
        return dict(zip(self.input_names, values))

    def infer_stream(
        self,
        f: TextFeatures,
        speaker_local_id: int,
        style_local_id: int,
        *,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        length_scale: float = 1.0,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        first_chunk_frames: int = 16,
        chunk_frames: int = 64,
        margin_frames: int = 16,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[np.ndarray]:
        """float32 波形 (-1..1) を区間ごとに yield する。"""
        z, g = self.run_front(
            f, speaker_local_id, style_local_id, style_weight=style_weight, length_scale=length_scale,
            sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
        )
        yield from self.decode_stream(
            z, g, first_chunk_frames=first_chunk_frames, chunk_frames=chunk_frames,
            margin_frames=margin_frames, is_cancelled=is_cancelled,
        )

    def run_front(
        self,
        f: TextFeatures,
        speaker_local_id: int,
        style_local_id: int,
        *,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        length_scale: float = 1.0,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """前段を実行して (z, g) を返す。分割できないモデルでは (波形, None)。"""
        feeds = self._front_feeds(
            f, speaker_local_id, self.style_vector(style_local_id, style_weight),
            length_scale, sdp_ratio, noise, noise_w,
        )
        with self.front_lock:
            if not self.streamable:
                return self.front.run([self.output_name], feeds)[0], None
            z, g = self.front.run([self.z_name, self.g_name], feeds)
        return z, g

    def decode_stream(
        self,
        z: np.ndarray,
        g: np.ndarray | None,
        *,
        first_chunk_frames: int = 16,
        chunk_frames: int = 64,
        margin_frames: int = 16,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[np.ndarray]:
        """run_front() の結果を区間ごとにデコードして float32 波形を yield する。"""
        if g is None:
            yield z[0, 0]
            return
        total = z.shape[2]
        hop = self.hop_length
        pos = 0
        n = first_chunk_frames
        while pos < total:
            if is_cancelled is not None and is_cancelled():
                return
            end = min(total, pos + n)
            # 残りがわずかなら今回の区間に含めてしまう (細切れのデコードを避ける)
            if total - end < n // 2:
                end = total
            s = max(0, pos - margin_frames)
            e = min(total, end + margin_frames)
            with self.dec_lock:
                wav = self.dec.run([self.output_name], {self.z_name: z[:, :, s:e], self.g_name: g})[0]
            yield wav[0, 0, (pos - s) * hop : (end - s) * hop]
            pos = end
            # 区間長は 16→32→64 と段階的に伸ばす (先頭が無音でトリムされても次の音がすぐ出るように)
            n = min(chunk_frames, n * 2)


@dataclass
class SynthParams:
    voice_id: str = ""
    speed: float = 1.0  # 話速 (1.0 = 標準)
    style_weight: float = DEFAULT_STYLE_WEIGHT
    sdp_ratio: float = DEFAULT_SDP_RATIO
    noise: float = DEFAULT_NOISE
    noise_w: float = DEFAULT_NOISEW
    volume: float = 1.0
    # 本家は発話ごとにピーク正規化しているが、ストリーミングでは発話全体のピークが分からないので固定ゲインにする。
    # モデル出力のピークは概ね 0.25〜0.4 なので 2.4 倍で本家に近い音量になる。超えた分はソフトクリップする。
    gain: float = 2.4


@dataclass
class StreamConfig:
    first_chunk_frames: int = 16  # 先頭区間 (44.1kHz/hop512 で 1 フレーム ≒ 11.6ms)
    # 区間長の上限。区間ごとに前後 12 フレームののりしろを余分にデコードするので、大きいほど効率が良い。
    # 16→32→64→128 と伸ばす (64 止まりだと 4 倍速再生で合成が間に合わないことがあった)
    chunk_frames: int = 128
    # デコーダの受容野は 12 フレーム以内 (scripts/verify.py)。12 以上なら一括デコードと同一波形
    margin_frames: int = 12
    # 文の分割。前段 (flow) の処理時間は文長に対して超線形に増える (80 文字で 0.6 秒) ので、
    # 先頭の文は読点で短く区切って再生開始を早め、極端に長い文も読点で区切る
    max_sentence_len: int = 80
    first_sentence_max_len: int = 30
    # 無音判定の閾値 (モデル生の出力スケール。本家はピーク正規化後に 0.0004)
    trim_threshold: float = 0.00015
    # 最初に出す音の最小の長さ (これより短いと次の区間が間に合わず音切れする)
    min_first_ms: int = 150


def _to_int16(x: np.ndarray, scale: float) -> np.ndarray:
    """ゲインをかけて int16 にする。|y| > 0.85 はソフトクリップ。"""
    y = x * scale
    a = np.abs(y)
    knee = 0.85
    over = a > knee
    if over.any():
        y[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return (y * 32767).astype(np.int16)


class AivisCore:
    """複数 AIVMX モデルを管理し、テキストからストリーミング合成する。"""

    def __init__(
        self,
        bert_dir: Path,
        model_paths: Sequence[Path],
        use_gpu: bool = False,
        threads: int = 0,
        disable_power_throttling: bool = True,
    ) -> None:
        if disable_power_throttling:
            # バックグラウンドプロセス (NVDA) が EcoQoS で 3〜4 倍遅くなるのを防ぐ (scripts/profile_latency.py)
            from .winperf import set_process_power_throttling

            try:
                set_process_power_throttling(False)
            except Exception:
                log.exception("failed to disable power throttling")
        self.use_gpu = use_gpu
        self.threads = threads
        self.providers = default_providers(use_gpu)
        # GPU を使う場合も前段 (enc + flow) は CPU で動かす。入力の長さが毎回違い細かい演算が多いので
        # GPU ではかえって遅い (CUDA 121ms / DirectML 220ms に対し CPU 43ms, scripts/bench_cuda.py, bench_dml.py)。
        # BERT とデコーダは GPU が速い (CUDA: BERT 47→8ms, デコーダ RTF 0.12→0.05)
        self.front_providers = ["CPUExecutionProvider"]
        # NOTE: GPU はしばらく (1〜2 秒) 使わないと省電力状態に落ち、次の 1 回目が遅い (最初の音まで 約 40ms → 約 250ms。
        # CPU だけなら 約 110ms。scripts/bench_gpu_idle.py)。待機後の 1 文目だけ BERT を CPU で計算すると 約 110〜125ms に
        # なるが、CPU 用の BERT の控えで 1.3GB (int8 でも 0.4GB で、抑揚が変わる) 余分に使うので入れていない。
        # 小さな推論を流して GPU を起こし続ける・デコーダだけ CPU・NVIDIA の電源管理モードは、どれも効果が無かった。
        bert_models.init(Path(bert_dir), make_session_options(threads), self.providers)
        self.styles: dict[str, StyleInfo] = {}
        self.model_paths: dict[str, Path] = {}
        self._models: dict[str, AivisModel] = {}
        self._models_lock = threading.Lock()
        self.set_model_paths(model_paths)
        self.stream = StreamConfig()

    def set_model_paths(self, model_paths: Sequence[Path]) -> None:
        """使う音声合成モデル (.aivmx) の一覧を設定し直す (モデルの追加・削除のあと)。

        同じモデル (UUID) が複数の場所にあれば、先に渡されたものを使う (呼び出し側で優先順に並べる)。
        一覧から消えたモデルは、読み込み済みなら解放する。合成と同じスレッドから呼ぶこと。
        """
        paths: dict[str, Path] = {}
        styles: dict[str, StyleInfo] = {}
        for p in model_paths:
            try:
                manifest, _, _ = read_aivmx_header(Path(p))
            except Exception:
                log.exception("failed to read %s", p)
                continue
            if manifest["uuid"] in paths:
                continue
            paths[manifest["uuid"]] = Path(p)
            for st in list_styles(manifest):
                styles[st.voice_id] = st
        with self._models_lock:
            for uuid in [u for u in self._models if paths.get(u) != self.model_paths.get(u)]:
                del self._models[uuid]
        self.model_paths = paths
        self.styles = styles

    @staticmethod
    def load_tsqyomi(model_dir: Path, threads: int = 4) -> bool:
        """tsqyomi (pyopenjtalk-plus の文脈による読み分けモデル) をローカルのフォルダから読み込む。

        読み込めれば以後の g2p で使う (frontend.text_to_features が tsqyomi.is_model_loaded() を見る)。
        失敗しても tsqyomi なしで合成できるので False を返すだけにする (AivisSpeech Engine と同じ)。
        AivisSpeech Engine と同じく CPU で動かす (軽量で、g2p の時間は中央値 0.4→0.6ms, scripts/bench_tsqyomi.py)。

        pyopenjtalk-plus は SessionOptions を指定せずにセッションを作るため、既定では全コア分のスレッドが
        推論後もスピン待ちをして、直後の BERT・デコーダと CPU を取り合う。読み込む間だけ
        onnxruntime.InferenceSession を差し替えて、このプロジェクトと同じ設定 (スピン待ちなし、スレッド数を絞る) を渡す。
        """
        from pyopenjtalk import tsqyomi

        if tsqyomi.is_model_loaded():
            return True
        orig = ort.InferenceSession
        so = make_session_options(threads)

        def session_with_options(path_or_bytes, sess_options=None, providers=None, **kwargs):
            return orig(path_or_bytes, sess_options or so, providers=providers, **kwargs)

        ort.InferenceSession = session_with_options
        try:
            tsqyomi.load_model(
                onnx_providers=["CPUExecutionProvider"], model_dir=Path(model_dir), allow_provider_fallback=False
            )
            return True
        except Exception:
            log.exception("failed to load tsqyomi from %s; reading disambiguation is disabled", model_dir)
            return False
        finally:
            ort.InferenceSession = orig

    @staticmethod
    def apply_dictionaries(dic_paths: Sequence[Path], user_dic_paths: Sequence[Path] = ()) -> None:
        """OpenJTalk にコンパイル済み辞書 (.dic) を適用する。

        dic_paths: AivisSpeech Engine の resources/dictionaries/*.dic のような既定辞書 (名前順に適用)
        user_dic_paths: ユーザー辞書 (AivisSpeech Engine の user.dic など)。読みが後段で補正されないよう保護する
        """
        import pyopenjtalk

        entries = [{"dic_path": str(Path(p).resolve()), "is_reading_protected": False} for p in sorted(dic_paths)]
        entries += [{"dic_path": str(Path(p).resolve()), "is_reading_protected": True} for p in user_dic_paths]
        if entries:
            pyopenjtalk.update_global_jtalk_with_user_dict(entries)
        else:
            pyopenjtalk.unset_user_dict()

    def get_model(self, uuid: str) -> AivisModel:
        with self._models_lock:
            m = self._models.get(uuid)
            if m is None:
                m = AivisModel(self.model_paths[uuid], self.providers, self.threads, self.front_providers)
                # 同時に保持するモデルは 1 つだけ (メモリ節約)
                self._models.clear()
                self._models[uuid] = m
            return m

    # ---- 合成 ----
    # NOTE: 文ごとの前段 (g2p + BERT + front) を別スレッドで先回りしてデコードと並行させる案も試したが、
    # ハイブリッド CPU (P/E コア) では 2 つの推論が CPU を奪い合って両方遅くなり、
    # 最初の音までの時間も音切れも悪化した (前段のスレッド数を 4/2 に絞っても同様)。
    # そのため前段は decode() の中で、必要になった時点で同じスレッドで実行する。

    def begin(
        self,
        text: str,
        params: SynthParams,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> "Utterance":
        """合成するテキストを文に分けてハンドルを返す。音声は decode() で取り出す。"""
        st = self.styles[params.voice_id]
        model = self.get_model(st.model_uuid)
        cfg = self.stream
        sentences = split_sentences(text, cfg.max_sentence_len, cfg.first_sentence_max_len)

        def prep(sentence: str):
            if is_cancelled is not None and is_cancelled():
                return None
            f = text_to_features(sentence, add_blank=model.add_blank, use_jp_extra=model.is_jp_extra)
            if f is None:
                return None
            return model.run_front(
                f, st.speaker_local_id, st.style_local_id,
                style_weight=params.style_weight, length_scale=1.0 / max(0.1, params.speed),
                sdp_ratio=params.sdp_ratio, noise=params.noise, noise_w=params.noise_w,
            )

        steps = [partial(prep, s) for s in sentences]
        return Utterance(model, params, steps, is_cancelled)

    def decode(
        self,
        utt: "Utterance",
        min_first_ms: float | None = None,
        first_chunk_frames: int | None = None,
    ) -> Iterator[np.ndarray]:
        """begin() したものをデコードし、int16 PCM (モノラル) の ndarray を区間ごとに yield する。

        最初の出力だけは min_first_ms (既定 StreamConfig.min_first_ms) 以上の長さになるまで溜めてから出す。
        先頭の無音をトリムすると最初の区間が数十 ms しか残らないことがあり、
        そのまま再生を始めると次の区間のデコードが間に合わず音が途切れるため (scripts/nvda_harness.py で確認)。
        後段で高速再生する場合は、その倍率をかけた値を渡す。

        first_chunk_frames: 最初にデコードする区間の長さ。既定では min_first_ms 分 (最低 StreamConfig.first_chunk_frames)。
        16→32→… と小さく刻むと区間ごとにのりしろ (前後 12 フレーム) を余分にデコードするので、
        どうせ min_first_ms 分は溜めるなら最初から 1 回でデコードした方が速い。
        直前の音声がまだ再生中で急がない場合 (連続読みの 2 つ目以降のテキスト) は大きな値を渡すと効率が良い。
        """
        if min_first_ms is None:
            min_first_ms = self.stream.min_first_ms
        sr, hop = utt.model.sampling_rate, utt.model.hop_length
        min_first = int(min_first_ms * sr / 1000)
        if first_chunk_frames is None:
            # 先頭の無音 (トリムされる) の分を少し足しておく
            first_chunk_frames = -(-min_first // hop) + 4
        first_chunk_frames = max(self.stream.first_chunk_frames, min(self.stream.chunk_frames, first_chunk_frames))
        buf: list[np.ndarray] | None = []
        n = 0
        for chunk in self._stream_raw(utt, first_chunk_frames):
            if buf is None:
                yield chunk
                continue
            buf.append(chunk)
            n += chunk.size
            if n >= min_first:
                yield np.concatenate(buf)
                buf = None
        if buf:
            yield np.concatenate(buf)

    def _stream_raw(self, utt: "Utterance", first_chunk_frames: int) -> Iterator[np.ndarray]:
        cfg = self.stream
        model, params, is_cancelled = utt.model, utt.params, utt.is_cancelled
        scale = params.gain * params.volume
        started = False
        last = len(utt.steps) - 1
        for si, step in enumerate(utt.steps):
            if is_cancelled is not None and is_cancelled():
                return
            res = step()
            if res is None:
                continue
            z, g = res
            pending_silence = 0
            for chunk in model.decode_stream(
                z, g,
                first_chunk_frames=first_chunk_frames if not started else cfg.chunk_frames,
                chunk_frames=cfg.chunk_frames,
                margin_frames=cfg.margin_frames,
                is_cancelled=is_cancelled,
            ):
                # 発話先頭の無音は捨てる (AivisSpeech Engine の _trim_silence 相当)
                if not started:
                    idx = np.flatnonzero(np.abs(chunk) > cfg.trim_threshold)
                    if idx.size == 0:
                        continue
                    chunk = chunk[idx[0] :]
                    started = True
                # 末尾の無音は、後ろに音が続くと分かるまで保留する
                idx = np.flatnonzero(np.abs(chunk) > cfg.trim_threshold)
                if idx.size == 0:
                    pending_silence += chunk.size
                    continue
                if pending_silence:
                    yield np.zeros(pending_silence, dtype=np.int16)
                    pending_silence = 0
                yield _to_int16(chunk[: idx[-1] + 1], scale)
                pending_silence = chunk.size - (idx[-1] + 1)
            # 文と文の間は、文末の無音のうち最大 0.25 秒だけ残す (句読点の間として)
            if si != last and pending_silence:
                keep = min(pending_silence, int(model.sampling_rate * 0.25 / max(0.1, params.speed)))
                yield np.zeros(keep, dtype=np.int16)

    def synthesize_stream(
        self,
        text: str,
        params: SynthParams,
        is_cancelled: Callable[[], bool] | None = None,
        min_first_ms: float | None = None,
    ) -> Iterator[np.ndarray]:
        """begin() + decode()。"""
        yield from self.decode(self.begin(text, params, is_cancelled), min_first_ms)

    def synthesize(self, text: str, params: SynthParams) -> np.ndarray:
        parts = list(self.synthesize_stream(text, params))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


@dataclass
class Utterance:
    model: AivisModel
    params: SynthParams
    steps: list  # 文ごとの前段。呼ぶと (z, g) か None を返す
    is_cancelled: Callable[[], bool] | None = None

