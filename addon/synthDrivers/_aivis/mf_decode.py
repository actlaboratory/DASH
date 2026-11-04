# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""Windows の Media Foundation で音声ファイル (AivisHub の試聴用音声 m4a / AAC など) を 16 ビット PCM に変換する。

AivisHub の試聴用音声は m4a (AAC) だけで、NVDA は AAC を再生できない。MCI (winmm) でも m4a は開けなかった。
Media Foundation は Windows 10/11 に標準で入っていて AAC のデコーダーもあるので、Source Reader で PCM に
変換し、NVDA の nvwave.WavePlayer で再生する (NVDA の出力デバイス・音量で鳴る)。依存は ctypes だけ。

COM のインターフェイスは ctypes で vtable の番号を直接呼ぶ:
  IMFAttributes: GetUINT32=7, SetUINT32=21, SetGUID=24 (IMFMediaType も同じ)
  IMFSourceReader: SetCurrentMediaType=7, GetCurrentMediaType=6, ReadSample=9
  IMFSample: ConvertToContiguousBuffer=41 / IMFMediaBuffer: Lock=3, Unlock=4 / IUnknown: Release=2
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_uint32, c_uint64, c_void_p, wintypes


class GUID(ctypes.Structure):
	_fields_ = [("Data1", c_uint32), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

	@classmethod
	def parse(cls, s: str) -> "GUID":
		h = s.replace("-", "")
		return cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16),
			(ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:32])))


MF_MT_MAJOR_TYPE = GUID.parse("48eba18e-f8c9-4687-bf11-0a74c9f96a8f")
MF_MT_SUBTYPE = GUID.parse("f7e34c9a-42e8-4714-b74b-cb29d72c35e5")
MF_MT_AUDIO_NUM_CHANNELS = GUID.parse("37e48bf5-645e-4c5b-89de-ada9e29b696a")
MF_MT_AUDIO_SAMPLES_PER_SECOND = GUID.parse("5faeeae7-0290-4c31-9e8a-c534f68d9dba")
MF_MT_AUDIO_BITS_PER_SAMPLE = GUID.parse("f2deb57f-40fa-4764-aa33-ed4f2d1ff669")
MFMediaType_Audio = GUID.parse("73647561-0000-0010-8000-00aa00389b71")
MFAudioFormat_PCM = GUID.parse("00000001-0000-0010-8000-00aa00389b71")
MF_VERSION = 0x00020070
MF_SOURCE_READER_FIRST_AUDIO_STREAM = 0xFFFFFFFD
MF_SOURCE_READERF_ENDOFSTREAM = 0x2
COINIT_MULTITHREADED = 0x0


def _method(obj: c_void_p, index: int, *argtypes):
	"""COM オブジェクトの vtable の index 番目のメソッドを返す (戻り値は HRESULT)。"""
	vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
	proto = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)
	fn = proto(vtbl[index])
	return lambda *args: fn(obj, *args)


def _release(obj: c_void_p) -> None:
	if obj:
		vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
		ctypes.WINFUNCTYPE(c_uint32, c_void_p)(vtbl[2])(obj)


def decode_to_pcm16(path: str) -> tuple[bytes, int, int]:
	"""音声ファイルを 16 ビット PCM にして (データ, サンプリング周波数, チャンネル数) を返す。どのスレッドからでも呼べる。"""
	ole32 = ctypes.WinDLL("ole32")
	mfplat = ctypes.WinDLL("mfplat")
	mfreadwrite = ctypes.WinDLL("mfreadwrite")
	hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
	co_init = hr in (0, 1)  # S_OK / S_FALSE (既に初期化済み)。RPC_E_CHANGED_MODE なら今のスレッドの設定のまま使う
	mfplat.MFStartup(c_uint32(MF_VERSION), c_uint32(0))
	reader = c_void_p()
	mtype = c_void_p()
	cur = c_void_p()
	try:
		mfreadwrite.MFCreateSourceReaderFromURL.argtypes = [wintypes.LPCWSTR, c_void_p, POINTER(c_void_p)]
		mfreadwrite.MFCreateSourceReaderFromURL.restype = ctypes.HRESULT
		mfreadwrite.MFCreateSourceReaderFromURL(str(path), None, byref(reader))
		mfplat.MFCreateMediaType.argtypes = [POINTER(c_void_p)]
		mfplat.MFCreateMediaType.restype = ctypes.HRESULT
		mfplat.MFCreateMediaType(byref(mtype))
		set_guid = _method(mtype, 24, POINTER(GUID), POINTER(GUID))
		set_guid(byref(MF_MT_MAJOR_TYPE), byref(MFMediaType_Audio))
		set_guid(byref(MF_MT_SUBTYPE), byref(MFAudioFormat_PCM))
		_method(mtype, 21, POINTER(GUID), c_uint32)(byref(MF_MT_AUDIO_BITS_PER_SAMPLE), 16)
		# デコーダーの出力を 16 ビット PCM にする (サンプリング周波数・チャンネル数は元のまま)
		_method(reader, 7, c_uint32, c_void_p, c_void_p)(MF_SOURCE_READER_FIRST_AUDIO_STREAM, None, mtype)
		_method(reader, 6, c_uint32, POINTER(c_void_p))(MF_SOURCE_READER_FIRST_AUDIO_STREAM, byref(cur))
		get_u32 = _method(cur, 7, POINTER(GUID), POINTER(c_uint32))
		rate, channels = c_uint32(), c_uint32()
		get_u32(byref(MF_MT_AUDIO_SAMPLES_PER_SECOND), byref(rate))
		get_u32(byref(MF_MT_AUDIO_NUM_CHANNELS), byref(channels))
		read_sample = _method(reader, 9, c_uint32, c_uint32, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint64),
			POINTER(c_void_p))
		chunks: list[bytes] = []
		while True:
			index, flags, ts, sample = c_uint32(), c_uint32(), c_uint64(), c_void_p()
			read_sample(MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, byref(index), byref(flags), byref(ts), byref(sample))
			if sample:
				buf = c_void_p()
				try:
					_method(sample, 41, POINTER(c_void_p))(byref(buf))
					data, max_len, cur_len = ctypes.POINTER(ctypes.c_ubyte)(), c_uint32(), c_uint32()
					_method(buf, 3, POINTER(POINTER(ctypes.c_ubyte)), POINTER(c_uint32), POINTER(c_uint32))(
						byref(data), byref(max_len), byref(cur_len))
					chunks.append(ctypes.string_at(data, cur_len.value))
					_method(buf, 4)()
				finally:
					_release(buf)
					_release(sample)
			if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
				break
		return b"".join(chunks), rate.value, channels.value
	finally:
		_release(cur)
		_release(mtype)
		_release(reader)
		mfplat.MFShutdown()
		if co_init:
			ole32.CoUninitialize()
