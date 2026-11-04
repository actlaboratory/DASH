# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
# DASH - AivisSpeech (Style-Bert-VITS2 / AIVMX) streaming synthDriver for NVDA

import addonHandler
import wx
from autoSettingsUtils.driverSetting import BooleanDriverSetting
from logHandler import log
from speech.commands import BreakCommand, IndexCommand, PitchCommand, RateCommand, VolumeCommand
from synthDriverHandler import SynthDriver as BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthDoneSpeaking, synthIndexReached

from . import _aivis

try:
	addonHandler.initTranslation()
except BaseException:

	def _(x):
		return x


class SynthDriver(BaseSynthDriver):
	name = "dash"
	description = "DASH (AivisSpeech)"

	supportedSettings = (
		BaseSynthDriver.VoiceSetting(),
		BaseSynthDriver.RateSetting(),
		BaseSynthDriver.PitchSetting(),
		BaseSynthDriver.InflectionSetting(),
		BaseSynthDriver.VolumeSetting(),
		BooleanDriverSetting("useGpu", _("Use GPU (DirectML/CUDA)"), defaultVal=False),
	)
	supportedCommands = {IndexCommand, BreakCommand, RateCommand, PitchCommand, VolumeCommand}
	supportedNotifications = {synthIndexReached, synthDoneSpeaking}

	@classmethod
	def check(cls):
		try:
			return _aivis.paths.has_minimum_data()
		except Exception:
			log.exception("DASH: check failed")
			return False

	def __init__(self):
		super().__init__()
		try:
			self._rt = _aivis.Runtime(self._onIndex, self._onDone)
		except BaseException as e:
			log.error("DASH: failed to initialize", exc_info=True)
			wx.CallAfter(_errmsg, e)
			raise

	def _onIndex(self, index):
		synthIndexReached.notify(synth=self, index=index)

	def _onDone(self):
		synthDoneSpeaking.notify(synth=self)

	def speak(self, speechSequence):
		self._rt.speak(speechSequence)

	def cancel(self):
		self._rt.cancel()

	def pause(self, switch):
		self._rt.pause(switch)

	def terminate(self):
		self._rt.terminate()

	def _get_availableVoices(self):
		out = {}
		for vid, name in self._rt.voices():
			out[vid] = VoiceInfo(vid, name, "ja")
		return out

	def _get_voice(self):
		return self._rt.voice

	def _set_voice(self, voice):
		self._rt.set_voice(voice)

	def _get_rate(self):
		return self._rt.rate

	def _set_rate(self, value):
		self._rt.rate = value

	def _get_pitch(self):
		return self._rt.pitch

	def _set_pitch(self, value):
		self._rt.pitch = value

	def _get_inflection(self):
		return self._rt.inflection

	def _set_inflection(self, value):
		self._rt.inflection = value

	def _get_volume(self):
		return self._rt.volume

	def _set_volume(self, value):
		self._rt.volume = value

	def _get_useGpu(self):
		return self._rt.use_gpu

	def _set_useGpu(self, value):
		self._rt.set_use_gpu(bool(value))


def _errmsg(e):
	import gui

	msg = _("Could not load DASH.") + "\n" + str(e)
	try:
		gui.message.MessageDialog.alert(msg, _("Error"))
	except Exception:
		gui.messageBox(msg, _("Error"))
