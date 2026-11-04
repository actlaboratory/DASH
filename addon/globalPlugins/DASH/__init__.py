# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
# DASH - NVDA メニューに「DASH」を追加し、アップデートの確認 (VSU と同じアップデーター) と
# CUDA 加速のダウンロード・削除を行う

import threading
import time

import addonHandler
import config
import globalPluginHandler
import globalVars
import gui
import wx
from logHandler import log

from . import compat, updater

try:
	addonHandler.initTranslation()
except BaseException:

	def _(x):
		return x


def _hub_label(text):
	"""model_hub のラベル (N_ で目印を付けた英語) を、画面に出すときに翻訳する。AivisHub の生の値はそのまま。"""
	return _(text) if text else text


confspec = {
	"checkForUpdatesOnStartup": "boolean(default=True)",
}
config.conf.spec["DASH_global"] = confspec


class ProgressDialog:
	"""ダウンロードの進捗ダイアログ。NVDA 自身の更新のダウンロード (updateCheck.UpdateDownloader) と同じく
	wx.ProgressDialog を同じ手順 (prePopup → 作成 → CentreOnScreen / Raise → Update → Hide / Destroy → postPopup) で使う。
	NVDA 標準の部品なので、文言と進捗が読み上げられる。

	update / close はメインスレッドで動かす。ダウンロードのスレッドからは update_later() / close_and_wait() を使う。

	注意: wx.ProgressDialog.Update() は画面を固まらせないよう、中で一時的にイベントを処理する (yield)。
	そのため Update の途中で、後から予約した close や完了メッセージまで実行されることがある。
	Update の途中でダイアログを破棄すると、Update から戻ったところで破棄済みのダイアログに触れて NVDA ごと落ちる
	(実機で発生)。Update の実行中に来た close は、Update が戻るまで遅らせる。
	"""

	def __init__(self, title, message):
		self.cancelled = False
		self._updating = False
		self._close_requested = False
		self._closed = threading.Event()
		gui.mainFrame.prePopup()
		self._dlg = wx.ProgressDialog(
			title,
			message,
			style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE,
			parent=gui.mainFrame,
		)
		self._dlg.CentreOnScreen()
		self._dlg.Raise()

	def update(self, pct, message):
		# 閉じた後・閉じる予定のとき・Update の中から入れ子で呼ばれたときは何もしない
		if self._dlg is None or self._close_requested or self._updating:
			return
		self._updating = True
		try:
			# Update は「続けるか」を返す。キャンセルボタンが押されると False になる
			cont, _skip = self._dlg.Update(min(pct, 99), message)
		finally:
			self._updating = False
		if not cont:
			self.cancelled = True
		if self._close_requested:
			self._close_now()

	def close(self):
		self._close_requested = True
		if self._updating:
			return  # Update が戻ったら update() の最後で閉じる
		self._close_now()

	def _close_now(self):
		if self._dlg is not None:
			dlg, self._dlg = self._dlg, None
			dlg.Hide()
			dlg.Destroy()
			wx.CallLater(50, gui.mainFrame.postPopup)
		self._closed.set()

	def update_later(self, pct, message):
		"""ダウンロードのスレッドから呼ぶ。"""
		wx.CallAfter(self.update, pct, message)

	def close_and_wait(self, timeout=10.0):
		"""ダウンロードのスレッドから呼ぶ。ダイアログが実際に閉じるまで待つ (このあとで完了メッセージなどを出すため)。"""
		wx.CallAfter(self.close)
		self._closed.wait(timeout)


class SamplePlayer:
	"""AivisHub の試聴用音声を NVDA の出力デバイスで鳴らす (nvwave.WavePlayer)。

	音声は m4a なので model_hub.fetch_sample が Media Foundation で PCM に変換する (取得したものは覚えておく)。
	新しい再生を始めると前の再生は止まる。0.25 秒ずつ流すので、止めるとすぐ止まる。
	"""

	def __init__(self):
		self._cache: dict = {}
		self._gen = 0
		self._player = None
		self._lock = threading.Lock()
		self.playing_url = None

	def play(self, url, on_error=None):
		with self._lock:
			self._gen += 1
			gen = self._gen
		self._stop_player()
		self.playing_url = url
		threading.Thread(target=self._run, args=(url, gen, on_error), name="DASH.Sample", daemon=True).start()

	def stop(self):
		with self._lock:
			self._gen += 1
		self.playing_url = None
		self._stop_player()

	def _stop_player(self):
		p = self._player
		if p is not None:
			try:
				p.stop()
			except Exception:
				pass

	def _run(self, url, gen, on_error):
		import nvwave
		from synthDrivers._aivis import model_hub

		try:
			pcm, rate, channels = model_hub.fetch_sample(url, self._cache)
			if gen != self._gen:
				return
			player = nvwave.WavePlayer(
				channels=channels, samplesPerSec=rate, bitsPerSample=16,
				outputDevice=config.conf["audio"]["outputDevice"],
			)
			self._player = player
			step = rate * channels * 2 // 4
			for i in range(0, len(pcm), step):
				if gen != self._gen:
					break
				player.feed(pcm[i:i + step])
			if gen == self._gen:
				player.idle()
			player.close()
		except Exception as e:
			log.error("DASH: voice sample playback failed", exc_info=True)
			if on_error is not None and gen == self._gen:
				wx.CallAfter(on_error, e)
		finally:
			if gen == self._gen:
				self.playing_url = None
				self._player = None


class ModelDownloadDialog(wx.Dialog):
	"""AivisHub の音声合成モデルを選んでダウンロードするダイアログ。

	NVDA 自身の設定画面と同じく gui.guiHelper でラベル付きの標準の wx の部品を並べる (スクリーンリーダーで扱えるように)。
	"""

	def __init__(self, plugin):
		super().__init__(gui.mainFrame, title=_("Download voice models (AivisHub)"))
		from synthDrivers._aivis import model_hub, paths

		self._plugin = plugin
		self._hub = model_hub
		self._paths = paths
		self._models: list = []
		self._sort_keys = list(model_hub.SORTS)
		main = wx.BoxSizer(wx.VERTICAL)
		sh = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)
		self.sortChoice = sh.addLabeledControl(_("&Sort by:"), wx.Choice, choices=[_hub_label(v) for v in model_hub.SORTS.values()])
		self.sortChoice.SetSelection(0)
		self.sortChoice.Bind(wx.EVT_CHOICE, lambda evt: self._load())
		self.keywordEdit = sh.addLabeledControl(_("&Keyword:"), wx.TextCtrl, style=wx.TE_PROCESS_ENTER)
		self.keywordEdit.Bind(wx.EVT_TEXT_ENTER, lambda evt: self._load())
		searchButton = sh.addItem(wx.Button(self, label=_("&Find")))
		searchButton.Bind(wx.EVT_BUTTON, lambda evt: self._load())
		self.modelList = sh.addLabeledControl(_("&Models (press Space to preview):"), wx.ListBox, size=(560, 220))
		self.modelList.Bind(wx.EVT_LISTBOX, lambda evt: self._on_model_selected())
		self.modelList.Bind(wx.EVT_LISTBOX_DCLICK, lambda evt: self._download())
		self.modelList.Bind(wx.EVT_KEY_DOWN, self._on_model_key)
		# 試聴するスタイル (選んだモデルの「話者: スタイル」。スペースキーかエンターで試聴)
		self.styleList = sh.addLabeledControl(_("S&tyles (press Space or Enter to preview):"), wx.ListBox, size=(560, 110))
		self.styleList.Bind(wx.EVT_KEY_DOWN, self._on_style_key)
		self.styleList.Bind(wx.EVT_LISTBOX_DCLICK, lambda evt: self._play_style(toggle=False))
		self._player = SamplePlayer()
		self.detailEdit = sh.addLabeledControl(
			_("Details and l&icense:"), wx.TextCtrl, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(560, 220)
		)
		bh = gui.guiHelper.ButtonHelper(wx.HORIZONTAL)
		self.playButton = bh.addButton(self, label=_("&Preview"))
		self.playButton.Bind(wx.EVT_BUTTON, lambda evt: self._play_style(toggle=True))
		self.downloadButton = bh.addButton(self, label=_("&Download"))
		self.downloadButton.Bind(wx.EVT_BUTTON, lambda evt: self._download())
		closeButton = bh.addButton(self, id=wx.ID_CLOSE, label=_("Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda evt: self.Close())
		sh.addDialogDismissButtons(bh)
		main.Add(sh.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizer(main)
		main.Fit(self)
		self.SetEscapeId(wx.ID_CLOSE)
		self.Bind(wx.EVT_CLOSE, self._on_close)
		self.CentreOnScreen()
		self._load()

	def _on_close(self, evt):
		self._player.stop()
		self.Destroy()
		wx.CallLater(50, gui.mainFrame.postPopup)

	# ---- 試聴 ----
	def _on_model_selected(self):
		# 別のモデルに移ったら試聴を止め、スタイルの一覧と詳細を切り替える
		self._player.stop()
		m = self._selected()
		samples = m["samples"] if m else []
		self.styleList.Set([f"{spk}: {style}" for spk, style, _url, _t in samples] or [_("No preview audio is available")])
		if samples:
			self.styleList.SetSelection(0)
		self._show_detail()

	def _on_model_key(self, evt):
		if evt.GetKeyCode() == wx.WXK_SPACE and not evt.HasAnyModifiers():
			# モデル一覧でスペース: 最初のスタイルを試聴 (再生中ならもう一度押すと止める)
			self._play_sample(0, toggle=True)
			return
		evt.Skip()

	def _on_style_key(self, evt):
		if evt.GetKeyCode() in (wx.WXK_SPACE, wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER) and not evt.HasAnyModifiers():
			self._play_style(toggle=True)
			return
		evt.Skip()

	def _play_style(self, toggle):
		self._play_sample(max(0, self.styleList.GetSelection()), toggle)

	def _play_sample(self, index, toggle):
		import ui

		m = self._selected()
		if m is None or not m["samples"]:
			ui.message(_("No preview audio is available"))
			return
		index = min(index, len(m["samples"]) - 1)
		url = m["samples"][index][2]
		if toggle and self._player.playing_url == url:
			self._player.stop()
			return
		self._player.play(url, on_error=lambda e: ui.message(_("Could not play the preview audio")))

	def _installed(self) -> set:
		return self._hub.installed_uuids(self._paths.find_models())

	def _load(self):
		"""AivisHub から一覧を取得する (別スレッドで)。"""
		self._player.stop()
		self._models = []
		self.modelList.Set([_("Loading...")])
		self.styleList.Set([])
		self.detailEdit.SetValue("")
		self.downloadButton.Disable()
		sort = self._sort_keys[max(0, self.sortChoice.GetSelection())]
		keyword = self.keywordEdit.GetValue().strip() or None

		def work():
			try:
				models = self._hub.search(sort, keyword)
				installed = self._installed()
				wx.CallAfter(self._populate, models, installed)
			except Exception as e:
				log.error("DASH: AivisHub search failed", exc_info=True)
				wx.CallAfter(self._populate_error, e)

		threading.Thread(target=work, name="DASH.HubSearch", daemon=True).start()

	def _populate(self, models, installed):
		if not self:
			return
		self._models = models
		if not models:
			self.modelList.Set([_("No models found")])
			return
		self.modelList.Set([self._label(m, installed) for m in models])
		self.modelList.SetSelection(0)
		self.downloadButton.Enable()
		self._on_model_selected()

	def _populate_error(self, e):
		if not self:
			return
		self._models = []
		self.modelList.Set([_("Could not connect to AivisHub")])
		self.detailEdit.SetValue(str(e))

	@staticmethod
	def _label(m, installed) -> str:
		label = _("{name} ({author}) {license} {size} MB").format(
			name=m["name"], author=m["author"], license=m["license_type"], size=m["size"] // 2**20
		)
		if m["uuid"] in installed:
			label += _(" [installed]")
		return label

	def _selected(self):
		i = self.modelList.GetSelection()
		if 0 <= i < len(self._models):
			return self._models[i]
		return None

	def _show_detail(self):
		m = self._selected()
		if m is None:
			return
		# Translators: separator between style names of a speaker
		sep = _(", ")
		speakers = "\n".join(f"  {name}: {sep.join(styles)}" for name, styles in m["speakers"])
		text = _(
			"{name}\nAuthor: {author}\nCategory: {category} / Voice: {timbre}\n"
			"Version: {version} / Size: {size} MB / Downloads: {downloads} / Likes: {likes}\n\n"
			"Speakers and styles:\n{speakers}\n\nDescription:\n{description}\n\n"
			"License: {license}\n\n{license_text}"
		).format(
			name=m["name"], author=m["author"], category=_hub_label(m["category"]), timbre=_hub_label(m["timbre"]),
			version=m["version"], size=m["size"] // 2**20, downloads=m["downloads"], likes=m["likes"],
			speakers=speakers, description=m["description"], license=_hub_label(m["license_label"]),
			license_text=m["license_text"],
		)
		self.detailEdit.SetValue(text.replace("\n", "\r\n"))

	def _download(self):
		m = self._selected()
		if m is None or self._plugin._busy:
			return
		self._player.stop()
		msg = _(
			"\"{name}\" ({author}) will be downloaded (about {size} MB).\n\n"
			"License: {license}\n"
			"You can read the full license text in \"Details and license\". "
			"Please use the model in accordance with its license terms.\n\n"
			"Do you agree to the license and download the model?"
		).format(name=m["name"], author=m["author"], size=m["size"] // 2**20, license=_hub_label(m["license_label"]))
		if not compat.yesno(msg, _("Download voice model"), defaultNo=True):
			return
		self._plugin._busy = True
		dlg = ProgressDialog(_("Downloading voice model"), _("Connecting"))
		threading.Thread(target=self._do_download, args=(m, dlg), name="DASH.ModelDownload", daemon=True).start()

	def _do_download(self, m, dlg):
		last = [0.0]

		def progress(done, total):
			now = time.perf_counter()
			if now - last[0] < 0.5 and done < total:
				return
			last[0] = now
			text = _("Downloading {name}... {done:.0f} / {total:.0f} MB").format(
				name=m["name"], done=done / 2**20, total=total / 2**20
			)
			dlg.update_later(int(done * 100 / max(1, total)), text)

		try:
			self._hub.download(m, self._paths.user_data_dir() / "models", progress, lambda: dlg.cancelled)
		except self._hub.Cancelled:
			dlg.close_and_wait()
			return
		except Exception as e:
			log.error("DASH: model download failed", exc_info=True)
			dlg.close_and_wait()
			compat.messageBox(_("An error occurred while downloading:\n{e}").format(e=e), _("Error"), error=True)
			return
		finally:
			self._plugin._busy = False
		dlg.close_and_wait()
		rt = GlobalPlugin._running_runtime()
		if rt is not None:
			rt.refresh_models()
			msg = _("\"{name}\" has been added to the voice list. You can select it under \"Voice\" in NVDA's speech settings.")
		else:
			msg = _("\"{name}\" has been downloaded. It will be added to the DASH voice list.")
		compat.messageBox(msg.format(name=m["name"]), _("Download complete"))
		wx.CallAfter(self._refresh_installed_marks)

	def _refresh_installed_marks(self):
		if not self or not self._models:
			return
		i = self.modelList.GetSelection()
		installed = self._installed()
		self.modelList.Set([self._label(m, installed) for m in self._models])
		if i != wx.NOT_FOUND:
			self.modelList.SetSelection(i)


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	scriptCategory = _("DASH")

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._menu_item = None
		self._busy = False
		if globalVars.appArgs.secure:
			return
		if self.getUpdateCheckSetting() is True:
			self.autoUpdateChecker = updater.AutoUpdateChecker()
			self.autoUpdateChecker.autoUpdateCheck(mode=updater.AUTO)
		try:
			self._setup_menu()
		except Exception:
			log.error("DASH: failed to add the menu", exc_info=True)

	def terminate(self):
		super().terminate()
		if self._menu_item is not None:
			try:
				gui.mainFrame.sysTrayIcon.menu.Remove(self._menu_item)
			except Exception:
				pass

	def _setup_menu(self):
		menu = wx.Menu()
		# アップデート (VSU と同じ)
		self.updateCheckToggleItem = menu.Append(
			wx.ID_ANY, self.updateCheckToggleString(), _("Toggles update checking on startup.")
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.toggleUpdateCheck, self.updateCheckToggleItem)
		item = menu.Append(wx.ID_ANY, _("Check for updates"), _("Checks for new updates manually."))
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.performUpdateCheck, item)
		menu.AppendSeparator()
		# 音声合成モデル (AivisHub)
		item = menu.Append(
			wx.ID_ANY,
			_("Download voice &models..."),
			_("Choose a voice model on AivisHub, download it and add its voices."),
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_model_download, item)
		item = menu.Append(
			wx.ID_ANY, _("R&emove voice models..."), _("Removes voice models that you downloaded (or placed yourself).")
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_model_remove, item)
		menu.AppendSeparator()
		# AivisSpeech の辞書
		item = menu.Append(
			wx.ID_ANY,
			_("Install AivisSpeech &dictionaries..."),
			_("Downloads the AivisSpeech Engine dictionaries (readings of proper nouns and more) from GitHub."),
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_dict_install, item)
		item = menu.Append(
			wx.ID_ANY, _("Remo&ve AivisSpeech dictionaries..."), _("Removes the downloaded AivisSpeech dictionaries.")
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_dict_remove, item)
		menu.AppendSeparator()
		# CUDA 加速
		item = menu.Append(
			wx.ID_ANY,
			_("Install CUDA &acceleration..."),
			_("Downloads the libraries for speech synthesis on an NVIDIA GPU. They take effect after NVDA is restarted."),
		)
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_install, item)
		item = menu.Append(wx.ID_ANY, _("Remove CUDA acce&leration..."), _("Removes the downloaded CUDA acceleration libraries."))
		gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, self.on_remove, item)
		self._menu_item = gui.mainFrame.sysTrayIcon.menu.Insert(2, wx.ID_ANY, _("DASH"), menu)

	# ---- アップデート (VSU の globalPlugins/VSU/__init__.py と同じ) ----
	def updateCheckToggleString(self):
		if self.getUpdateCheckSetting() is True:
			return _("Disable update checking on startup")
		return _("Enable update checking on startup")

	def toggleUpdateCheck(self, evt):
		changed = not self.getUpdateCheckSetting()
		self.setUpdateCheckSetting(changed)
		if changed is True:
			msg = _("Updates will be checked when NVDA starts.")
		else:
			msg = _("Updates will not be checked when NVDA starts.")
		self.updateCheckToggleItem.SetItemLabel(self.updateCheckToggleString())
		compat.messageBox(msg, _("Settings changed"))

	def performUpdateCheck(self, evt):
		updater.AutoUpdateChecker().autoUpdateCheck(mode=updater.MANUAL)

	def getUpdateCheckSetting(self):
		return config.conf["DASH_global"]["checkForUpdatesOnStartup"]

	def setUpdateCheckSetting(self, val):
		config.conf["DASH_global"]["checkForUpdatesOnStartup"] = val

	# ---- AivisSpeech の辞書 ----
	@staticmethod
	def _running_runtime():
		"""DASH が今のシンセサイザーなら、その実行部分 (_aivis.Runtime) を返す。"""
		import synthDriverHandler

		synth = synthDriverHandler.getSynth()
		if synth is not None and getattr(synth, "name", None) == "dash":
			return getattr(synth, "_rt", None)
		return None

	# ---- 音声合成モデル (AivisHub) ----
	def on_model_download(self, evt):
		gui.mainFrame.prePopup()
		ModelDownloadDialog(self).Show()

	def on_model_remove(self, evt):
		if self._busy:
			return
		from synthDrivers._aivis import model_hub, paths

		models = model_hub.downloaded_models(paths.user_data_dir() / "models")
		if not models:
			compat.messageBox(_("There are no voice models that can be removed (bundled models and AivisSpeech models cannot be removed)."),
				_("Remove voice model"))
			return
		choices = [f"{m.get('name', p.stem)} ({', '.join(m.get('creators') or [])})" for p, m in models]
		gui.mainFrame.prePopup()
		dlg = wx.SingleChoiceDialog(
			gui.mainFrame, _("Choose the voice model to remove."), _("Remove voice model"), choices
		)
		ok = dlg.ShowModal() == wx.ID_OK
		i = dlg.GetSelection()
		dlg.Destroy()
		gui.mainFrame.postPopup()
		if not ok:
			return
		path, manifest = models[i]
		if not compat.yesno(
			_("\"{name}\" will be removed.\nDo you want to continue?").format(name=manifest.get("name", path.stem)),
			_("Remove voice model"),
			defaultNo=True,
		):
			return
		try:
			path.unlink()
		except Exception as e:
			log.error("DASH: model remove failed", exc_info=True)
			compat.messageBox(_("Could not remove the voice model:\n{e}").format(e=e), _("Error"), error=True)
			return
		rt = self._running_runtime()
		if rt is not None:
			rt.refresh_models()
		compat.messageBox(_("\"{name}\" has been removed.").format(name=manifest.get("name", path.stem)), _("Remove voice model"))

	def on_dict_install(self, evt):
		if self._busy:
			return
		from synthDrivers._aivis import dict_installer as di
		from synthDrivers._aivis import paths

		man = di.load_manifest()
		base = paths.user_data_dir()
		msg = _(
			"The AivisSpeech Engine dictionaries ({n} files: readings and accents of proper nouns, "
			"technical terms and more) will be downloaded directly from the AivisSpeech Engine repository "
			"on GitHub and installed.\n\n"
			"Download: about {mb:.0f}MB\n"
			"These are the dictionaries included in the AivisSpeech Engine (LGPL-3.0) repository. "
			"The sources and licenses of the dictionaries follow AivisSpeech Engine.\n"
		).format(n=len(man["files"]), mb=man["download_bytes"] / 2**20)
		if di.installed_version(base):
			msg += _("\n(Already installed. It will be reinstalled.)\n")
		msg += _("Do you want to continue?")
		if not compat.yesno(msg, _("Install AivisSpeech dictionaries"), defaultNo=True):
			return
		self._busy = True
		dlg = ProgressDialog(_("Installing AivisSpeech dictionaries"), _("Connecting"))
		threading.Thread(target=self._do_dict_install, args=(di, base, dlg), name="DASH.DictInstall", daemon=True).start()

	def _do_dict_install(self, di, base, dlg):
		last = [0.0]

		def progress(done, total, name):
			now = time.perf_counter()
			if now - last[0] < 0.5 and done < total:
				return
			last[0] = now
			text = _("Downloading {name}... {done:.0f} / {total:.0f} MB").format(
				name=name, done=done / 2**20, total=total / 2**20
			)
			dlg.update_later(int(done * 100 / max(1, total)), text)

		try:
			di.install(base, progress, lambda: dlg.cancelled)
			# 使用中の辞書が無ければすぐに切り替え、DASH が動いていればその場で辞書を適用し直す
			activated = di.activate_pending(base)
		except di.Cancelled:
			log.info("DASH: dictionary install cancelled")
			dlg.close_and_wait()
			return
		except Exception as e:
			log.error("DASH: dictionary install failed", exc_info=True)
			dlg.close_and_wait()
			compat.messageBox(_("An error occurred while installing:\n{e}").format(e=e), _("Error"), error=True)
			return
		finally:
			self._busy = False
		dlg.close_and_wait()
		rt = self._running_runtime() if activated else None
		if rt is not None:
			rt.reload_dictionaries()
			compat.messageBox(_("The AivisSpeech dictionaries have been installed. They will be used from the next utterance."), _("Installation complete"))
		else:
			compat.messageBox(
				_("The AivisSpeech dictionaries have been installed. They will be used after NVDA is restarted."), _("Installation complete")
			)

	def on_dict_remove(self, evt):
		if self._busy:
			return
		from synthDrivers._aivis import dict_installer as di
		from synthDrivers._aivis import paths

		base = paths.user_data_dir()
		if not di.installed_version(base):
			compat.messageBox(_("The AivisSpeech dictionaries are not installed."), _("AivisSpeech dictionaries"))
			return
		if not compat.yesno(
			_("The downloaded AivisSpeech dictionaries will be removed.\nDo you want to continue?"),
			_("Remove AivisSpeech dictionaries"),
			defaultNo=True,
		):
			return
		removed = False
		try:
			# 削除を予約してすぐに反映する (pyopenjtalk は辞書のファイルを開いたままにしないので、使用中でも消せる)。
			# 消せなかった場合は予約が残り、NVDA の次回起動時に削除される
			di.request_remove(base)
			di.activate_pending(base)
			removed = not di.installed_dir(base).exists()
		except Exception:
			log.debugWarning("DASH: dictionary remove", exc_info=True)
		rt = self._running_runtime()
		if removed and rt is not None:
			rt.reload_dictionaries()
		if removed:
			compat.messageBox(_("The AivisSpeech dictionaries have been removed."), _("Remove AivisSpeech dictionaries"))
		else:
			compat.messageBox(_("It will be removed after NVDA is restarted."), _("Remove AivisSpeech dictionaries"))

	# ---- CUDA 加速 ----
	@staticmethod
	def _modules():
		from synthDrivers._aivis import cuda_installer, paths

		return cuda_installer, paths

	def on_install(self, evt):
		if self._busy:
			return
		ci, paths = self._modules()
		if not ci.has_nvidia_driver():
			compat.messageBox(
				_("The NVIDIA GPU driver was not found. CUDA acceleration can only be used with an NVIDIA GPU."),
				_("CUDA acceleration"),
				error=True,
			)
			return
		man = ci.load_manifest()
		base = paths.user_data_dir()
		ver = ci.installed_version(base)
		msg = _(
			"The CUDA acceleration libraries (onnxruntime-gpu 1.20.1, CUDA 11.8, cuDNN 8.9) "
			"for speech synthesis on an NVIDIA GPU will be downloaded and installed.\n\n"
			"Download: about {dl:.1f}GB / Disk space required: about {inst:.1f}GB\n"
			"Sources: Microsoft (onnxruntime), NVIDIA (PyPI), zlib (winimage)\n"
			"The RTX 50 series is not supported.\n\n"
			"After installation, restart NVDA and \"Use GPU\" in the speech settings will be turned on.\n"
		).format(dl=man["download_bytes"] / 2**30, inst=man["install_bytes"] / 2**30)
		if ver:
			msg += _("\n(Already installed. It will be reinstalled.)\n")
		msg += _("Do you want to continue?")
		if not compat.yesno(msg, _("Install CUDA acceleration"), defaultNo=True):
			return
		self._busy = True
		dlg = ProgressDialog(_("Installing CUDA acceleration"), _("Connecting"))
		threading.Thread(target=self._do_install, args=(ci, base, dlg), name="DASH.CudaInstall", daemon=True).start()

	def _do_install(self, ci, base, dlg):
		last = [0.0]

		def progress(done, total, label):
			now = time.perf_counter()
			if now - last[0] < 0.5 and done < total:
				return
			last[0] = now
			pct = int(done * 100 / max(1, total))
			text = _("Downloading {label}... {done:.0f} / {total:.0f} MB").format(
				label=label, done=done / 2**20, total=total / 2**20
			)
			dlg.update_later(pct, text)

		try:
			ci.install(base, progress, lambda: dlg.cancelled)
		except ci.Cancelled:
			log.info("DASH: CUDA install cancelled")
			dlg.close_and_wait()
			return
		except Exception as e:
			log.error("DASH: CUDA install failed", exc_info=True)
			dlg.close_and_wait()
			compat.messageBox(_("An error occurred while installing:\n{e}").format(e=e), _("Error"), error=True)
			return
		finally:
			self._busy = False
		# 再起動後に GPU を使うように設定しておく
		try:
			config.conf["speech"]["dash"]["useGpu"] = True
		except Exception:
			log.debugWarning("DASH: could not set useGpu", exc_info=True)
		dlg.close_and_wait()
		if compat.yesno(
			_("CUDA acceleration has been installed. It will take effect after NVDA is restarted.\nRestart NVDA now?"),
			_("Installation complete"),
		):
			import core

			wx.CallAfter(core.restart)

	def on_remove(self, evt):
		if self._busy:
			return
		ci, paths = self._modules()
		base = paths.user_data_dir()
		if not ci.installed_version(base):
			compat.messageBox(_("CUDA acceleration is not installed."), _("CUDA acceleration"))
			return
		if not compat.yesno(
			_("The CUDA acceleration libraries will be removed. Files in use will be removed after NVDA is restarted.\nDo you want to continue?"),
			_("Remove CUDA acceleration"),
			defaultNo=True,
		):
			return
		try:
			ci.request_remove(base)
			config.conf["speech"]["dash"]["useGpu"] = False
		except Exception:
			log.debugWarning("DASH: remove request", exc_info=True)
		compat.messageBox(_("It will be removed after NVDA is restarted."), _("Remove CUDA acceleration"))
