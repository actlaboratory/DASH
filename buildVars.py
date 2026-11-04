# -*- coding: UTF-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory

# Build customizations (VSU と同じ形。tools/build.py・tools/bumpup.py・sconstruct が読む)
# manifest.ini は manifest.ini.tpl とこのファイルからビルド時に作る


def _(arg):
    return arg


# tools/bumpup.py が下の 2 行を書き換える (スナップショットのビルドでも一時的に書き換える)
ADDON_VERSION = "0.1.0"
ADDON_RELEASE_DATE = "2026-11-04"
ADDON_NAME = "Direct Aivis Streaming Handler"
ADDON_KEYWORD = "DASH"

# Add-on information variables
addon_info = {
    # add-on Name/identifier, internal for NVDA
    "addon_name": ADDON_NAME,
    # actlab custom: keyword (アップデートサーバーにはこの略称を送る)
    "addon_keyword": ADDON_KEYWORD,
    "addon_summary": _("DASH, Direct Aivis Streaming Handler"),
    # Translators: the add-on description shown in the NVDA add-on store / manager
    "addon_description": _("""A speech synthesizer for NVDA that speaks with AivisSpeech (Style-Bert-VITS2 / AIVMX) voices.
Synthesis is streamed, so speech starts without waiting for the whole sentence to be synthesized."""),
    "addon_version": ADDON_VERSION,
    "addon_author": "yamahubuki, ACT Laboratory",
    "addon_url": "https://actlab.org/software/DASH",
    "addon_docFileName": "readme.html",
    "addon_minimumNVDAVersion": "2026.1",
    "addon_lastTestedNVDAVersion": "2026.1",
    "addon_updateChannel": None,
}
