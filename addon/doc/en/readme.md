# DASH: Direct Aivis Streaming Handler

## Overview

DASH (Direct Aivis Streaming Handler) is an add-on that lets you use AivisSpeech voice models (AIVMX) as a speech synthesizer for NVDA.
It does not use AivisSpeech Engine (the HTTP server); instead, only the core of the speech synthesis runs directly inside NVDA.
The BERT model and the voice model "Mao" needed to run are bundled, so you do not need to install AivisSpeech separately.

DASH speaks Japanese. English words are read in Japanese-style pronunciation (see "Reading English words" below).

- **Streaming playback**: the voice model is split into "text → latent representation" and "latent representation → waveform",
  and the waveform is generated a little at a time and played as soon as each part is ready. Because it does not wait for
  the whole sentence to be synthesized, short utterances start speaking in about 0.1 to 0.2 seconds
  (measured on a CPU, Core i7-13700F).
- Long text is split into sentences and clauses, and synthesized in order from the first sentence.
- It uses the same text processing as AivisSpeech Engine (normalization, readings, accents and BERT).
- It supports context-dependent readings (tsqyomi): the same kanji is read in the way that fits the context,
  for example 人気 (ひとけ / にんき), 上手 (かみて / じょうず) and 市場 (いちば / しじょう).

### About AivisSpeech

AivisSpeech is free Japanese speech synthesis software developed by Aivis Project.
Its voice models are based on Style-Bert-VITS2 and can speak expressively and naturally.
Many voice models are published on AivisHub, and you can add the voices you like.
DASH makes the AivisSpeech voice models and text processing usable directly inside NVDA.

- AivisSpeech official site: https://aivis-project.com/
- AivisHub (site where voice models are published): https://hub.aivis-project.com/

## System requirements

- NVDA 2026.1 or later (64-bit)
- Windows 10 / Windows 11 (64-bit)
- Memory: at least 4GB free (the BERT model, the voice model and the reading model use about 2.3GB)
- Disk: about 2.2GB (about 2GB more if you install CUDA acceleration)
- CPU: 4 or more cores recommended. DASH works with the CPU only
- GPU (optional): a DirectX 12 GPU (DirectML), or an NVIDIA GPU (CUDA acceleration)

## Installation

Open the nvda-addon file with NVDA to install it, then restart NVDA.
In the "Speech" category of NVDA's settings, change the synthesizer to "DASH (AivisSpeech)".

When NVDA starts, loading the models takes about 10 seconds. Speech is slightly delayed until loading finishes.

## Voices

By default, DASH uses the voice model "Mao" (Aivis Project) bundled with the add-on.
You can choose a voice (a combination of speaker and style) with "Voice" in NVDA's speech settings.

### Adding voice models

You can add voice models published on AivisHub (https://hub.aivis-project.com/) from the NVDA menu.

1. Choose NVDA menu → "DASH" → "Download voice models...".
2. Search with "Sort by" (most popular, newest, most liked) and "Keyword", and choose a model from the "Models" list.
   "Details and license" shows the author, the type of voice, the speakers and styles, the description,
   and the license type and full text.
   - Preview: press Space in the "Models" list to play the preview audio of the model (its first style).
     Press it again to stop. Moving to another model also stops it.
   - In the "Styles" list, you can play the preview audio for each speaker and style with Space or Enter.
     The "Preview" button also plays it.
   - The preview audio is published on AivisHub, and is played through NVDA's output device.
3. Press "Download" and confirm that you agree to the license to download the model (about 240MB each).
   The file is saved after its content has been verified (SHA-256).
4. When it finishes, you can choose the new voices with "Voice" in NVDA's speech settings right away (no restart needed).

- Location: `aivis\models` in the NVDA configuration folder (usually `%APPDATA%\nvda`)
- Removal: NVDA menu → "DASH" → "Remove voice models..." (models in `aivis\models` can be removed)
- The first utterance right after switching voices takes about 2 to 3 seconds to load the model
  (only one model is loaded at a time to save memory).

You can also add models in these ways:

- Install and use AivisSpeech: models added to AivisSpeech (`%APPDATA%\AivisSpeech-Engine\Models`)
  are used read-only. The AivisSpeech user dictionary is also loaded.
- Place them yourself: put `.aivmx` files in the `aivis\models` folder (they can be used after NVDA is restarted).

### Terms of use of voice models

Each voice model has its own terms of use. Always follow the terms of each model.

The bundled "Mao" is licensed under the Aivis Common Model License (ACML) 1.0
(full text: `*_LICENSE.md` in `synthDrivers\_aivis\data\models\` in the add-on folder).
It can be used for both commercial and non-commercial purposes, but the following uses are prohibited:

- Uses that mislead people into thinking that it is the speaker or an official party (such as deepfakes)
- Uses that damage the image or dignity of the speaker, or criticism, attacks, harassment or discrimination
  against real people or organizations
- Publishing false information intended to deceive people, or exaggerated or false marketing
- Activities that call for support for or opposition to particular political, religious or ideological positions
  or conspiracy theories
- Antisocial or criminal uses

The terms of models you add differ from model to model (some models are for non-commercial use only).
Check them on the model's page on AivisHub, or in "Model management" in AivisSpeech.

## Synthesis parameters

You can adjust the following parameters in NVDA's speech settings.

| Parameter | Description | Behavior |
|---|---|---|
| Voice | Combination of speaker and style | The first voice found ("まお (ノーマル)" if only the bundled model is present) |
| Rate | Speaking rate | 50 is normal speed, 75 is about 2.2 times, 100 is 5 times. 0 is 0.5 times |
| Pitch | Pitch of the voice | 50 means no change. 0 is half an octave lower, 100 is half an octave higher |
| Inflection | Strength of emotional expression ("Emotional intensity" in AivisSpeech) | 50 is standard |
| Volume | Volume | 100 is standard |
| Use GPU (DirectML/CUDA) | Run inference on the GPU | Off (CPU) |

- Rate: speech is synthesized without changing its speed, and then sped up with Sonic, which comes with NVDA
  (to keep it easy to understand at high speed). When running on the CPU, only the part above 4 times (rate 94 and above)
  also raises the speaking speed in AIVIS a little so that synthesis keeps up (5 times = AIVIS 1.25 times × Sonic 4 times).
  With CUDA acceleration, the speaking speed in AIVIS is not changed up to 5 times.
- Pitch: changed with Sonic, so the speed does not change. The pitch change when reading capital letters
  (NVDA's "Capital pitch change percentage") is also supported.
- Use GPU: uses CUDA if CUDA acceleration (below) is installed, and DirectML otherwise.
  DirectML can actually be slower for short utterances, so this is off by default.

## NVDA menu items

When DASH is installed, a "DASH" item is added to the NVDA menu.

- **Disable / Enable update checking on startup**: turns the automatic update check when NVDA starts on or off
- **Check for updates**: checks for updates manually
- **Download voice models...**: chooses a voice model on AivisHub, downloads it and adds its voices
  (see "Adding voice models" above)
- **Remove voice models...**: removes voice models that you downloaded (or placed yourself)
- **Install AivisSpeech dictionaries...**: downloads and installs the AivisSpeech Engine dictionaries from GitHub
  (see "Dictionaries" below)
- **Remove AivisSpeech dictionaries...**: removes the downloaded AivisSpeech dictionaries
- **Install CUDA acceleration...**: downloads and installs the CUDA acceleration libraries for NVIDIA GPUs
- **Remove CUDA acceleration...**: removes the downloaded CUDA acceleration libraries (they are removed after NVDA restarts)

## CUDA acceleration

If you have an NVIDIA GPU, DASH can synthesize speech with CUDA. The CPU load drops to a fraction,
and continuous reading, fast speech and the start of speech when you press keys in quick succession become faster.

### Installing CUDA acceleration

1. Choose NVDA menu → "DASH" → "Install CUDA acceleration...".
2. Choose "Yes" in the confirmation dialog (download size: about 1.1GB).
3. The following libraries are downloaded directly from their distributors and installed.
   Only the files needed are extracted, and each is saved after its content has been verified (SHA-256).
   - onnxruntime-gpu 1.20.1 (Microsoft)
   - CUDA 11.8 runtime, cuBLAS, cuFFT, cuDNN 8.9 (NVIDIA, PyPI)
   - zlibwapi.dll (a dependency of cuDNN 8, winimage)
4. When it finishes, restart NVDA. "Use GPU" is turned on automatically.

### Notes

- CUDA acceleration requires an NVIDIA GPU (such as the GeForce RTX series) and the NVIDIA driver.
- Installation requires a stable internet connection and about 2GB of free disk space.
- It is saved in `aivis\cuda` in the NVDA configuration folder. You do not need to reinstall it when you update the add-on.
- DASH uses an older onnxruntime and cuDNN 8, because the newer version (cuDNN 9) is slow with this model.
  For this reason, the RTX 50 series is not supported.
- The first utterance after a period of silence may be slightly slower than on the CPU,
  because the GPU has to wake up from its power-saving state.

## Dictionaries

- Readings and accents use the dictionary built into pyopenjtalk-plus.
- If you have a user dictionary registered in AivisSpeech (`%APPDATA%\AivisSpeech-Engine\user.dic`), it is loaded automatically.
- Dictionaries in OpenJTalk format (.dic) placed in `aivis\dictionaries\` in the NVDA configuration folder are also loaded.

### AivisSpeech dictionaries

Adding the dictionaries that come with AivisSpeech Engine (38 files: readings and accents of place names,
personal names, station names, technical terms and more) makes DASH read text the same way as AivisSpeech.

1. Choose NVDA menu → "DASH" → "Install AivisSpeech dictionaries...".
2. Choose "Yes" in the confirmation dialog to download them directly from the AivisSpeech Engine repository on GitHub
   (about 225MB). Each file is saved after its content has been verified (SHA-256).
3. When it finishes, the dictionaries are used from the next utterance (no need to restart NVDA).

- Location: `aivis\aivisspeech_dictionaries` in the NVDA configuration folder
  (separate from `aivis\dictionaries`, where you place your own dictionaries)
- Removal: NVDA menu → "DASH" → "Remove AivisSpeech dictionaries..."
- These dictionaries are included in the AivisSpeech Engine (LGPL-3.0) repository. Because the source and license
  of each dictionary cannot be confirmed, they are not bundled with the add-on; users download them instead.
  The sources and licenses of the dictionaries follow AivisSpeech Engine.
- Installing the dictionaries hardly changes the time until speech starts (about 2 milliseconds per utterance).

## Reading English words

As in AivisSpeech Engine, English words are automatically converted to katakana readings before they are spoken (e2k).
Words that are not in the dictionary get an estimated reading, which may sound unnatural.

## Memory

To make inference on the CPU fast, the fp32 version of the BERT model (about 1.3GB) is bundled
(about 3 times faster on the CPU than the fp16 version).
Only the selected voice model is loaded. The reading model (tsqyomi) uses about 0.3GB,
and DASH uses about 2.3GB of memory in total.

## Known limitations

- On the CPU at 3.5 times or faster, a slight gap (10 to 90 milliseconds) may appear between items when very short items
  such as "File" and "Edit" are read one after another.
- CUDA acceleration on the RTX 50 series is not supported.

## Version history

- Ver 0.1.0 (2026-11)
	- Initial version

## Contact

Software page: https://actlab.org/software/DASH

Email: support@actlab.org

## Copyright

This add-on can be used under the terms of the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
(`LICENSE.txt` in the add-on folder).
Copyright (c) 2026 yamahubuki, ACT Laboratory

Part of the Japanese text processing (normalization, readings and accents) uses code from Style-Bert-VITS2,
and that part is used under the GNU Affero General Public License v3.0 (the origin is stated at the top of each file).
Copyright (c) 2024-2026 tsukumijima, litagin02 (Style-Bert-VITS2)
Copyright (c) Bert-VITS2 contributors (fishaudio/Bert-VITS2)

Part of the mora table comes from VOICEVOX ENGINE (GNU Lesser General Public License v3.0).

The BERT model (tsukumijima/deberta-v2-large-japanese-char-wwm-onnx) is bundled unmodified under
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
Original model: ku-nlp/deberta-v2-large-japanese-char-wwm (Kurohashi-Chu-Murawaki Lab, Kyoto University)

The voice model "Mao" (Aivis Project) is bundled under the Aivis Common Model License (ACML) 1.0.

DirectML.dll is bundled under the Microsoft Software License Terms (Microsoft DirectX Machine Learning).

ONNX Runtime is bundled under the MIT License. Copyright (c) Microsoft Corporation

The reading model tsqyomi (tsukumijima/tsqyomi-models) is bundled under the MIT License.
Copyright (c) 2026 Aivis Project

Hugging Face Tokenizers is bundled under the Apache License 2.0.

pyopenjtalk-plus is bundled under the MIT License.
Copyright (c) 2018 Ryuichi Yamamoto

The following software and data included in pyopenjtalk-plus are bundled under their respective licenses:

- Open JTalk: Modified BSD License
  Copyright (c) 2008-2016 Nagoya Institute of Technology, Department of Computer Science
- hts_engine API: Modified BSD License
  Copyright (c) 2001-2014 Nagoya Institute of Technology, Department of Computer Science;
  2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering
- MeCab: BSD License
  Copyright (c) 2001-2008 Taku Kudo; Copyright (c) 2004-2008 Nippon Telegraph and Telephone Corporation
- NAIST Japanese Dictionary (reading dictionary): BSD License
  Copyright (c) 2009 Nara Institute of Science and Technology, Japan
- HTS Voice "Mei" (not used by this add-on): Creative Commons Attribution 3.0
  Copyright (c) 2009-2013 Nagoya Institute of Technology, Department of Computer Science

The full texts of these licenses and the list of licenses of the other bundled libraries (NumPy, ONNX, SudachiPy,
pydantic, e2k and others) are in [third_party.html](../ja/third_party.html) (in Japanese; the license texts themselves
are in their original language).
The license documents of each library are also bundled in the add-on folder (`synthDrivers\_aivis\deps\`).
