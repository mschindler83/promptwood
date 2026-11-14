"""A character's voice made with text to speech: designed from a description (Qwen3-TTS, local) or one of the
account's ElevenLabs voices. A sample sentence is made to listen to; "Use this voice" makes it the character's voice."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from .. import tts
from .widgets import label


class _Signals(QObject):
    sample = Signal(object, str)               # secs or None, error
    voices = Signal(object, str)               # [voice] or None, error


class VoiceTtsDialog(QDialog):
    """play(path, title): how a sample is listened to; local(): the app's Qwen3-TTS worker; eleven(): an
    ElevenLabs client (with the kept key)."""

    def __init__(self, parent, name: str, desc: str, lang: str, provider: str, model: str, play, local, eleven,
                 voices_cache: list | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Voice of @{name} with text to speech")
        self.name, self.lang, self.model = name, lang, model
        self._play, self._local, self._eleven = play, local, eleven
        self._sig = _Signals(self)
        self._sig.sample.connect(self._sample_done)
        self._sig.voices.connect(self._voices_done)
        self._tmp = tempfile.TemporaryDirectory(prefix="video-chain-voice-")
        self.wav = Path(self._tmp.name) / "sample.wav"
        self.made: dict | None = None            # what the sample was made with (use_sample's label / text / eleven)
        self._pending: dict | None = None
        self.voices: list = []                   # the account's ElevenLabs voices (once loaded)
        self.busy = False
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem("Design a new voice from a description (Qwen3-TTS, local)", "local")
        self.kind.addItem("One of my ElevenLabs voices", "elevenlabs")
        self.kind.setCurrentIndex(max(0, self.kind.findData(provider)))
        form.addRow("Voice", self.kind)
        v.addLayout(form)
        self.pages = QStackedWidget()
        lp = QWidget()
        lf = QFormLayout(lp)
        lf.setContentsMargins(0, 0, 0, 0)
        self.desc = QPlainTextEdit()
        self.desc.setPlaceholderText("Who speaks: age, gender, how the voice sounds, e.g. a young woman, about "
                                     "twenty, warm and clear voice, calm and friendly, medium pitch")
        self.desc.setPlainText(desc)
        self.desc.setFixedHeight(72)
        lf.addRow("Description", self.desc)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(1)
        self.seed.setToolTip("Another seed: another voice for the same description")
        lf.addRow("Seed", self.seed)
        self.pages.addWidget(lp)
        ep = QWidget()
        ef = QFormLayout(ep)
        ef.setContentsMargins(0, 0, 0, 0)
        erow = QHBoxLayout()
        self.voice = QComboBox()
        self.voice.setMinimumWidth(260)
        self.b_voices = QPushButton("Load my voices")
        self.b_voices.clicked.connect(self._load_voices)
        erow.addWidget(self.voice, 1)
        erow.addWidget(self.b_voices)
        ef.addRow("ElevenLabs voice", erow)
        self.pages.addWidget(ep)
        v.addWidget(self.pages)
        tf = QFormLayout()
        self.text = QLineEdit(tts.SAMPLE_TEXT.get(lang, tts.SAMPLE_TEXT["English"]))
        self.text.setToolTip(f"What the sample says ({lang}, the project's speech language); 4-7 s are best")
        tf.addRow("Sample text", self.text)
        v.addLayout(tf)
        brow = QHBoxLayout()
        self.b_make = QPushButton("Make a sample")
        self.b_make.setDefault(True)
        self.b_make.clicked.connect(self._make)
        self.b_play = QPushButton("Listen")
        self.b_play.setEnabled(False)
        self.b_play.clicked.connect(lambda: self._play(self.wav, f"Voice sample for @{self.name}"))
        brow.addWidget(self.b_make)
        brow.addWidget(self.b_play)
        brow.addStretch(1)
        v.addLayout(brow)
        self.info = label("", level="tertiary")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.b_use = self.buttons.addButton("Use this voice", QDialogButtonBox.AcceptRole)
        self.b_use.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        if voices_cache:
            self._fill_voices(voices_cache)
        self._kind_changed()
        self.resize(560, self.sizeHint().height())

    def provider(self) -> str:
        return self.kind.currentData()

    def _kind_changed(self, *_):
        self.pages.setCurrentIndex(0 if self.provider() == "local" else 1)
        self.info.setText("Made on this computer with Qwen3-TTS (about 5 GB of GPU memory while it runs; ComfyUI's "
                          "models are unloaded first when there is not enough)." if self.provider() == "local" else
                          "Made by ElevenLabs with your API key; the character's lines are then spoken in this "
                          "voice (and the clip model hears the sample).")
        self._enable()

    def _enable(self):
        ok = not self.busy and (self.provider() == "local" or self.voice.currentData() is not None)
        self.b_make.setEnabled(ok and bool(self.text.text().strip()))
        self.b_voices.setEnabled(not self.busy)
        self.kind.setEnabled(not self.busy)
        self.buttons.button(QDialogButtonBox.Cancel).setEnabled(not self.busy)   # (a thread is working for it)

    def _emit_safe(self, signal, *args) -> None:
        """Emit from a worker thread; the dialog (and its signals) may be gone by then."""
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    # ---- ElevenLabs voices
    def _load_voices(self):
        self.busy = True
        self._enable()
        self.info.setText("Asking ElevenLabs for your voices…")

        def work():
            try:
                voices = self._eleven().voices()
            except Exception as e:  # noqa: BLE001
                self._emit_safe(self._sig.voices, None, str(e))
                return
            self._emit_safe(self._sig.voices, voices, "")
        threading.Thread(target=work, daemon=True).start()

    def _voices_done(self, voices, err: str):
        self.busy = False
        if voices is None:
            self.info.setText(f"Could not load the voices: {err}")
        else:
            self._fill_voices(voices)
            self.info.setText(f"{len(voices)} voice{'s' if len(voices) != 1 else ''} in your account.")
        self._enable()

    def _fill_voices(self, voices: list):
        self.voices = voices
        self.voice.clear()
        for vc in voices:
            lab = vc.get("labels") or {}
            extra = ", ".join(x for x in (lab.get("gender"), lab.get("age"), lab.get("accent")) if x)
            self.voice.addItem(vc["name"] + (f" ({extra})" if extra else ""), vc)
        self._enable()

    # ---- the sample
    def _make(self):
        text, kind = self.text.text().strip(), self.provider()
        desc, seed = self.desc.toPlainText().strip(), self.seed.value()
        voice = self.voice.currentData()
        if kind == "local" and not desc:
            self.info.setText("Describe the voice first.")
            return
        self.busy = True
        self.b_use.setEnabled(False)
        self._enable()
        self.info.setText("Making the sample…" + (" (the model loads first: about 15 s)" if kind == "local" else ""))
        voice_id = voice["id"] if kind == "elevenlabs" and voice else None      # (the local engine has no voice ids)
        made = ({"label": f"designed: {desc[:40]}", "text": text, "eleven": None} if kind == "local" else
                {"label": f"ElevenLabs: {voice['name']}", "text": text,
                 "eleven": {"id": voice["id"], "name": voice["name"]}})

        def work():
            try:
                eng = self._local() if kind == "local" else self._eleven()
                secs = tts.make_sample(eng, text, self.lang, self.wav, description=desc, voice_id=voice_id,
                                       model=self.model, seed=seed)
                self._pending = made
            except Exception as e:  # noqa: BLE001
                self._emit_safe(self._sig.sample, None, str(e))
                return
            self._emit_safe(self._sig.sample, secs, "")
        threading.Thread(target=work, daemon=True).start()

    def _sample_done(self, secs, err: str):
        self.busy = False
        self._enable()
        if secs is None:
            self.info.setText(f"Could not make the sample: {err}")
            return
        self.made = self._pending
        self.b_play.setEnabled(True)
        self.b_use.setEnabled(True)
        self.info.setText(f"Sample made ({secs:.1f} s). Listen to it; make another (other seed / voice) or use it.")
        self._play(self.wav, f"Voice sample for @{self.name}")

    def reject(self):
        if self.busy:                            # (Cancel is off while busy; Esc still comes here)
            self.info.setText("Wait until the voice is made, then close.")
            return
        super().reject()

    def accept(self):
        if not self.busy and self.made is not None:
            super().accept()

    def cleanup(self):
        self._tmp.cleanup()


def run_dialog(dlg: VoiceTtsDialog) -> bool:
    return dlg.exec() == QDialog.Accepted

