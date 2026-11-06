"""Small themed widgets shared by the windows (same conventions as another app / the design)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

from .theme import icon, theme


def label(text: str = "", *, level: str | None = None, type_: str | None = None, role: str | None = None,
          severity: str | None = None, parent=None) -> QLabel:
    lab = QLabel(text, parent)
    for k, v in (("level", level), ("type", type_), ("role", role), ("severity", severity)):
        if v:
            lab.setProperty(k, v)
    return lab


def badge(text: str, severity: str | None = None) -> QLabel:
    b = label(text, role="badge", severity=severity)
    b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return b


def icon_button(name: str, tip: str, parent=None, checkable: bool = False) -> QToolButton:
    b = QToolButton(parent)
    b.setProperty("role", "iconButton")
    b.setIcon(icon(name))
    b.setIconSize(QSize(16, 16))
    b.setToolTip(tip)
    b.setAutoRaise(True)
    b.setCheckable(checkable)
    return b


def section_bar(title: str, parent=None) -> tuple[QWidget, QHBoxLayout]:
    w = QWidget(parent)
    w.setProperty("role", "sectionBar")
    w.setAttribute(Qt.WA_StyledBackground)
    w.setFixedHeight(theme().sectionHeader)
    h = QHBoxLayout(w)
    h.setContentsMargins(8, 0, 4, 0)
    h.setSpacing(2)
    h.addWidget(label(title, role="sectionTitle"))
    h.addStretch(1)
    return w, h


class FrameTile(QLabel):
    """Fixed-size image tile; FrameService puts frames into it. Click emits the tile's time."""

    clicked = Signal(float)

    def __init__(self, w: int, h: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("frameTile")
        self.key = ""
        self.t = 0.0
        self._img = QImage()
        self.clear_frame("")

    def clear_frame(self, text: str = "…") -> None:
        self._img = QImage()
        self.setPixmap(QPixmap())
        self.setText(text)

    def has_image(self) -> bool:
        return not self._img.isNull()

    def set_image(self, img: QImage) -> None:
        if img.isNull():
            self.clear_frame("no frame")
            return
        self._img = img
        dpr = self.devicePixelRatioF()
        pm = QPixmap.fromImage(img).scaled(self.size() * dpr, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.setText("")
        self.setPixmap(pm)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.t)
        super().mousePressEvent(e)
