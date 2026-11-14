"""Graphite / Cobalt theme, shared with the other pipeline apps (from the design).

The tokens (tokens.json), the stylesheet template (theme.qss.in), the fonts and the icons are copies of the design's, so the
designer looks like the main app. ${...} placeholders are filled exactly like the C++ fillTemplate().
"""
from __future__ import annotations

import json
import re
import warnings
import zlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QByteArray, QRectF, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

THEME_DIR = Path(__file__).resolve().parent.parent / "theme"


def _colour(v: str) -> QColor:
    v = v.strip()
    m = re.fullmatch(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)", v)
    if m:
        return QColor(int(m[1]), int(m[2]), int(m[3]), round(float(m[4]) * 255))
    return QColor(v)


def _to_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0


def with_alpha(c: QColor, a: float) -> QColor:
    c = QColor(c)
    c.setAlphaF(a)
    return c


class Theme(SimpleNamespace):
    """Token values: colours as QColor attributes (t.bgDock, t.accent ...), metrics as ints, font sizes in t.font."""

    def accent_muted(self):
        return with_alpha(self.accent, self.accentMutedAlpha)

    def accent_line(self):
        return with_alpha(self.accent, self.accentLineAlpha)

    def accent_scrub(self):
        return with_alpha(self.accent, self.accentScrubAlpha)

    def hover(self):
        return with_alpha(QColor(255, 255, 255), self.hoverAlpha)

    def sel_inactive(self):
        return with_alpha(QColor(255, 255, 255), self.selInactiveAlpha)

    def row_height(self) -> int:
        return self.rows[self.density]

    def colour(self, name: str) -> QColor | None:
        derived = {"accentMuted": self.accent_muted, "accentLine": self.accent_line, "hover": self.hover,
                   "selInactive": self.sel_inactive}
        if name in derived:
            return derived[name]()
        if name == "white":
            return QColor(255, 255, 255)
        if name == "black":
            return QColor(0, 0, 0)
        v = getattr(self, name, None)
        return QColor(v) if isinstance(v, QColor) else None


def load_tokens(accent: str | None = None) -> Theme:
    tok = json.loads((THEME_DIR / "tokens.json").read_text())
    t = Theme()
    for group in ("surface", "line", "text", "axis"):
        for k, v in tok[group].items():
            setattr(t, k, _colour(v))
    for group in ("semantic", "component", "state"):
        for k, v in tok[group].items():
            setattr(t, k, _colour(v) if isinstance(v, str) else v)
    fams = tok["accents"]["families"]
    default = next(f for f in fams if f["name"] == tok["accents"]["default"])
    fam = next((f for f in fams if f["name"] == (accent or tok["accents"]["default"])), default)
    t.accent, t.accentHover = QColor(fam["accent"]), QColor(fam["hover"])
    t.accentPressed, t.onAccent = QColor(fam["pressed"]), QColor(fam["onAccent"])
    t.accent_families = [f["name"] for f in fams]
    t.space = SimpleNamespace(**tok["space"])
    t.radius = SimpleNamespace(**tok["radius"])
    t.rows = tok["density"]["rows"]
    t.density = tok["density"]["default"]
    for k, v in tok["metrics"].items():
        setattr(t, k, v)
    t.labelWidth = tok["metrics"]["labelWidthMin"]
    ty = tok["type"]
    t.font = SimpleNamespace(**{k: ty[k]["px"] for k in ("micro", "small", "body", "title", "heading", "display", "mono")})
    t.uiFamily, t.monoFamily = ty["uiFamily"], ty["monoFamily"]
    return t


# ---------------------------------------------------------------- icons

def _stroke_for(px: int) -> float:
    table = ((12, 1.5), (16, 1.25), (20, 1.2), (24, 1.1), (32, 0.9))
    if px <= table[0][0]:
        return table[0][1]
    for (p0, w0), (p1, w1) in zip(table, table[1:]):
        if px <= p1:
            return w0 + (w1 - w0) * (px - p0) / (p1 - p0)
    return table[-1][1]


@lru_cache(maxsize=None)
def _svg_text(name: str) -> str | None:
    """The icon's SVG source, read once; None (with a warning, once) for a name that has no icon."""
    path = THEME_DIR / "icons" / f"{name}.svg"
    if not path.exists():
        warnings.warn(f"no icon named {name!r} in the theme", stacklevel=3)
        return None
    return path.read_text()


def tinted_svg(name: str, tint: QColor, px: int = 16) -> bytes:
    s = _svg_text(name)
    if s is None:
        return b""
    c = tint.name().upper()
    s = re.sub("#000000", c, s, flags=re.IGNORECASE).replace("currentColor", c)
    # authored at 16 px with stroke 1.25; at other sizes the stroke is specified in output px (the design IconEngine)
    w = _stroke_for(px) * 16.0 / px
    s = re.sub(r'stroke-width="[0-9.]+"', f'stroke-width="{w:.3f}"', s)
    if tint.alpha() < 255:
        s = s.replace("<svg ", f'<svg opacity="{tint.alphaF():.3f}" ', 1)
    return s.encode()


def icon_pixmap(name: str, px: int, tint: QColor, dpr: float = 1.0) -> QPixmap:
    """The icon in this tint and size (cached: buttons ask again on every repaint)."""
    return _icon_pixmap(name, int(px), tint.rgba(), float(dpr))


@lru_cache(maxsize=512)
def _icon_pixmap(name: str, px: int, rgba: int, dpr: float) -> QPixmap:
    tint = QColor.fromRgba(rgba)
    r = QSvgRenderer(QByteArray(tinted_svg(name, tint, px)))
    pm = QPixmap(round(px * dpr), round(px * dpr))
    pm.fill(Qt.GlobalColor.transparent)
    if r.isValid():
        p = QPainter(pm)
        r.render(p, QRectF(0, 0, pm.width(), pm.height()))
        p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


class _TintedIconEngine(QIconEngine):
    """Renders the design icon in the theme tint for the requested mode / state (the design TintedIconEngine)."""

    def __init__(self, name: str, normal: QColor | None):
        super().__init__()
        self._name, self._normal = name, normal

    def _tint(self, mode, state) -> QColor:
        t = ThemeManager.instance().theme
        c = self._normal or t.textSecondary
        if mode == QIcon.Mode.Active:
            c = t.textPrimary
        if mode == QIcon.Mode.Selected or state == QIcon.State.On:
            c = t.accentHover
        if mode == QIcon.Mode.Disabled:
            c = t.textDisabled
        return c

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        px = max(1, min(size.width(), size.height()))
        return icon_pixmap(self._name, px, self._tint(mode, state), scale)

    def paint(self, painter, rect, mode, state):
        dpr = painter.device().devicePixelRatioF() if painter.device() else 1.0
        painter.drawPixmap(rect, self.scaledPixmap(rect.size(), mode, state, dpr))

    def clone(self):
        return _TintedIconEngine(self._name, self._normal)


def icon(name: str, tint: QColor | None = None) -> QIcon:
    return QIcon(_TintedIconEngine(name, tint))


# ---------------------------------------------------------------- manager

class ThemeManager:
    _inst: "ThemeManager | None" = None

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        self.theme = load_tokens()
        self._tpl = (THEME_DIR / "theme.qss.in").read_text()
        self.qss = ""
        self._fonts: tuple[str, str] | None = None       # (UI family, mono family), registered once

    def _icon_file(self, name: str, c: QColor) -> str:
        d = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / "theme-icons"
        d.mkdir(parents=True, exist_ok=True)
        svg = tinted_svg(name, c, 32)
        # the name carries a hash of the content: a changed icon is a new file, never the stale one
        f = d / f"{name}_{c.name(QColor.NameFormat.HexArgb)[1:]}_{zlib.crc32(svg):08x}.svg"
        if not f.exists():
            f.write_bytes(svg)
        return f.as_posix()

    def fill_template(self, tpl: str) -> str:
        t = self.theme

        def css(c: QColor) -> str:
            if c.alpha() == 255:
                return c.name().upper()
            return f"rgba({c.red()},{c.green()},{c.blue()},{c.alphaF():.3g})"

        def sub(m: re.Match) -> str:
            key = m.group(1).strip()
            if key.startswith("icon:"):
                parts = key.split(":")
                c = t.colour(parts[2]) if len(parts) > 2 else t.textSecondary
                return self._icon_file(parts[1], c or t.textSecondary)
            if key == "uiFamily":
                return t.uiFamily
            if key == "monoFamily":
                return t.monoFamily
            if key.startswith("rowHeight"):
                return str(t.row_height() + _to_int(key[9:]))  # "rowHeight-2"; like QString::toInt, junk counts as 0
            if key.startswith("radius.r"):
                return str(getattr(t.radius, key[7:], t.radius.r4))
            if key.startswith("space.s"):
                return str(getattr(t.space, key[6:], 0))
            if key.startswith("font."):
                return str(getattr(t.font, key[5:], t.font.body))
            v = getattr(t, key, None)
            if isinstance(v, int) and not isinstance(v, bool):
                return str(v)
            name, _, alpha = key.partition("@")
            c = t.colour(name)
            if c is None:
                return ""
            if alpha:
                c.setAlphaF(float(alpha))
            return css(c)

        return re.sub(r"\$\{([^}]+)\}", sub, tpl)

    @staticmethod
    def palette(t: Theme) -> QPalette:
        p = QPalette()
        groups = (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled)

        def all_(role, c):
            for g in groups:
                p.setColor(g, role, c)

        R = QPalette.ColorRole
        all_(R.Window, t.bgApp)
        all_(R.WindowText, t.textPrimary)
        all_(R.Base, t.bgSunken)
        all_(R.AlternateBase, t.bgDock)
        all_(R.Button, t.bgOverlay)
        all_(R.ButtonText, t.textPrimary)
        all_(R.Text, t.textPrimary)
        all_(R.PlaceholderText, t.textTertiary)
        all_(R.Highlight, t.accent_muted())
        all_(R.HighlightedText, t.textPrimary)
        all_(R.Accent, t.accent)
        all_(R.Link, t.accentHover)
        all_(R.ToolTipBase, t.bgTop)
        all_(R.ToolTipText, t.textPrimary)
        all_(R.BrightText, t.textPrimary)
        all_(R.Light, t.lineStrong)
        all_(R.Midlight, t.lineControl)
        all_(R.Mid, t.lineSubtle)
        all_(R.Dark, t.bgSunken)
        all_(R.Shadow, t.bgApp)
        for role in (R.Text, R.ButtonText, R.WindowText):
            p.setColor(QPalette.ColorGroup.Disabled, role, t.textDisabled)
        p.setColor(QPalette.ColorGroup.Disabled, R.Button, t.bgRaised)
        p.setColor(QPalette.ColorGroup.Inactive, R.Highlight, t.sel_inactive())
        return p

    def apply(self, app: QApplication) -> None:
        if self._fonts is None:
            fams = []
            for f in ("InterVariable.ttf", "JetBrainsMono-Variable.ttf"):
                fid = QFontDatabase.addApplicationFont(str(THEME_DIR / "fonts" / f))
                names = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                fams.append(names[0] if names else "")
            self._fonts = (fams[0], fams[1])
        fams = self._fonts
        if fams[0]:
            self.theme.uiFamily = fams[0]
        if fams[1]:
            self.theme.monoFamily = fams[1]
        app.setStyle(QStyleFactory.create("Fusion"))
        t = self.theme
        f = QFont(t.uiFamily)
        f.setPixelSize(t.font.body)
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        for tag, on in ((b"tnum", 1), (b"cv11", 1), (b"ss03", 1)):
            try:
                f.setFeature(QFont.Tag(tag.decode()), on)
            except (AttributeError, TypeError):
                pass
        app.setFont(f)
        app.setPalette(self.palette(t))
        self.qss = self.fill_template(self._tpl)
        app.setStyleSheet(self.qss)


def theme() -> Theme:
    return ThemeManager.instance().theme


def repolish(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()
