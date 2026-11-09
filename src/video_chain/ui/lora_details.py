"""What CivitAI tells about a LoRA (triggers.py keeps it): model and version, base model, trigger words, the
descriptions (often with sample prompts), the prompts of its example images (one click copies one), files and
license. Text only: pictures in the descriptions are left out, nothing is loaded from the net while it is shown."""
from __future__ import annotations

import html
import re
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

from .. import triggers
from .widgets import label


def clean_html(text: str) -> str:
    """CivitAI's description HTML without pictures, scripts, styles and embedded frames."""
    s = str(text or "")
    s = re.sub(r"(?is)<(script|style|iframe|object|embed)\b.*?</\1\s*>", "", s)
    s = re.sub(r"(?is)<(img|iframe|video|source|embed)\b[^>]*>", "", s)
    s = re.sub(r"(?i)\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", s)
    # no styles or backgrounds (they could name local files), and links only to the web
    s = re.sub(r"(?i)(?<![\w-])(style|background|src)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", s)
    s = re.sub(r"(?i)(?<![\w-])href\s*=\s*(?:\"(?!https?://)[^\"]*\"|'(?!https?://)[^']*'|(?![\"']|https?://)[^\s>]+)",
               "", s)
    return s.strip()


def _size(kb) -> str:
    try:
        kb = float(kb)
    except (TypeError, ValueError):
        return ""
    return f"{kb / 1024 / 1024:.2f} GB" if kb >= 1024 * 1024 else f"{kb / 1024:.0f} MB"


def details_html(details: dict, prompts: list[dict]) -> str:
    """The page of a LoRA's CivitAI details (prompts: triggers.example_prompts; link copy:N copies prompt N)."""
    v = details.get("version") or {}
    m = details.get("model") or {}
    e = html.escape
    name = m.get("name") or (v.get("model") or {}).get("name") or "?"
    mid, vid = v.get("modelId") or m.get("id"), v.get("id")
    url = f"https://civitai.com/models/{mid}?modelVersionId={vid}" if mid else ""
    out = [f"<h2>{e(str(name))}</h2>"]
    facts = [("Version", v.get("name")), ("Base model", v.get("baseModel")),
             ("Type", m.get("type") or (v.get("model") or {}).get("type")),
             ("Creator", (m.get("creator") or {}).get("username")),
             ("Published", str(v.get("publishedAt") or "")[:10]),
             ("Tags", ", ".join(str(t) for t in m.get("tags") or [] if t))]
    stats = m.get("stats") or v.get("stats") or {}
    if stats:
        facts.append(("Stats", ", ".join(f"{k} {stats[k]}" for k in ("downloadCount", "thumbsUpCount", "rating")
                                         if stats.get(k) is not None)))
    lic = [("commercial use", m.get("allowCommercialUse")), ("no credit needed", m.get("allowNoCredit")),
           ("derivatives", m.get("allowDerivatives"))]
    if any(x is not None for _k, x in lic):
        facts.append(("License", "; ".join(f"{k}: {', '.join(x) if isinstance(x, list) else x}"
                                          for k, x in lic if x is not None)))
    out.append("<table cellspacing='2'>" + "".join(
        f"<tr><td><b>{e(k)}</b>&nbsp;&nbsp;</td><td>{e(str(x))}</td></tr>" for k, x in facts if x) + "</table>")
    if url:
        out.append(f"<p><a href='{e(url)}'>{e(url)}</a></p>")
    words = v.get("trainedWords") or []
    out.append("<h3>Trigger words</h3><p>" + (e(", ".join(str(w) for w in words)) if words else
                                               "<i>none given</i>") + "</p>")
    if v.get("description"):
        out.append("<h3>About this version</h3>" + clean_html(v["description"]))
    if m.get("description"):
        out.append("<h3>About the model</h3>" + clean_html(m["description"]))
    out.append(f"<h3>Prompts of the example images ({len(prompts)})</h3>")
    if not prompts:
        out.append("<p><i>CivitAI's API gave no prompts for the example images (it often holds their generation "
                   "data back, at least without an API key: set yours in the settings; the page on civitai.com may show them). Sample prompts in the descriptions above "
                   "are kept.</i></p>")
    for i, p in enumerate(prompts):
        settings = ", ".join(f"{k} {p[k]}" for k in ("seed", "steps", "cfg", "sampler") if p.get(k) not in (None, ""))
        out.append(f"<p><a href='copy:{i}'>[copy]</a> {e(p['prompt'])}"
                   + (f"<br><span style='color:gray'>negative: {e(p['negative'])}</span>" if p.get("negative") else "")
                   + (f"<br><span style='color:gray'>{e(settings)}</span>" if settings else "") + "</p>")
    files = v.get("files") or []
    if files:
        out.append("<h3>Files</h3><ul>" + "".join(
            f"<li>{e(str(f.get('name') or '?'))} ({e(_size(f.get('sizeKB')))})"
            f"<br><span style='color:gray'>SHA256 {e(str((f.get('hashes') or {}).get('SHA256') or ''))}</span></li>"
            for f in files if isinstance(f, dict)) + "</ul>")
    return "\n".join(out)


class LoraDetailsDialog(QDialog):
    def __init__(self, path: str, parent=None, details: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"CivitAI: {Path(path).stem}")
        self.resize(820, 760)
        self.details = details if details is not None else triggers.db().details(path)
        self.prompts = triggers.example_prompts(self.details)
        v = QVBoxLayout(self)
        self.view = QTextBrowser(self)
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._link)
        v.addWidget(self.view, 1)
        self.note = label("", level="tertiary")
        v.addWidget(self.note)
        if self.details:
            self.view.setHtml(details_html(self.details, self.prompts))
            self.note.setText("Click [copy] to copy an example prompt. Links open in your browser.")
        else:
            info = triggers.db().info(path) or {}
            self.view.setHtml("<p>CivitAI has told nothing about this LoRA yet.</p><p>"
                              + ("CivitAI does not know this file." if info.get("source") == "none" or
                                 (info and not info.get("url")) else
                                 "It is looked up in the background when \"Look up LoRA trigger words on CivitAI\" "
                                 "is on in the settings.") + "</p>")
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def _link(self, url: QUrl):
        s = url.toString()
        if s.startswith("copy:"):
            try:
                p = self.prompts[int(s[5:])]
            except (ValueError, IndexError):
                return
            QGuiApplication.clipboard().setText(p["prompt"])
            self.note.setText(f"Copied example prompt {int(s[5:]) + 1}.")
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
