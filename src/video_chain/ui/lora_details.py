"""What CivitAI tells about a LoRA (triggers.py keeps it) as a page of text: model and version, base model, trigger
words, the descriptions (often with sample prompts), files and license. Pictures in the descriptions are left out.
Shown in the LoRA window (lora_window), which has the gallery's prompts with their pictures."""
from __future__ import annotations

import html
import re


def clean_html(text: str) -> str:
    """CivitAI's description HTML without pictures, scripts, styles and embedded frames."""
    s = str(text or "")
    s = re.sub(r"(?is)<(script|style|iframe|object|embed)\b.*?</\1\s*>", "", s)
    s = re.sub(r"(?is)<(img|iframe|video|source|embed|link|meta|base)\b[^>]*>", "", s)
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


def details_html(details: dict, prompts: list[dict] | None = None, note: str = "", settings: str = "") -> str:
    """The page of a LoRA's CivitAI details. prompts (triggers.example_prompts): listed as text, link copy:N
    copies prompt N; None: left out (the LoRA window shows them with their pictures). note, settings: the prompt
    refiner's note on the LoRA and the settings its example pictures were made with (lora_notes)."""
    v = details.get("version") or {}
    m = details.get("model") or {}
    e = html.escape
    name = m.get("name") or (v.get("model") or {}).get("name") or "?"
    mid, vid = v.get("modelId") or m.get("id"), v.get("id")
    from .. import triggers
    url = f"{triggers.site()}/models/{mid}?modelVersionId={vid}" if mid else ""
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
    if settings:
        out.append(f"<h3>Settings of the example pictures</h3><p>{e(settings)}</p>")
    if note:
        out.append("<h3>How to prompt it (the prompt refiner's note, written from this page)</h3><p>"
                   + "<br>".join(e(x) for x in note.splitlines()) + "</p>")
    if v.get("description"):
        out.append("<h3>About this version</h3>" + clean_html(v["description"]))
    if m.get("description"):
        out.append("<h3>About the model</h3>" + clean_html(m["description"]))
    seen = details.get("examples_seen")
    shown, prompts = prompts is not None, prompts or []
    if shown:
        out.append(f"<h3>Prompts of the gallery's pictures ({len(prompts)}"
                   + (f", of {int(seen)} pictures looked at" if isinstance(seen, int) and seen else "") + ")</h3>")
    if shown and not prompts:
        out.append("<p><i>CivitAI's API gave no prompts for the example images (it often holds their generation "
                   "data back, at least without an API key: set yours in the settings; the LoRA's page on CivitAI may show them). Sample prompts in the descriptions above "
                   "are kept.</i></p>")
    for i, p in enumerate(prompts):
        gen = ", ".join(f"{k} {p[k]}" for k in ("seed", "steps", "cfg", "sampler") if p.get(k) not in (None, ""))
        out.append(f"<p><a href='copy:{i}'>[copy]</a> {e(p['prompt'])}"
                   + (f"<br><span style='color:gray'>negative: {e(p['negative'])}</span>" if p.get("negative") else "")
                   + (f"<br><span style='color:gray'>{e(gen)}</span>" if gen else "") + "</p>")
    files = v.get("files") or []
    if files:
        out.append("<h3>Files</h3><ul>" + "".join(
            f"<li>{e(str(f.get('name') or '?'))} ({e(_size(f.get('sizeKB')))})"
            f"<br><span style='color:gray'>SHA256 {e(str((f.get('hashes') or {}).get('SHA256') or ''))}</span></li>"
            for f in files if isinstance(f, dict)) + "</ul>")
    return "\n".join(out)
