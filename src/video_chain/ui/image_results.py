"""The image dialog's results: the strip below the large view, the gallery, the filters, favourites, the selected
result's facts, and deleting with Undo.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QListWidgetItem

from .. import edits
from ..edits import ResultRecord
from .image_common import RESULT_ICON
from . import settings_keys as SK
from .imageio import upright
from .lora_stack import lora_label
from .results_gallery import badged, model_key, model_name, run_groups
from .image_host import ImageEditHost



class ResultsMixin(ImageEditHost):
    # ---- results
    def _load_results(self, select: str | None = None, keep: bool = False):
        """The results with the filters applied: the strip below the large view and the gallery (grouped by
        Generate run); thumbnails are decoded in the background (each file once)."""
        cur = self.selected()
        if select is None and (keep or cur is not None):
            select = cur[0] if cur else None
        entries = []
        for png_file, meta in edits.project_edits(self.project):
            try:
                entries.append((str(png_file), meta, png_file.stat().st_mtime, png_file.stat().st_mtime_ns))
            except OSError:
                continue
        self._entries = [(p, m, t) for p, m, t, _ns in entries]
        used = {p for p, m, _t in self._entries if self.uses(m)}
        self._in_use = used
        models = sorted({model_key(m) for _p, m, _t in self._entries}, key=model_name)
        want = self.f_model.currentData()
        self.f_model.blockSignals(True)
        self.f_model.clear()
        self.f_model.addItem("All models", None)
        for k in models:
            self.f_model.addItem(model_name(k), k)
        self.f_model.setCurrentIndex(max(0, self.f_model.findData(want)))
        self.f_model.blockSignals(False)
        shown = [(p, m, t, ns) for p, m, t, ns in entries if self._passes(p, m)]
        self._shown = shown
        self.results.blockSignals(True)
        self.results.clear()
        # png -> its item (a thumbnail arriving finds it at once)
        self._strip_items: dict[str, QListWidgetItem] = {}
        for png, meta, _t, ns in shown:
            self.results.addItem(self._strip_item(png, meta, ns))
        self.results.blockSignals(False)
        self.gallery.set_groups(run_groups([(p, m, t) for p, m, t, _ns in shown]), used)
        for png, (_ns, pix) in self._thumbs.items():
            self.gallery.set_thumb(png, pix)
        self._show_count(len(shown), len(entries))
        alive = {p for p, *_r in entries}
        for png in [p for p in self._thumbs if p not in alive]:     # deleted results: their thumbnails go
            del self._thumbs[png]
        if self.results.count():
            row = next((i for i in range(self.results.count())
                        if self.results.item(i).data(Qt.ItemDataRole.UserRole)[0] == select), 0)
            self.results.blockSignals(True)            # (shown once below, not also through the signal)
            self.results.setCurrentRow(row)
            self.results.blockSignals(False)
            self._strip_selection(self.results.currentItem(), None)
        self._show_selected()

    def _passes(self, png: str, meta: ResultRecord) -> bool:
        """Whether the filters (favourites / in use, model, search words) show this result."""
        show, model = self.f_show.currentData(), self.f_model.currentData()
        if show == "fav" and not meta.get("favorite"):
            return False
        if show == "used" and png not in self._in_use:
            return False
        if model and model_key(meta) != model:
            return False
        text = " ".join(str(meta.get(k) or "") for k in ("prompt", "negative")).lower()
        return all(w in text for w in self.f_search.text().lower().split())

    def _show_count(self, n: int, total: int):
        self.f_count.setText(f"{n} result{'s' if n != 1 else ''}" if n == total else f"{n} of {total}")

    def _strip_item(self, png: str, meta: ResultRecord, ns: int) -> QListWidgetItem:
        """A strip item for a result; its thumbnail is asked for unless it is there or was asked (or failed)."""
        it = QListWidgetItem(self._strip_icon(png, meta), "")
        it.setData(Qt.ItemDataRole.UserRole, (png, meta))
        it.setToolTip(meta.get("prompt", Path(png).name))
        self._strip_items[png] = it
        if self._thumbs.get(png, (None,))[0] != ns and (png, ns) not in self._requested \
                and (png, ns) not in self._failed:
            self._requested.add((png, ns))
            self._loader.request(png, ns)
        return it

    def _result_added(self, png: str, meta: ResultRecord):
        """A new result (the newest) joins the strip and the gallery without reloading the others. When the filters
        hide it, it is a new model, or the strip is empty, everything is loaded as usual."""
        try:
            st = Path(png).stat()
        except OSError:
            return self._load_results(select=png)
        used = bool(self.uses(meta))
        if used:
            self._in_use.add(png)
        known = self.f_model.findData(model_key(meta)) >= 0
        if png in self._strip_items or not known or not self.results.count() or not self._passes(png, meta):
            self._in_use.discard(png)
            return self._load_results(select=png)
        self._entries.insert(0, (png, meta, st.st_mtime))
        self._shown.insert(0, (png, meta, st.st_mtime, st.st_mtime_ns))
        prev = self.results.currentItem()
        self.results.blockSignals(True)
        self.results.insertItem(0, self._strip_item(png, meta, st.st_mtime_ns))
        self.results.setCurrentRow(0)
        self.results.blockSignals(False)
        self._strip_selection(self.results.currentItem(), prev)
        self.gallery.set_groups(run_groups([(p, m, t) for p, m, t, _ns in self._shown]), self._in_use)
        self._show_count(len(self._shown), len(self._entries))
        self._show_selected()

    def _strip_icon(self, png: str, meta: edits.ResultRecord, selected: bool = False) -> QIcon:
        pix = self._thumbs.get(png, (None, QPixmap()))[1]
        return QIcon(badged(pix, RESULT_ICON, "", bool(meta.get("favorite")), png in self._in_use, selected,
                            self.devicePixelRatioF()))

    def _strip_selection(self, cur, prev):
        for it, on in ((prev, False), (cur, True)):
            if it is not None:
                png, meta = it.data(Qt.ItemDataRole.UserRole)
                it.setIcon(self._strip_icon(png, meta, on))

    def _thumb_ready(self, png: str, ns: int, img: QImage):
        self._requested.discard((png, ns))
        if img.isNull():
            self._failed.add((png, ns))                # (not asked again until the file changes)
            return
        pix = QPixmap.fromImage(img)
        self._thumbs[png] = (ns, pix)
        it = self._strip_items.get(png)
        if it is not None:
            it.setIcon(self._strip_icon(png, it.data(Qt.ItemDataRole.UserRole)[1], it is self.results.currentItem()))
        self.gallery.set_thumb(png, pix)

    def set_view(self, view: str, remember: bool = True):
        gallery = view == "gallery"
        self.stack.setCurrentIndex(1 if gallery else 0)
        (self.b_gallery if gallery else self.b_single).setChecked(True)
        self.thumb_size.setVisible(gallery)
        if remember and self.qs is not None:
            self.qs.setValue(SK.RESULTS_VIEW, "gallery" if gallery else "single")
        sel = self.selected()
        if gallery:
            self.gallery.set_selected(sel[0] if sel else None)
            self.gallery.setFocus()
        else:
            self.results.setFocus()
            it = self.results.currentItem()
            if it is not None:
                self.results.scrollToItem(it)

    def view_mode(self) -> str:
        return "gallery" if self.stack.currentIndex() == 1 else "single"

    def _thumb_size_changed(self, px: int):
        self.gallery.set_thumb_size(px)
        if self.qs is not None:
            self.qs.setValue(SK.RESULTS_THUMB, px)

    def _select_png(self, png: str):
        for i in range(self.results.count()):
            if self.results.item(i).data(Qt.ItemDataRole.UserRole)[0] == png:
                self.results.setCurrentRow(i)
                return

    def _gallery_clicked(self, png: str):
        self._select_png(png)
        self.gallery.setFocus()

    def _gallery_opened(self, png: str):
        self._select_png(png)
        self.set_view("single")

    def toggle_favorite(self, png: str | None = None) -> bool:
        """★ a result (the selected one if none is given) or take its star away; kept in its record. A result
        whose record is not written yet (still being made) or cannot be read is left alone."""
        sel = self.selected()
        png = png or (sel[0] if sel else None)
        if not png:
            return False
        meta = edits.read_meta(png)
        if meta is None:
            self.state.setText("This result's record is not written yet (or cannot be read): try again in a moment.")
            return False
        meta["favorite"] = not meta.get("favorite")
        if not meta["favorite"]:
            meta.pop("favorite")
        try:
            edits.write_meta(Path(png).with_suffix(".json"), meta)
        except OSError as e:
            self.state.setText(f"Could not mark it: {e}")
            return False
        if self.f_show.currentData() == "fav" and not meta.get("favorite"):
            self._load_results(keep=True)              # (it leaves the Favourites view)
        else:                                          # in place: no rebuild, the view stays where it is
            for i in range(self.results.count()):
                it = self.results.item(i)
                if it.data(Qt.ItemDataRole.UserRole)[0] == png:
                    it.setData(Qt.ItemDataRole.UserRole, (png, meta))
                    it.setIcon(self._strip_icon(png, meta, it is self.results.currentItem()))
            self.gallery.update_thumb(png, meta)
            if sel is not None and sel[0] == png:
                self.b_fav.setChecked(bool(meta.get("favorite")))
        if self.view_mode() == "gallery":
            self.gallery.setFocus()
        return bool(meta.get("favorite"))

    def _use_seed(self, seed: int):
        self.seed.setValue(int(seed))
        self.fresh_seed.setChecked(False)
        self.state.setText(f"Seed {seed} is set for the next Generate (New seed after each Generate is off).")

    def selected(self) -> tuple[str, edits.ResultRecord] | None:
        """The selected result and its record, read again (a use is recorded while the window is open)."""
        it = self.results.currentItem()
        if it is None:
            return None
        png, meta = it.data(Qt.ItemDataRole.UserRole)
        return png, edits.read_meta(png) or meta

    def _show_selected(self):
        sel = self.selected()
        if sel is None:
            self.view.clear_frame("Results appear here")
            self.res_info.set(None)
            self.view.setToolTip("")
            self.gallery.set_selected(None)
            self.b_fav.setChecked(False)
        else:
            png, meta = sel
            orig = meta.get("source_copy") or meta.get("source")
            show = orig if self.b_before.isChecked() and orig and Path(orig).is_file() else png
            img = upright(show)                      # (a phone photo turned, like the result)
            self.view.set_image(img)
            how = edits.how_made(meta)
            smp = " · ".join(x for x in (f"{meta['steps']} steps" if meta.get("steps") else "",
                                         f"CFG {meta['cfg']:g}" if isinstance(meta.get("cfg"), (int, float)) else "",
                                         meta.get("sampler") or "", meta.get("scheduler") or "") if x)
            w, h = meta.get("width"), meta.get("height")
            if not (w and h) and not img.isNull():
                w, h = img.width(), img.height()
            self.res_info.set({
                "original": show != png, "where": self.uses(meta), "model": how.strip(), "sampling": smp,
                "size": f"{w} × {h}" if w and h else "",
                "loras": [f"{lora_label(lo['path'])} ×{lo['strength']:g}" for lo in meta.get("loras") or []],
                "seed": meta.get("seed"),
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(edits.mtime(Path(png)))),
                "values": (meta.get("variation") or {}).get("values") or {},
                "prompt": meta.get("prompt", ""), "trigger_words": meta.get("trigger_words") or [],
                "negative": meta.get("negative") or ""})
            self.view.setToolTip(self.res_info.toolTip())
            self.gallery.set_selected(png, scroll=self.view_mode() == "gallery")
            self.b_fav.setChecked(bool(meta.get("favorite")))
        self._update_buttons()

    def delete_selected(self):
        """Delete the selected result if it is not in use; Undo delete brings the deleted ones back (the last one
        first) until the window closes."""
        sel = self.selected()
        if sel is None or self._busy or self.uses(sel[1]):
            return
        png = Path(sel[0])
        row = self.results.currentRow()
        self._trash.append(edits.trash_result(self.project, png))
        self._load_results()
        if self.results.count():
            self.results.setCurrentRow(min(row, self.results.count() - 1))
        self.b_undo.show()
        self.state.setText(f"Deleted {png.name}.")

    def undo_delete(self):
        if not self._trash:
            return
        back = edits.restore_result(self.project, self._trash.pop())
        self.b_undo.setVisible(bool(self._trash))
        self._load_results(select=str(back))
        self.state.setText(f"{back.name} is back.")

    def _empty_trash(self, keep_sources: bool = False):
        """Delete for good what was deleted, and originals no result needs any more."""
        edits.empty_trash(self.project, keep_sources)
        self._trash: list[Path] = []
