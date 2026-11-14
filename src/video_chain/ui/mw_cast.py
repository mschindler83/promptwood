"""Characters and their references."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImageReader
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QListWidgetItem

from .. import faces, tree as T
from . import dialogs as D, image_picker as IP
from .clip_inputs import KIND_NAMES
from .face_crop import FaceCropDialog
from .clip_cards import PromptCard
from .ui_logic import _SELECTED, plural


class CastMixin:
    """Characters and their references."""

    # ---- the cast (characters with face pictures, @Name in prompts)
    def _identity_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        ident = dict(self.tree.options.get("identity") or {})
        lang = self.dialogue_lang.currentText().strip()
        if not lang:                                # an emptied field: back to the language in use
            lang = ident.get("language") or "English"
            self.dialogue_lang.setCurrentText(lang)
        new = {"size": self.ref_size.currentData(), "hint": self.ref_hint.isChecked(), "language": lang}
        if all(ident.get(k) == v for k, v in new.items()):
            return
        before = self._done_clips()
        ident.update(new)
        self.tree.options["identity"] = ident
        self._show_references()
        self._changed()
        self._report_remade(before, "Face settings changed")

    def _cast_names(self) -> list[str]:
        return [c["name"] for c in T.cast_of(self.tree)] if self.tree is not None else []

    def _selected_character(self) -> dict | None:
        item = self.cast_list.currentItem()
        return T.character(self.tree, item.data(Qt.UserRole)) if item is not None and self.tree else None

    def _mention_map(self) -> dict[str, list[str]]:
        """Every character (lower-case name) -> the clips whose prompt (with the style text) mentions @name. One look
        at each clip, whatever the number of characters."""
        out: dict[str, list[str]] = {}
        if self.tree is None:
            return out
        for n in self.tree.nodes.values():
            if n.is_video:
                continue
            found = T.mentions(self.tree.options, T.full_prompt(self.tree.options, n.prompt))[0]
            for low in {c["name"].lower() for c in found}:
                out.setdefault(low, []).append(n.id)
        return out

    def _mentioning(self, name: str, mention_map: dict[str, list[str]] | None = None) -> list[str]:
        """Clips whose prompt (with the style text) mentions @name (mention_map: one already made for the list)."""
        return (self._mention_map() if mention_map is None else mention_map).get(name.lower(), [])

    def _cast_edit_ok(self) -> bool:
        if self.project is None or self.tree is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return False
        return True

    def _ref_word(self, c: dict | None = _SELECTED) -> str:
        """How the pictures of a cast entry are called: "face picture" for a person (and for none: a picture then
        goes to the lead or a new person), "picture" for a place, object or style."""
        c = self._selected_character() if c is _SELECTED else c
        return "picture" if c is not None and T.kind_of(c) != "person" else "face picture"

    def _new_character(self, kind: str = "person"):
        if not self._cast_edit_ok():
            return
        person = kind == "person"
        name, ok = QInputDialog.getText(self, "New character" if person else f"New {kind}",
                                        "Name (you mention it in prompts as @Name):",
                                        text=T.next_character_name(self.tree) if person else KIND_NAMES[kind])
        if not ok:
            return
        try:
            c = T.add_character(self.project, self.tree, name, kind=kind)
        except ValueError as e:
            self._report_error(f"Could not add the {'character' if person else kind}: {e}")
            return
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        self._say(f"@{c['name']} added. Now add a face: Image…, Start image or Selected clip." if person else
                            f"@{c['name']} added ({kind}). Now add a picture: Image…, Start image or Selected clip.")

    def _character_kind_changed(self):
        """The kind of the selected entry was chosen (person, place, object, style)."""
        c = self._selected_character()
        kind = self.char_kind.currentData()
        if c is None or self._loading or kind == T.kind_of(c):
            return
        if not self._cast_edit_ok():
            self._show_character()
            return
        before = self._done_clips()
        try:
            T.set_kind(self.tree, c["name"], kind)
        except ValueError as e:
            self._report_error(f"Could not change @{c['name']}: {e}.")
            self._show_character()                 # (the combo back to its kind)
            return
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        if not self._report_remade(before, f"@{c['name']} is a {kind} now"):
            self._say(f"@{c['name']} is a {kind} now" + (
                "." if kind == "person" else
                (" (no longer the lead)" if c.get("lead") else "") + "; its pictures stay as they are (Whole "
                "picture gives a cropped one its whole picture again)."))

    def _rename_character(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        new, ok = QInputDialog.getText(self, "Rename character", f"New name for @{c['name']}:", text=c["name"])
        if not ok or new.strip() == c["name"]:
            return
        try:
            T.check_name(self.tree, new, keep=c["name"])
        except ValueError as e:
            self._report_error(f"Could not rename: {e}")
            return
        users = self._mentioning(c["name"])
        made = [n for n in users if self.tree.nodes[n].made]
        if made and not D.confirm(
                self, "Rename character",
                f"{len(users)} clip{'s mention' if len(users) != 1 else ' mentions'} @{c['name']}; the prompts "
                f"change to @{new.strip()}. The name is part of what the model reads, so {len(made)} made "
                f"clip{'s' if len(made) != 1 else ''} will count as CHANGED."):
            return
        before = self._done_clips()
        try:
            T.rename_character(self.tree, c["name"], new)
        except ValueError as e:
            self._report_error(f"Could not rename: {e}")
            return
        self._show_references(select=new.strip())
        self._write_tree()
        style, self._loading = self.tree.options.get("style") or {}, True
        try:
            self.style_prefix.setText(style.get("prefix") or "")
            self.style_suffix.setText(style.get("suffix") or "")
        finally:
            self._loading = False
        self._rebuild_path()
        self._changed()
        if not self._report_remade(before, "Character renamed"):
            self._say(f"Renamed to @{new.strip()}.")

    def _remove_character(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        users = self._mentioning(c["name"])
        text = (f"Remove @{c['name']} and its {len(c['images'])} "
                f"{self._ref_word(c)}{'s' if len(c['images']) != 1 else ''}"
                + (" and its voice?" if c.get("voice") else "?"))
        if users:
            text += (f" {len(users)} clip{'s mention' if len(users) != 1 else ' mentions'} @{c['name']}; "
                     "they cannot be made until their prompts are changed.")
        if not D.confirm(self, "Remove character" if T.kind_of(c) == "person" else f"Remove {T.kind_of(c)}",
                                text):
            return
        before = self._done_clips()
        try:
            T.remove_character(self.project, self.tree, c["name"])
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not remove @{c['name']}: {e}")
            return
        self._pix.clear()
        self._show_references()
        self._write_tree()
        self._changed()
        if not self._report_remade(before, "Character removed"):
            self._say(f"@{c['name']} removed.")

    def _character_desc_changed(self):
        c = self._selected_character()
        text = self.char_desc.text().strip()
        if c is None or self._loading or text == c.get("desc", "") or not self._cast_edit_ok():
            return
        before = self._done_clips()
        T.update_character(self.tree, c["name"], desc=text)
        self._write_tree()
        self._changed()
        if not self._report_remade(before, f"Description of @{c['name']} changed"):
            self._say(f"Description of @{c['name']} saved (used by clips that mention it).")

    def _character_lead_changed(self, on: bool):
        c = self._selected_character()
        if c is None or self._loading or bool(c.get("lead")) == on or not self._cast_edit_ok():
            return
        before = self._done_clips()
        try:
            T.update_character(self.tree, c["name"], lead=on)
        except ValueError as e:
            self._report_error(f"Could not change the lead: {e}.")
            self._show_character()
            return
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        if not self._report_remade(before, "Lead changed"):
            self._say(f"@{c['name']} is the lead now." if on else "No lead: clips that mention nobody "
                                                                               "are made without a face.")

    def _done_clips(self) -> set[str]:
        return {nid for nid, st in self.status_of.items() if st == "done"}

    def _report_remade(self, before: set[str], what: str) -> int:
        """Say how many made clips a change turned CHANGED (they are made again with their paths), and how many
        are done again (a setting set back). What it said about CHANGED clips stays in self._remade_note (empty when
        none), so callers need not read it back from the status label."""
        self._remade_note = ""
        if self.tree is None:
            return 0
        try:
            checker = T.Checker(self.project, self.tree)
            now = {n for n in self.tree.nodes if checker.status(n) == "done"}
        except Exception:  # noqa: BLE001
            return 0
        lost = [n for n in before if n in self.tree.nodes and n not in now]
        back = [n for n in now if n not in before and n in self.status_of]   # e.g. a setting set back
        parts = []
        if lost:
            parts.append(f"{len(lost)} made clip{'s are' if len(lost) != 1 else ' is'} now CHANGED and will be made "
                         "again with their paths")
        if back:
            parts.append(f"{len(back)} clip{'s are' if len(back) != 1 else ' is'} DONE again")
        if parts:
            self._say(f"{what}: " + "; ".join(parts) + ".")
            if lost and not self._write_error:
                self._remade_note = self.status.text()
        return len(lost) + len(back)

    def _reference_from_selected(self):
        """The selected video's first frame, or a generated clip's last frame."""
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if nid else None
        if node is None or not node.made:
            self.status.setText("Select a converted video or a made clip first.")
            return
        self.reference_from_node(nid, "first" if node.is_video else "last")

    def _add_reference(self):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        c = self._selected_character()
        title = ("Reference image (best a close-up of the face)" if self._ref_word(c) == "face picture" else
                 f"Picture of @{c['name']} (a {T.kind_of(c)}; the whole picture is used)")
        f = IP.pick_image(self, title, self.qs, None, self._picker_places(), self.project_images())
        if f:
            self.add_reference(f)

    def _start_image_there(self) -> bool:
        return bool(self.project and (self.project / T.START_FILE).is_file())

    def _reference_from_start(self):
        if self.project is not None and self._start_image_there():
            self.add_reference(str(self.project / T.START_FILE))

    def reference_from_node(self, nid: str, which: str) -> bool:
        """A clip's first (videos only) or last frame as a face reference (a picture of a place, object, style)."""
        node = self.tree.nodes.get(nid) if self.tree else None
        rel = (node.made or {}).get("first" if which == "first" else "frame") if node else None
        if not rel:
            self.status.setText("That clip is not made yet.")
            return False
        if self.add_reference(str(self.project / rel)):
            if not self._remade_note:                                          # keep the CHANGED count
                self.status.setText(f"The {which} frame of #{T.short_id(nid)}: {self.status.text()}")
            return True
        return False

    def face_detector(self) -> Path | None:
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        return faces.find_detector(Path(comfy_dir)) if comfy_dir else None

    def crop_face(self, path: str, title: str, crop=None):
        """The crop window: the square (x, y, side), None for the whole picture, or False (cancelled)."""
        try:
            dlg = FaceCropDialog(QApplication.activeModalWidget() or self, Path(path), self.face_detector(), title,
                                 crop)                # over the video / image window it was opened from
        except ValueError as e:
            self.status.setText(str(e))
            return False
        try:
            return dlg.crop() if dlg.exec() == QDialog.Accepted else False
        finally:
            dlg.deleteLater()

    def create_reference(self, dialog: bool = True):
        """The Create image window for a face picture of the selected character (its "Add as face reference")."""
        c = self._selected_character()
        return self.edit_image(mode="create", dialog=dialog, character=c["name"] if c else None)

    def add_reference(self, path: str, crop="ask") -> bool:
        """A face picture for the selected character (none selected: the lead, else the first person, else a new
        character); for a selected place, object or style a picture (always whole: no crop window). crop: "ask"
        opens the crop window, else the square (x, y, side) or None (the whole picture)."""
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return False
        if not QImageReader(str(path)).canRead():      # (the header only: the picture is not decoded)
            self.status.setText(f"Not an image: {Path(path).name}")
            return False
        target = self._selected_character()
        owner = target or T.lead_of(self.tree) or next((c for c in T.cast_of(self.tree)
                                                        if T.kind_of(c) == "person"), None)
        word = self._ref_word(owner)
        if owner is not None and len(owner["images"]) >= T.MAX_PICTURES_PER_CHARACTER:
            self.status.setText(f"@{owner['name']} has {T.MAX_PICTURES_PER_CHARACTER} {word}s already; remove "
                                f"one first (or select another {'character' if word == 'face picture' else 'entry'}).")
            return False
        if len(T.all_pictures(self.tree)) >= T.MAX_CAST_PICTURES:
            self.status.setText(f"The cast has {T.MAX_CAST_PICTURES} pictures, the most the model takes; "
                                "remove one first.")
            return False
        if word == "picture":                      # a place, object or style: the whole picture, no crop window
            crop = None
        if crop == "ask":
            who = f"@{target['name']}" if target else (f"@{T.lead_of(self.tree)['name']}" if T.lead_of(self.tree)
                                                       else "a new character")
            crop = self.crop_face(path, f"Face picture of {who}")
            if crop is False:
                self.status.setText("Face not added.")
                return False
        before = self._done_clips()
        known = set(self._cast_names())
        try:
            rel = T.add_reference(self.project, Path(path), self.tree, target["name"] if target else None, crop=crop)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not add the {'face' if word == 'face picture' else 'picture'}: {e}")
            return False
        owner = next((c["name"] for c in T.cast_of(self.tree) if rel in c["images"]), None)
        if owner is None:
            self._report_error("The picture was added, but no character has it: look at the cast list.")
            self._show_references()
            return False
        self._show_references(select=owner)
        self._write_tree()
        self._changed()
        what = "Face" if word == "face picture" else "Picture"
        self._say(f"{what} added to @{owner}." if owner in known else
                            f"New character @{owner} with this face (rename it with Rename…).")
        self._report_remade(before, f"{what} added to @{owner}")
        return True

    def _recrop_reference(self):
        item = self.refs.currentItem()
        if item is None or not self._cast_edit_ok():
            return
        rel = item.data(Qt.UserRole)
        full = self.project / T.source_of(rel)
        whole = self._ref_word() == "picture"      # a place, object or style: its whole picture again, no window
        crop = None if whole else self.crop_face(str(full if full.is_file() else self.project / rel),
                                                 "Crop the face picture again")
        if crop is False:
            return
        before = self._done_clips()
        try:
            T.recrop_picture(self.project, self.tree, rel, crop)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not crop: {e}")
            return
        self._pix.clear()
        c = self._selected_character()
        self._show_references(select=c["name"] if c else None)
        self._write_tree()
        self._changed()
        what = "Whole picture used" if whole else "Face picture cropped"
        if not self._report_remade(before, what):
            self._say(f"{what}.")

    def _remove_reference(self):
        item = self.refs.currentItem()
        if item is None or not self._cast_edit_ok():
            return
        rel = item.data(Qt.UserRole)
        before = self._done_clips()
        made = [n for n in before if n in self.tree.nodes and not self.tree.nodes[n].is_video
                and rel in T.node_cast(self.tree.options, self.tree.nodes[n])["images"]]
        word = self._ref_word()
        if made and not D.confirm(
                self, f"Remove {word}",
                f"Remove this {word}? {len(made)} made clip{'s use' if len(made) != 1 else ' uses'} it and "
                f"will count as CHANGED."):
            return
        try:
            T.remove_picture(self.project, self.tree, rel)
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not remove the {word}: {e}")
            return
        self._pix.clear()
        c = self._selected_character()
        self._show_references(select=c["name"] if c else None)
        self._write_tree()
        self._changed()
        self._report_remade(before, f"{word.capitalize()} removed")

    def _show_references(self, select: str | None = None):
        """The cast list (keeps the selected character), then the selected one's fields and pictures."""
        cur = select or (self.cast_list.currentItem().data(Qt.UserRole) if self.cast_list.currentItem() else None)
        cast = T.cast_of(self.tree) if self.tree else []
        self.cast_list.blockSignals(True)
        self.cast_list.clear()
        row = 0
        mention_map = self._mention_map()          # (one look at the clips, not one per character)
        for k, c in enumerate(cast):
            pm = self._pixmap(self.project / c["images"][0]) if self.project and c["images"] else None
            n = len(c["images"])
            if T.kind_of(c) == "person":
                text = (f"@{c['name']}   {plural(n, 'face')}" if n else f"@{c['name']}   no face yet") + \
                    ("   · voice" if c.get("voice") else "") + ("   · lead" if c.get("lead") else "")
            else:
                text = (f"@{c['name']}   {plural(n, 'picture')}" if n else
                        f"@{c['name']}   no picture yet") + f"   · {T.kind_of(c)}"
            item = QListWidgetItem(QIcon(pm) if pm else QIcon(), text)
            item.setData(Qt.UserRole, c["name"])
            mentions = len(self._mentioning(c["name"], mention_map))
            item.setToolTip((c.get("desc") or "")
                            + f" ({mentions} clip{'s mention' if mentions != 1 else ' mentions'} it)")
            self.cast_list.addItem(item)
            if cur and c["name"].lower() == cur.lower():
                row = k
        if cast:
            self.cast_list.setCurrentRow(row)
        self.cast_list.blockSignals(False)
        self._show_character()
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.mentions.rehighlight()
        self._update_cards() if self.cards else None
        lead = T.lead_of(self.tree) if self.tree else None
        if not cast:
            text = ("No characters: each clip only continues from the last frame of the clip before, so a face can "
                    "drift. Add a face (Image…, Start image, Selected clip, or a video's Edit range…) or New… "
                    "for a named character.")
        else:
            people = sum(1 for c in cast if T.kind_of(c) == "person")
            others = len(cast) - people
            text = (f"{plural(people, 'character')}"
                    + (f" and {plural(others, 'reference')} (places, objects, style)" if others
                       else "") + ". Write @Name in a prompt (typing @ offers the names): the clip is made with "
                    "exactly those faces" + (" and pictures" if others else "") + ". "
                    + (f"Clips that mention nobody get @{lead['name']}'s face. " if lead and lead["images"] else
                       "Clips that mention nobody get no face. ")
                    + "A character with a voice speaks its lines in quotes with it (\"...\" in the sentence of "
                    "its @Name). LoRAs can act weaker in clips with faces.")
        self.ident_info.setText(text)

    def _show_character(self):
        c = self._selected_character()
        self._loading, was = True, self._loading
        try:
            self.char_desc.setText(c.get("desc", "") if c else "")
            self.char_lead.setChecked(bool(c and c.get("lead")))
            self.char_kind.setCurrentIndex(max(0, self.char_kind.findData(T.kind_of(c))))
        finally:
            self._loading = was
        self._show_kind_wording(c)
        v = (c or {}).get("voice")
        el = ((c or {}).get("tts") or {}).get("eleven")
        long = f"; cloned from {T.clone_secs(v):.1f} s" if v and v.get("ref") else ""
        self.char_voice.setText((f"Voice: {v.get('name') or Path(v['file']).name} "
                                 f"({float(v.get('secs') or 0):.1f} s{long})"
                                 if v else "Voice: none") + (f" · ElevenLabs: {el['name']}" if el else ""))
        self.refs.clear()
        for k, rel in enumerate(c["images"] if c else [], start=1):
            pm = self._pixmap(self.project / rel) if self.project else None
            item = QListWidgetItem(QIcon(pm) if pm else QIcon(), f"{c['name']} {k}")
            item.setData(Qt.UserRole, rel)
            item.setToolTip(rel)
            self.refs.addItem(item)
        self._update_ref_start_button()
        self._update_cast_buttons()

    def _show_kind_wording(self, c: dict | None):
        """The cast panel for the selected entry's kind: a person has face pictures (cropped), a voice and the lead
        switch; a place, object or style has (whole) pictures only."""
        person = c is None or T.kind_of(c) == "person"
        self._voice_row.setVisible(person)
        self.char_lead.setVisible(person)
        self.ref_add_label.setText("Add face:" if person else "Add picture:")
        if person:
            self.refs.setToolTip("Face pictures of the selected character (up to 3, e.g. front and side): best a "
                                 "sharp close-up of the face, without the scene around it")
            self.b_ref_add.setToolTip("Add an image file as a face picture of the selected character")
            self.b_ref_create.setToolTip("Create a face picture with AI (the Create image window); there “Add as "
                                         "face reference” adds a result to the selected character")
            self.b_ref_del.setToolTip("Remove the selected face picture")
            self.b_ref_crop.setText("Crop…")
            self.b_ref_crop.setToolTip("Crop the selected face picture again (from the full picture it came from)")
        else:
            k = T.kind_of(c)
            self.refs.setToolTip(f"Pictures of @{c['name']} (a {k}; up to {T.MAX_PICTURES_PER_CHARACTER}): the whole "
                                 "picture is sent, nothing is cropped")
            self.b_ref_add.setToolTip(f"Add an image file as a picture of @{c['name']} (the whole picture)")
            self.b_ref_create.setToolTip(f"Create a picture of @{c['name']} with AI (the Create image window); there "
                                         "the Add button adds a result to it")
            self.b_ref_del.setToolTip("Remove the selected picture")
            self.b_ref_crop.setText("Whole picture")
            self.b_ref_crop.setToolTip("Use the whole picture again for the selected one (if it was cropped as a "
                                       "face before)")
        self.char_desc.setPlaceholderText("Description (optional), e.g. woman, 30s, grey knit sweater" if person else
                                          {"place": "Description (optional), e.g. a small sunny kitchen",
                                           "object": "Description (optional), e.g. a red vintage bicycle",
                                           "style": "Description (optional), e.g. hand-drawn watercolour look"
                                           }[T.kind_of(c)])

    def _update_cast_buttons(self):
        """The cast buttons that need a selected character, voice or face picture (and no run)."""
        c = self._selected_character()
        idle = self.project is not None and not self.running
        v = bool(c and c.get("voice"))
        self.char_kind.setEnabled(idle and c is not None)
        for w in (self.b_char_rename, self.b_char_del, self.char_desc, self.char_lead, self.b_voice_file,
                  self.b_voice_tts):
            w.setEnabled(idle and c is not None)
        self.b_voice_del.setEnabled(idle and v)
        self.b_voice_clone.setEnabled(idle and v)
        self.b_voice_play.setEnabled(v)
        pic = self.refs.currentItem() is not None
        self.b_ref_crop.setEnabled(idle and pic)
        self.b_ref_del.setEnabled(idle and pic)
