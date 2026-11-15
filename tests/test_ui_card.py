"""GUI tests: the clip card, its controls and chips, the Simple view, layouts and the inspector.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import json
import os


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    FAKE, PromptCard_open, _chips, _make_sound, _row_hidden, app, crop_accepted, flush, image, new_project,
    run_and_wait, silent_player, type_prompt, wait, win)


def test_undo_changes_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("walks in", "sits down"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    type_prompt(win, ids[0], "walks in slowly")
    flush(app, win)
    card = win.cards[0]
    assert win.state(ids[0]) == "changed" and win.state(ids[1]) == "changed"
    assert card.b_revert.isVisibleTo(card) and "prompt changed" in card.step.text()
    assert not win.cards[1].b_revert.isVisibleTo(win.cards[1])      # only changed because of the clip before
    card.b_revert.click()
    flush(app, win)
    assert win.cards[0].edit.toPlainText() == "walks in"
    assert [win.state(i) for i in ids] == ["done", "done"] and "done again" in win.status.text()
    assert T.load(tmp_path / "proj").nodes[ids[0]].prompt == "walks in"


def test_clearer_controls(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert card.tag.text() == "NEW" and "Not made yet" in card.tag.toolTip()
    assert not win.b_play_path.isEnabled() and "2 clips on this path are new or changed" in win.b_play_path.toolTip()
    assert card.settings_toggle.text() == "Own settings: project defaults"
    card.settings_toggle.click()
    assert card.own.isVisibleTo(card) and PromptCard_open(win)
    assert win.cards[1].own.isVisibleTo(win.cards[1])                    # every card opens together
    card.length.setCurrentIndex(card.length.findData(90))
    flush(app, win)
    assert win.cards[0].settings_toggle.text() == "Own settings: 3.8 s" and win.cards[0].own.isVisibleTo(win.cards[0])
    win.cards[0].settings_toggle.click()
    assert not win.cards[1].own.isVisibleTo(win.cards[1]) and not PromptCard_open(win)
    assert not win._clips_form.isRowVisible(win.turbo_lora)              # turbo settings only in turbo mode
    win.speed.setCurrentIndex(win.speed.findData("turbo"))
    assert win._clips_form.isRowVisible(win.turbo_lora)
    win.speed.setCurrentIndex(win.speed.findData("normal"))
    run_and_wait(app, win, win.generate_path)
    assert win.b_play_path.toolTip() == "Join the clips of the selected path and play them"


def test_wheel_does_not_change_unfocused_fields(app, win, tmp_path, image):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    ids = new_project(win, tmp_path, image, ("a",))
    card = win.cards[0]

    def wheel(w):
        e = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(w, e)
    for field, value in ((win.resolution, lambda: win.resolution.currentIndex()), (card.seed, card.seed.value),
                         (card.length, card.length.currentIndex)):
        before = value()
        wheel(field)
        assert value() == before, field
    flush(app, win)
    assert win.state(ids[0]) == "new" and win.tree.nodes[ids[0]].seed == card.seed.value()
    card.seed.setFocus()
    app.processEvents()
    if card.seed.hasFocus():                                     # focused (clicked into): the wheel works
        before = card.seed.value()
        wheel(card.seed)
        assert card.seed.value() != before


def test_no_delete_while_files_are_busy_and_small_card_fixes(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image)
    win._joins = 1                                                    # a path is being joined
    win.node_action("delete", ids[1])
    assert ids[1] in win.tree.nodes and "Wait until the path is joined" in win.status.text()
    win._joins = 0
    win.refresh()
    # carry-over has nothing to take from on the first clip
    assert not win.cards[0].guide.isEnabled() and win.cards[1].guide.isEnabled()
    # a made prompt that differs only in spacing stays as made
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    n = win.tree.nodes[ids[0]]
    n.made["prompt"] = n.prompt = "one\ntwo  three"
    win.refresh(rebuild_path=True)
    win.cards[0].edit.setPlainText("one two three")
    assert n.prompt == "one\ntwo  three"
    win.cards[0].edit.setPlainText("one two four")
    assert n.prompt == "one two four"


def test_style_text_in_the_window(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("walks in",))
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    win.style_prefix.setText("cinematic,")
    win.style_prefix.editingFinished.emit()
    assert win.tree.options["style"] == {"prefix": "cinematic,", "suffix": ""}
    assert "1 made clip is now CHANGED" in win.status.text()
    flush(app, win)
    assert win.cards[0].words.text().endswith("+ style")
    assert win.cards[0].edit.toolTip() == "Sent to the model: cinematic, walks in"
    win.style_suffix.setText("slow motion")                             # not finished with Enter: taken at Generate
    run_and_wait(app, win, win.generate_path)
    made = T.load(win.project).nodes[ids[0]].made["settings"]
    assert made["prompt_prefix"] == "cinematic," and made["prompt_suffix"] == "slow motion"
    wait(app, lambda: not win._joins)                                    # (a join still running blocks leaving)
    win.close_project()
    win.open_project(tmp_path / "proj2")                                 # a new project starts without it
    assert win.style_prefix.text() == "" and (win.tree.options.get("style") or {}).get("prefix", "") == ""


def test_inspector_and_timeline(app, win, tmp_path, image):
    """The editor layout: the path as a timeline; the clip in focus in the inspector; clicking a clip of the path
    shows it without changing the path; a new clip gets the focus."""
    ids = new_project(win, tmp_path, image, ("one", "two", "three"))
    assert [b.nid for b in win.timeline._blocks] == ids and win.focus == ids[-1]
    assert win.card_stack.currentWidget().nid == ids[-1] and "3 of 3" in win.focus_info.text()
    win.timeline.clicked.emit(ids[0])                      # a block: shown, the path stays
    assert win.card_stack.currentWidget().nid == ids[0] and [c.nid for c in win.cards] == ids
    assert not win.b_prev_clip.isEnabled() and win.b_next_clip.isEnabled()
    win._step_focus(1)
    assert win.focus == ids[1]
    win._graph_clicked(ids[2])                             # a clip of the path in the graph: focus only
    assert win.focus == ids[2] and win.selected_id() == ids[2]
    win._graph_clicked(ids[0])
    assert win.focus == ids[0] and [c.nid for c in win.cards] == ids    # (the path is not cut short)
    win.node_action("continue", ids[2])                    # a new clip: in focus at once, to write its prompt
    assert win.focus == win.selected_id() not in ids and win.card_stack.currentWidget().nid == win.focus
    assert win.timeline._blocks[-1].nid == win.focus and win.inspector.tabText(0) == "Clip"


def test_layout_presets_and_saved_layout(app, win):
    """View presets size the panels; the layout is saved with the settings and restored; an unknown one is reset."""
    from video_chain.ui.main_window import LAYOUT_VERSION
    win.resize(1600, 900)
    win.apply_layout("watch")
    assert not win.act_left.isChecked() and win.left_tabs.isHidden()
    win.apply_layout("write")
    assert win.act_left.isChecked() and not win.left_tabs.isHidden()
    sizes = win.split.sizes()
    assert sizes[2] > sizes[0]                       # a wide inspector for writing
    win.save_settings()
    d = json.loads(win.qs.value("layout", type=str))
    assert d["v"] == LAYOUT_VERSION and d["left"] is True and len(d["split"]) == 3
    win.apply_layout("watch")
    win._restore_layout()
    assert win.act_left.isChecked() and win.split.sizes()[2] > win.split.sizes()[0]
    win.qs.setValue("layout", "not json")
    win._restore_layout()                            # falls back to the default layout
    assert win.act_left.isChecked()
    for bad in ("[]", "5", '{"v": 1, "split": ["a", 1, 1], "center": [1, 1]}',
                '{"v": 1, "split": [0, 0, 0], "center": [1, 1]}'):
        win.apply_layout("watch")
        win.qs.setValue("layout", bad)
        win._restore_layout()
        assert win.act_left.isChecked(), bad
    # hidden, saved, restored and shown again: the left panel gets its width back
    win.show()
    app.processEvents()
    win.apply_layout("edit")
    app.processEvents()
    win.act_left.setChecked(False)
    win.save_settings()
    win._restore_layout()
    assert not win.act_left.isChecked()
    win.act_left.setChecked(True)
    app.processEvents()
    assert win.split.sizes()[0] >= win.left_tabs.minimumWidth()
    win.act_left.setChecked(False)
    win._show_left_tab(1)                            # a tab chosen from the menu shows the panel
    assert win.act_left.isChecked() and win.left_tabs.currentIndex() == 1
    win.hide()


def test_start_node_opens_the_project_tab(app, win, tmp_path, image):
    """A click on the Start node shows the start image settings (the inspector's Project tab)."""
    from video_chain.ui.graph import START
    ids = new_project(win, tmp_path, image, ("one",))
    assert win.inspector.currentIndex() == 0
    win._graph_clicked(START)
    assert win.inspector.currentWidget() is win.proj_scroll
    win._graph_clicked(ids[0])                               # a clip: back to the Clip tab
    assert win.inspector.currentIndex() == 0


def test_buttons_open_their_windows(app, win, tmp_path, image, monkeypatch):
    """Real clicks (a button's clicked passes checked=False: it must not land in an optional parameter)."""
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    opened = []
    monkeypatch.setattr(M.ExportDialog, "exec", lambda dlg: opened.append(dlg.windowTitle()) or 0)
    win.b_export.click()                                                    # Export path…: the path's dialog
    assert opened == ["Export path"], (opened, win.status.text())
    win.select(ids[0])
    flush(app, win)
    win.b_export_clip.click()
    assert opened[-1] == f"Export clip #{ids[0][1:]}"
    seen = []
    monkeypatch.setattr(win, "edit_image", lambda **kw: seen.append(kw))
    win.b_ref_create.click()                                                # Cast → Create…: the window opens
    assert seen and seen[0]["mode"] == "create"


def test_civitai_api_key_field(app, win):
    """The settings field keeps the key in the (test's in-memory) keyring, never shows it back, and removes it
    when saved empty."""
    from video_chain import triggers
    win.triggers_online.setChecked(False)
    win.civitai_key.setText(" abc123 ")
    win._save_civitai_key()
    assert triggers.api_key() == "abc123" and win.civitai_key.text() == ""
    assert "kept" in win.civitai_key.placeholderText() and "keyring" in win.status.text()
    win._save_civitai_key()                                   # empty: removed
    assert triggers.api_key() == "" and "none" in win.civitai_key.placeholderText()


def test_clip_chips_and_the_model_prompt(app, win, tmp_path, image, monkeypatch):
    """The chips under a clip's prompt, in the order the model gets its inputs; the lead chip switches the clip's
    lead; the Model prompt window shows exactly what the model gets."""
    ids = new_project(win, tmp_path, image, ("A cat jumps.",))
    flush(app, win)
    card, chips = _chips(win, ids[0])
    assert chips == ["Start frame", "From the start frame"] and not card.chips.isHidden()
    assert card.chips.chips[0].toolTip() == "The clip starts exactly from this frame (start.png)."
    # the official format: a person, a place and the start frame numbered after their pictures
    win.prompt_format.setCurrentIndex(win.prompt_format.findData("h3"))
    p = win.project
    for k, name in enumerate(("a1", "a2", "k")):
        Image.new("RGB", (64, 80), (200 - 10 * k, 150, 130)).save(tmp_path / f"{name}.png")
    T.add_character(p, win.tree, "Anna", tmp_path / "a1.png", desc="woman, 30s")
    T.add_picture(p, win.tree, "Anna", tmp_path / "a2.png")
    T.add_character(p, win.tree, "Kitchen", tmp_path / "k.png", desc="a small sunny kitchen", kind="place")
    win._show_references()
    type_prompt(win, ids[0], "@Anna walks into @Kitchen.")
    flush(app, win)
    card, chips = _chips(win, ids[0])
    assert chips == ["Start frame · Picture 4", "@Anna · face ×2", "@Kitchen · place", "With reference pictures"]
    tips = [b.toolTip() for b in card.chips.chips]
    assert "<Picture 4>" in tips[0] and "<Picture 1> and <Picture 2>" in tips[1] and "<Picture 3>" in tips[2]
    info = T.clip_inputs(p, win.tree, ids[0])
    assert card.edit.toolTip() == "Sent to the model: " + info["prompt"]
    win.modal_dialogs = False
    card.chips.b_prompt.click()
    dlg = win._model_prompt_dialog
    assert dlg.text.toPlainText() == info["prompt"] and dlg.text.isReadOnly()
    assert dlg.inputs.text().splitlines() == ["<Picture 1> — @Anna (ref1.png)", "<Picture 2> — @Anna (ref2.png)",
                                              "<Picture 3> — @Kitchen (ref3.png)",
                                              "<Picture 4> — start frame (start.png)"]
    dlg.b_copy.click()
    assert QApplication.clipboard().text() == info["prompt"]
    dlg.deleteLater()
    # the implicit lead: a checkable chip; off, the clip has no lead pictures and the chip stays to switch it on
    type_prompt(win, ids[0], "She waves.")
    flush(app, win)
    card, chips = _chips(win, ids[0])
    assert chips == ["Start frame · Picture 3", "@Anna · lead", "With reference pictures"]
    lead = card.chips.chips[1]
    assert lead.isCheckable() and lead.isChecked() and lead.isEnabled()
    lead.click()
    assert win.tree.nodes[ids[0]].overrides == {"lead": False} and "without @Anna" in win.status.text()
    flush(app, win)
    card, chips = _chips(win, ids[0])
    assert chips == ["Start frame", "@Anna · lead off", "From the start frame"]
    assert not card.chips.chips[1].isChecked()
    s = T.Checker(p, win.tree).settings(ids[0])
    assert "refs" not in s and "frame_pictures" not in s and "no lead" in card.settings_toggle.text()
    card.cam_move.setCurrentIndex(card.cam_move.findData("pan-left"))              # another own setting: kept
    assert win.tree.nodes[ids[0]].overrides == {"camera": {"move": "pan-left"}, "lead": False}
    card.cam_move.setCurrentIndex(0)
    flush(app, win)
    assert T.load(p).nodes[ids[0]].overrides == {"lead": False}                    # saved
    card, _chips_now = _chips(win, ids[0])
    card.chips.chips[1].click()                                                     # on again: as before
    assert win.tree.nodes[ids[0]].overrides == {}
    flush(app, win)
    assert _chips(win, ids[0])[1] == ["Start frame · Picture 3", "@Anna · lead", "With reference pictures"]
    assert T.Checker(p, win.tree).settings(ids[0])["frame_pictures"] == ["start"]


def test_prompt_format_choice(app, win, tmp_path, image):
    """Project → Prompt format switches options["prompt_format"]; the official format disables the face hint; a
    project file from before the setting opens as classic."""
    new_project(win, tmp_path, image, ("a",))
    assert win.prompt_format.currentData() == "classic" and win.ref_hint.isEnabled()      # (the tests' default)
    win.prompt_format.setCurrentIndex(win.prompt_format.findData("h3"))
    assert win.tree.options["prompt_format"] == "h3" and not win.ref_hint.isEnabled()
    assert "official H3 format always ties" in win.ref_hint.toolTip()
    win._update_buttons()                                                           # (the idle loop keeps it off)
    assert not win.ref_hint.isEnabled()
    flush(app, win)
    assert T.load(win.project).options["prompt_format"] == "h3"
    win.prompt_format.setCurrentIndex(win.prompt_format.findData("classic"))
    assert win.tree.options["prompt_format"] == "classic" and win.ref_hint.isEnabled()
    old = tmp_path / "old"
    d = T.Tree().to_json()
    del d["options"]["prompt_format"]                                               # saved before the setting
    old.mkdir()
    (old / T.TREE_FILE).write_text(json.dumps(d))
    win.open_project(old)
    assert win.tree.options["prompt_format"] == "classic" and win.prompt_format.currentData() == "classic"
    assert win.ref_hint.isEnabled()
    new = tmp_path / "new"
    t = T.Tree()
    T.set_prompt_format(t, "h3")
    T.save(new, t)
    win.open_project(new)
    assert win.prompt_format.currentData() == "h3" and not win.ref_hint.isEnabled()


def test_simple_view_hides_the_expert_controls_also_on_rebuilt_cards(app, win, tmp_path, image):
    """Simple hides the registered controls (form rows with their labels, the Log tab, the card's expert parts),
    keeps what a beginner needs, changes no value; new cards follow the view; Advanced shows everything again, and
    a control the app hides itself (the turbo rows outside turbo mode) stays hidden there."""
    ids = new_project(win, tmp_path, image)
    flush(app, win)
    assert not win.simple and win.act_advanced.isChecked() and win.b_advanced.isChecked()
    steps_before = win.steps.value()
    win.b_simple.click()
    assert win.simple and win.qs.value("viewMode") == "simple" and win.act_simple.isChecked()
    c = win._clips_form
    for f in (win.steps, win.speed, win._cfg_row, win.veda, win.prompt_format, win.style_prefix, win.lora_triggers):
        assert _row_hidden(c, f), f
    assert win.tabs.isTabVisible(win.tabs.indexOf(win.log)) and win.ref_hint.isHidden()   # (the log stays: UX-31)
    for f in (win.frames, win.resolution, win.aspect, win.audio):        # kept
        assert not f.isHidden() and not (c.labelForField(f) and c.labelForField(f).isHidden())
    for w in (win.b_gen, win.b_stop, win.b_play_path, win.b_export, win.start_tile, win.cast_list, win.comfy_label):
        assert not w.isHidden()
    card = win.cards[-1]
    for w in (card.seed, card.b_seed, card.seed_label, card.steps, card.chips.model_chip, card.chips.b_prompt):
        assert w.isHidden(), w
    for f in (card.loras, card.guide, card.keys_row, card.hand_row):
        assert _row_hidden(card.link_form, f)
    for w in (card.edit, card.cam_move, card.b_end, card.length, card.b_variant, card.b_more, card.b_refine,
              card.settings_toggle):
        assert not w.isHidden(), w
    win._rebuild_path()                                                   # new cards: in the current view at once
    new = win.cards[-1]
    assert new is not card and new.seed.isHidden() and _row_hidden(new.link_form, new.loras)
    win.speed.setCurrentIndex(win.speed.findData("turbo"))                # the app shows the turbo rows: Simple wins
    assert _row_hidden(c, win.turbo_lora) and _row_hidden(c, win._turbo_row)
    win.speed.setCurrentIndex(win.speed.findData("normal"))
    win.show_log()                                                        # View → Log: shown when asked for
    assert win.tabs.isTabVisible(win.tabs.indexOf(win.log)) and win.tabs.currentWidget() is win.log
    seen = []
    win.viewChanged.connect(seen.append)
    win.act_advanced.trigger()
    assert seen == [False] and not win.simple and win.qs.value("viewMode") == "advanced"
    new = win.cards[-1]
    assert not new.seed.isHidden() and not _row_hidden(new.link_form, new.loras) and not _row_hidden(c, win.steps)
    assert new.chips.model_chip is not None and not new.chips.model_chip.isHidden()
    assert _row_hidden(c, win.turbo_lora)                                 # not turbo: hidden by the app itself
    assert win.steps.value() == steps_before and win.tabs.isTabVisible(win.tabs.indexOf(win.log))


def test_simple_view_says_when_hidden_settings_are_active(app, win, tmp_path, image):
    """A non-default hidden project setting / clip setting shows "N advanced settings are active: Show" in Simple
    only; Show switches to Advanced."""
    ids = new_project(win, tmp_path, image)
    flush(app, win)
    assert win.advanced_active() == []
    win.steps.setValue(30)
    win.veda.setChecked(False)
    flush(app, win)
    assert win.advanced_active() == ["Steps", "Fast attention (Veda)"]
    assert win.adv_note.isHidden()                                        # Advanced: the controls are in sight
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.advanced_active() == [] and card.adv_note.isHidden()
    card.guide.setCurrentIndex(card.guide.findData(5))
    assert card.advanced_active() == ["Carry-over"]
    win.set_simple(True)
    assert not win.adv_note.isHidden() and win.adv_note.text.text() == "2 advanced settings are active:"
    assert "Steps" in win.adv_note.toolTip()
    card = next(c for c in win.cards if c.nid == ids[1])
    assert not card.adv_note.isHidden() and card.adv_note.text.text() == "1 advanced setting is active:"
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.adv_note.isHidden()                                      # nothing set on it
    win._rebuild_path()                                                   # a rebuilt card: its note too
    card = next(c for c in win.cards if c.nid == ids[1])
    assert not card.adv_note.isHidden()
    card.adv_note.b_show.click()
    assert not win.simple and card.adv_note.isHidden() and win.adv_note.isHidden()
    win.set_simple(True)
    win.steps.setValue(T.DEFAULT_OPTIONS["steps"])
    win.veda.setChecked(True)
    flush(app, win)
    assert win.simple and win.adv_note.isHidden() and win.advanced_active() == []     # back to the defaults


def test_view_default_first_run_simple_existing_settings_advanced(app, tmp_path, monkeypatch):
    from video_chain.ui import ui_logic as UL
    """No settings at all (a first run): Simple; settings from before the views: Advanced. Stored at once, so the
    choice does not flip once the app has written other settings."""
    from PySide6.QtCore import QSettings
    from video_chain.ui import main_window as MW
    for name in ("_poll_comfy", "scan_loras", "_poll_gpu"):              # no ComfyUI, no GPU, no LoRA folders
        monkeypatch.setattr(MW.MainWindow, name, lambda self, *a, **k: None)
    monkeypatch.setattr(UL, "DEFAULT_COMFY_DIR", str(tmp_path / "no-comfyui"))
    q = QSettings(str(tmp_path / "first.ini"), QSettings.IniFormat)
    assert q.allKeys() == []
    w = MW.MainWindow(q, command=FAKE)
    try:
        assert w.simple and q.value("viewMode") == "simple" and w.steps.isHidden()
        w.save_settings()
    finally:
        w.close()
    w = MW.MainWindow(q, command=FAKE)                                    # now with other keys: still Simple
    try:
        assert w.simple
    finally:
        w.close()
    old = settings(tmp_path, "old.ini", view=None)                       # an existing user's settings
    assert old.value("viewMode") is None
    w = MW.MainWindow(old, command=FAKE)
    try:
        assert not w.simple and old.value("viewMode") == "advanced" and not w.steps.isHidden()
    finally:
        w.close()


def test_card_more_menu_triggers_the_same_actions_and_the_card_fits(app):
    """The header: Variant with its name, the rest in More (icon, text, the shortcut shown); each entry emits the
    same action the old buttons did; the header and the footer fit the card's width (size hints, offscreen)."""
    from video_chain.ui.main_window import CARD_WIDTH, PromptCard, VideoCard
    card = PromptCard(T.Node("n3", "n2", "a cat"))
    got = []
    card.action.connect(lambda a, nid: got.append((a, nid)))
    menu = card.b_more.menu()
    entries = [a for a in menu.actions() if not a.isSeparator() and a.isVisible()]
    assert [a.text() for a in entries] == ["Variants…", "Compare…", "Branch", "Freeze", "Delete…"]
    card.set_rows(True, 1)                                      # a clip that starts the video, with other pictures
    assert [a.text() for a in menu.actions() if not a.isSeparator() and a.isVisible()][3:5] == [
        "More start images…", "Apply to its other start images"]
    card.set_rows(False, 0)
    assert all(not a.icon().isNull() and a.toolTip() for a in entries)
    assert card.more_actions["branch"].shortcut().toString() == "Ctrl+B"
    assert card.more_actions["delete"].shortcut().toString() == "Del"
    card.b_variant.click()
    for a in entries:
        a.trigger()
    assert got == [(a, "n3") for a in ("variant", "variants", "compare", "branch", "freeze_toggle", "delete")]
    got.clear()
    card.b_freeze.click()                                       # the old attributes still work
    card.buttons[0].click()
    assert got == [("freeze_toggle", "n3"), ("variant", "n3")]
    card.set_compare(0)
    assert not card.b_compare.isEnabled()
    card.set_state("generating", None)                          # the longest state tag
    m = card.layout().contentsMargins()
    room = CARD_WIDTH - m.left() - m.right()
    head = card.layout().itemAt(0).layout()
    assert head.sizeHint().width() <= room and head.minimumSize().width() <= room
    assert card.b_refine.text() == "Refine" and card.b_variant.text() == "Variant" and card.b_more.text() == "More"
    foot = next(card.layout().itemAt(k).layout() for k in range(card.layout().count())
                if card.layout().itemAt(k).layout() is not None
                and card.layout().itemAt(k).layout().indexOf(card.b_refine) >= 0)
    assert foot.minimumSize().width() <= room
    video = VideoCard(T.Node("n4", "n3", kind=T.VIDEO, source={"file": "/nonexistent.mp4", "start": 0, "end": 1}))
    got = []
    video.action.connect(lambda a, nid: got.append(a))
    for a in video.b_more.menu().actions():
        if not a.isSeparator():
            a.trigger()
    assert got == ["edit", "branch", "freeze_toggle", "delete"]
    head = video.layout().itemAt(0).layout()
    assert head.sizeHint().width() <= room
    card.deleteLater()
    video.deleteLater()


def test_prompt_helpers_insert_undo_and_fit_the_card(app):
    """@ Person or place (none: Cast & references opens), Camera and Examples change the prompt, Ctrl+Z takes each
    back; the foot rows fit the card's width (size hints, offscreen)."""
    from video_chain import camera as CAM
    from video_chain.ui.main_window import CARD_WIDTH, PromptCard
    from video_chain.ui.problems import EXAMPLES
    assert 4 <= len(EXAMPLES) <= 6 and any('"' in e for e in EXAMPLES) and any("#" in e for e in EXAMPLES)
    assert all(CAM.expand(e).count("#") == 0 for e in EXAMPLES)          # the camera example is a real token
    cast = []
    card = PromptCard(T.Node("n1", None, ""), names=lambda: list(cast))
    asked = []
    card.fix.connect(lambda a, nid: asked.append((a, nid)))
    card.b_mention.click()
    assert asked == [("cast_new", "n1")]
    assert "@" in card.edit.placeholderText() and "Camera" in card.edit.placeholderText()
    examples = card.examples_menu.actions()
    examples[0].trigger()
    assert card.edit.toPlainText() == EXAMPLES[0]
    examples[1].trigger()
    assert card.edit.toPlainText() == EXAMPLES[0] + " " + EXAMPLES[1]
    card.edit.undo()
    assert card.edit.toPlainText() == EXAMPLES[0]
    card.edit.setPlainText("A man waves")
    examples[3].trigger()                                                 # a new sentence after the text
    assert card.edit.toPlainText() == "A man waves. " + EXAMPLES[3]
    card.edit.undo()
    card.edit.moveCursor(QTextCursor.End)
    cast.append("Anna")
    assert card._fill_mentions() == ["Anna"]
    card.mention_menu.actions()[0].trigger()
    assert card.edit.toPlainText() == "A man waves @Anna "
    cam = next(a for m in card.camera_menu.actions() if m.menu() for a in m.menu().actions()
               if a.toolTip() == "#push-in-slow")
    assert cam.text() == "Push in, slow"
    cam.trigger()
    assert card.edit.toPlainText() == "A man waves @Anna #push-in-slow "
    assert "the camera pushes in at slow speed" in CAM.expand(card.edit.toPlainText())
    card.edit.undo()
    card.edit.undo()
    assert card.edit.toPlainText() == "A man waves"
    m = card.layout().contentsMargins()
    room = CARD_WIDTH - m.left() - m.right()
    for b in (card.b_refine, card.seed):
        foot = next(card.layout().itemAt(k).layout() for k in range(card.layout().count())
                    if card.layout().itemAt(k).layout() is not None
                    and card.layout().itemAt(k).layout().indexOf(b) >= 0)
        assert foot.sizeHint().width() <= room, (b, foot.sizeHint().width(), room)
    card.deleteLater()


def test_problems_show_on_the_right_card_with_an_action(app, win, tmp_path, image):
    """A cast problem and a sound limit appear on the card of the clip they block, in plain words (the check's own
    text in the tooltip), each with a button that goes to the right place; a made clip has none."""
    ids = new_project(win, tmp_path, image, ("@Anna waves.", "A dog barks."))
    T.add_character(win.project, win.tree, "Anna")
    win._show_references()
    flush(app, win)
    win.focus_clip(ids[0])
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.problems.isVisibleTo(first)
    item = first.problems.items[0]
    assert item["text"].startswith("@Anna has no picture yet") and "clip #1" in item["tip"]
    assert "clip #" not in item["text"]
    win.left_tabs.setCurrentIndex(0)
    first.problems.buttons()["Add a picture"].click()
    assert win.left_tabs.currentIndex() == 1 and win.cast_list.currentItem().data(0x0100) == "Anna"
    second = next(c for c in win.cards if c.nid == ids[1])
    win.focus_clip(ids[1])
    assert not second.problems.isVisibleTo(second)                        # nothing blocks the second clip
    for k in (1, 2, 3):                                                   # three sound references and a track
        assert win.add_sound(str(_make_sound(tmp_path / f"s{k}.wav")))
    tid = win.add_track(str(_make_sound(tmp_path / "song.wav", 20)), 0.0, 20.0, "Song")
    n = win.tree.nodes[ids[1]]
    n.prompt = "A dog barks <Audio 3>."
    n.overrides = {"source_sound": f"track:{tid}"}
    win._rebuild_path()
    win.focus_clip(ids[1])
    second = next(c for c in win.cards if c.nid == ids[1])
    texts = [p["text"] for p in second.problems.items]
    assert any("too many sounds" in t and t.startswith("This clip") for t in texts), texts
    p = next(p for p in second.problems.items if "too many sounds" in p["text"])
    assert p["action"] == "sound" and "Sound" in p["tip"]
    win.fix_problem("sound", ids[1])
    assert win.focus == ids[1]


def test_failed_run_shows_on_the_card_in_simple_and_details_shows_the_log(app, win, tmp_path, image, monkeypatch):
    """Simple: a failed run does not jump to the (hidden) Log; the card of the failed clip says what went wrong,
    Details shows the Log; editing the clip takes the note away."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    win.set_simple(True)
    monkeypatch.setenv("FAKE_FAIL", "1")
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    assert win.status.text().startswith("Failed:") and win.state(ids[1]) == "failed"
    assert win.focus == ids[1]
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.problems.isVisibleTo(card) and card.problems.property("severity") == "error"
    fail = card.problems.items[0]
    assert fail["text"].startswith("Making this clip failed") and "fake failure in clip 2" in fail["tip"]
    for word in ("ref2va", "fl2va", "node", "workflow", "latent", "sampler"):
        assert word not in fail["text"].lower()
    card.problems.buttons()["Details"].click()
    assert win.tabs.isTabVisible(win.tabs.indexOf(win.log)) and win.tabs.currentWidget() is win.log
    type_prompt(win, ids[1], "b again")
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert not card.problems.isVisibleTo(card) and ids[1] not in win._failures

def test_a_clip_seed_can_be_as_large_as_comfyui_takes(app, win, tmp_path, image):
    """ComfyUI's noise takes 64-bit seeds (workflows made elsewhere often have such): the card's field takes them
    too, typed or set, and the project keeps them exactly."""
    from PySide6.QtTest import QTest
    ids = new_project(win, tmp_path, image, ("a",))
    card = win.cards[0]
    big = 18398908909522129001
    assert big > 2**63 and T.SEED_LIMIT == 2**64 - 1
    card.seed.setValue(big)
    flush(app, win)
    assert card.seed.value() == big and win.tree.nodes[ids[0]].seed == big
    assert T.Tree.from_json(json.loads(json.dumps(win.tree.to_json()))).nodes[ids[0]].seed == big
    edit = card.seed.lineEdit()
    edit.selectAll()
    QTest.keyClicks(edit, str(T.SEED_LIMIT))
    flush(app, win)
    assert win.tree.nodes[ids[0]].seed == T.SEED_LIMIT
    card.seed.stepBy(1)                                           # at the end of the range: it stays
    assert card.seed.value() == T.SEED_LIMIT
    edit.selectAll()
    QTest.keyClicks(edit, str(T.SEED_LIMIT + 1) + "x-")           # more than fits, letters, a sign: not taken
    flush(app, win)
    assert 0 <= card.seed.value() <= T.SEED_LIMIT and win.tree.nodes[ids[0]].seed == card.seed.value()
    card.seed.setValue(7)
    card.seed.stepBy(-10)
    assert card.seed.value() == 0
    card.random_seed()                                            # a new random seed stays in the usual range
    assert 1 <= card.seed.value() <= T.SEED_MAX
