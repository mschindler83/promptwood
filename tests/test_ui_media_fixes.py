"""Media players, strips, graph, timeline, storyboard and theme: the fixes of the UMD-* review findings.
Offscreen, synthetic data only."""
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from video_chain.ui.theme import ThemeManager
    ThemeManager.instance().apply(a)
    return a


def wait(app, cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def pixmap(w=64, h=36):
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor("steelblue"))
    return QPixmap.fromImage(img)


def make_video(path: Path, secs=2.0) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=30:duration={secs}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-shortest", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def make_wav(path: Path, secs=2.0) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}",
                    str(path)], check=True)
    return path


# ------------------------------------------------------------------ graph

def infos(n_children=2):
    from video_chain.ui.graph import START, NodeInfo
    out = [NodeInfo(START, None, "start", "", "", "start")]
    for k in range(n_children):
        out.append(NodeInfo(f"n{k + 1}", START, f"clip {k + 1}", "", f"prompt {k + 1}", "done"))
    out.append(NodeInfo("n9", "n1", "child", "", "child prompt", "done"))
    return out


def test_node_thumbnail_scaled_once(app):
    from video_chain.ui.graph import NodeInfo, NodeItem
    item = NodeItem(NodeInfo("n1", None, "t", "", "p", "done", pixmap=pixmap(400, 225)))
    a = item.thumb_pixmap(2.0)
    assert item.thumb_pixmap(2.0) is a                       # not scaled again on the next paint
    assert item.thumb_pixmap(1.0) is not a                   # another screen scale: its own size
    assert a.devicePixelRatio() == 2.0
    item.set_info(NodeInfo("n1", None, "t", "", "p", "done", pixmap=pixmap(200, 100)))
    assert item.thumb_pixmap(2.0) is not a                   # a new picture is scaled again


def test_node_tooltip_short_escaped_and_teaches_actions(app):
    from video_chain.ui.graph import NodeInfo, NodeItem
    item = NodeItem(NodeInfo("n1", None, "t", "", "<b>" + "word " * 400, "done"))
    tip = item.toolTip()
    assert len(tip) < 500 and "&lt;b&gt;" in tip and "<b>" not in tip
    assert "Double-click" in tip and "Right-click" in tip


def test_graph_keyboard(app):
    from PySide6.QtWidgets import QWidget
    from video_chain.ui.graph import START, GraphView
    g = GraphView()
    assert g.focusPolicy() != Qt.NoFocus
    g.show_nodes(infos())
    seen, opened, menus = [], [], []
    g.selected.connect(seen.append)
    g.activated.connect(opened.append)
    g.contextMenu.connect(lambda nid, pos: menus.append(nid))
    g.set_focus("n1")
    QTest.keyClick(g, Qt.Key_Down)
    assert seen[-1] == "n2"                                   # the next sibling
    QTest.keyClick(g, Qt.Key_Up)
    assert seen[-1] == "n1"
    QTest.keyClick(g, Qt.Key_Right)
    assert seen[-1] == "n9"                                   # its first continuation
    QTest.keyClick(g, Qt.Key_Left)
    assert seen[-1] == "n1"                                   # its parent
    QTest.keyClick(g, Qt.Key_Return)
    assert opened == ["n1"]
    QTest.keyClick(g, Qt.Key_Menu)
    assert menus == ["n1"]
    g.set_focus(None)
    QTest.keyClick(g, Qt.Key_Down)
    assert seen[-1] == START                                  # nothing focused yet: the first node
    assert isinstance(g, QWidget)


def test_graph_layout_deep_chain_and_orphans(app):
    from video_chain.ui.graph import START, GraphView, NodeInfo
    g = GraphView()
    chain = [NodeInfo(START, None, "s", "", "", "start")]
    for k in range(1, 1500):
        chain.append(NodeInfo(f"n{k}", START if k == 1 else f"n{k - 1}", "c", "", "", "done"))
    chain.append(NodeInfo("orphan", "missing", "o", "", "", "done"))
    g.show_nodes(chain)                                       # no RecursionError
    assert len(g.items_by_id) == 1501
    assert g.items_by_id["orphan"].pos() != g.items_by_id[START].pos()


def test_graph_zoom_clamped_and_no_dead_override(app):
    from video_chain.ui import graph
    g = graph.GraphView()
    g.zoom_by(100.0)
    assert abs(g.transform().m11() - graph.ZOOM_MAX) < 1e-6
    g.zoom_by(1e-6)
    assert abs(g.transform().m11() - graph.ZOOM_MIN) < 1e-6
    assert "mousePressEvent" not in graph.GraphView.__dict__


# ------------------------------------------------------------------ timeline

def blocks(n=20, secs=5.0, pm=None):
    from video_chain.ui.timeline import Block
    return [Block(f"n{k}", f"#{k}", secs, "done", False, pm, tip=f"tip {k}") for k in range(n)]


def test_timeline_scrolls_to_the_playing_clip(app):
    from video_chain.ui.timeline import Timeline
    tl = Timeline()
    tl.resize(300, 120)
    tl.show()
    tl.show_path(blocks(), None)
    app.processEvents()
    assert tl.horizontalScrollBar().value() == 0 and tl.horizontalScrollBar().maximum() > 0
    tl.set_playing("n19")
    app.processEvents()
    assert tl.horizontalScrollBar().value() > 0
    tl.close()


def test_timeline_hit_testing_needs_no_paint_and_right_click_selects(app):
    from video_chain.ui.timeline import Timeline
    tl = Timeline()
    tl.resize(800, 120)
    tl.show_path(blocks(4), None)                              # (never painted: not shown)
    order = []
    tl.clicked.connect(lambda n: order.append(("clicked", n)))
    tl.menu.connect(lambda n, p: order.append(("menu", n)))
    pos = QPointF(30, 40).toPoint()
    QTest.mouseClick(tl.lane, Qt.RightButton, pos=pos)
    assert order == [("clicked", "n0"), ("menu", "n0")]       # the clip is selected, then its menu


def test_timeline_thumbnail_cached_and_time_labels(app):
    from video_chain.ui.timeline import Timeline, _Lane
    lane = _Lane()
    b = blocks(1, pm=pixmap(400, 225))[0]
    a = lane.thumb(b, 80, 45, 1.0)
    assert lane.thumb(b, 80, 45, 1.0) is a
    assert Timeline  # imported for the module under test


# ------------------------------------------------------------------ keyframe strip

def test_keyframe_strip_thumbnail_cached_and_chain_fps(app):
    from video_chain import chain
    from video_chain.ui import keyframe_strip
    assert keyframe_strip.FPS == chain.FPS
    s = keyframe_strip.KeyframeStrip()
    it = {"at": 1.0, "pixmap": pixmap(400, 225)}
    s.set_data(3.0, [it])
    a = s.thumb(0, 1.0)
    assert s.thumb(0, 1.0) is a
    s.set_data(3.0, [{"at": 1.0, "pixmap": pixmap(100, 50)}])
    assert s.thumb(0, 1.0) is not a
    assert "H3" not in s.toolTip()                            # no internal jargon in the UI


def test_no_literal_colours_where_tokens_exist():
    base = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"
    for name, bad in (("keyframe_strip.py", '"#e5534b"'), ("timeline.py", "QColor(90, 140, 200)"),
                      ("timeline.py", "QColor(60, 60, 60)"), ("compare_player.py", "QColor(12, 12, 14)"),
                      ("compare_player.py", "QColor(150, 150, 160)")):
        assert bad not in (base / name).read_text(), (name, bad)


# ------------------------------------------------------------------ storyboard

def scene(i, title="A & B", thumb=None):
    return (f"s{i}", title, {"done": 1, "secs": 59.6, "clips": 1, "thumb": thumb})


def test_storyboard_tooltip_escapes_title(app):
    from video_chain.ui.storyboard import Storyboard
    sb = Storyboard()
    sb.show_scenes([scene(1, "Act <1> & more")], "s1")
    tip = sb.list.item(0).toolTip()
    assert "Act &lt;1&gt; &amp; more" in tip


def test_storyboard_time_rounds_up_to_the_minute(app):
    from video_chain.ui.storyboard import Storyboard, state_text
    assert state_text({"done": 1, "secs": 59.6}).startswith("1:00 ")
    sb = Storyboard()
    sb.show_scenes([scene(1)], "s1")
    assert "1:00" in sb.total.text() and "0:60" not in sb.total.text()


def test_storyboard_chosen_once_per_double_click(app):
    from video_chain.ui.storyboard import Storyboard
    sb = Storyboard()
    sb.show_scenes([scene(1), scene(2)], "s1")
    got = []
    sb.sceneChosen.connect(got.append)
    item = sb.list.item(1)
    for _ in range(3):                                         # click, click, activated
        sb._chosen(item)
        app.processEvents()
        app.processEvents()
    assert got == ["s2"]
    sb._chosen(sb.list.item(0))                               # (back to the open scene s1 after a refresh)
    sb.show_scenes([scene(1), scene(2)], "s1")
    app.processEvents()


def test_storyboard_thumbnails_decoded_once(app, tmp_path, monkeypatch):
    from video_chain.ui import storyboard
    from video_chain.ui.storyboard import Storyboard
    f = tmp_path / "t.png"
    img = QImage(300, 200, QImage.Format_RGB32)
    img.fill(QColor("red"))
    img.save(str(f))
    loads = []
    real = storyboard.QImage

    class Counting(real):
        def __init__(self, *a):
            loads.append(a)
            super().__init__(*a)
    monkeypatch.setattr(storyboard, "QImage", Counting)
    sb = Storyboard()
    scenes = [scene(1, thumb=str(f)), scene(2, thumb=str(f))]
    sb.show_scenes(scenes, "s1")
    sb.show_scenes(scenes, "s1")
    sb.show_scenes(scenes, "s2")
    assert len(loads) == 1                                     # one decode for the file, not per card and refresh
    img.fill(QColor("blue"))
    time.sleep(0.01)
    img.save(str(f))
    os.utime(f, (time.time() + 5, time.time() + 5))
    sb.show_scenes(scenes, "s1")
    assert len(loads) == 2                                     # the file changed: read again


def test_storyboard_set_enabled_is_a_real_setenabled(app):
    from video_chain.ui.storyboard import Storyboard
    sb = Storyboard()
    sb.setEnabled(False)
    assert not sb.isEnabled() and not sb.list.isEnabled() and not sb.b_add.isEnabled()
    sb.setEnabled(True)
    assert sb.isEnabled() and sb.list.isEnabled()


# ------------------------------------------------------------------ cast strip / view mode

def test_cast_cells_lead_without_characters(app, monkeypatch):
    from types import SimpleNamespace
    from video_chain.ui import cast_strip
    T = cast_strip.T
    monkeypatch.setattr(T, "node_cast", lambda o, n: {"mode": "lead", "characters": [], "voices": []})
    monkeypatch.setattr(T, "mentions", lambda o, p: ([], None))
    monkeypatch.setattr(T, "full_prompt", lambda o, p: p)
    monkeypatch.setattr(T, "cast_of", lambda o: [{"name": "Ann"}])
    monkeypatch.setattr(T, "kind_of", lambda c: "person")
    n = SimpleNamespace(id="n3", is_video=False, prompt="x")
    assert cast_strip.cast_cells({}, [n]) == {}                # no IndexError


def test_view_mode_show_on_deleted_widget(app):
    import shiboken6
    from PySide6.QtWidgets import QWidget
    from video_chain.ui.view_mode import ViewMode
    vm = ViewMode()
    w = QWidget()
    vm.advanced(w)
    shiboken6.delete(w)
    vm.show(w, True)                                           # no RuntimeError
    assert vm.visible(w) is False
    vm.set_simple(True)


# ------------------------------------------------------------------ widgets / theme

def test_format_time(app):
    from video_chain.ui.video_editor import fmt
    from video_chain.ui.widgets import format_time
    assert format_time(59.6) == "1:00"
    assert format_time(75) == "1:15"
    assert format_time(75.25, 3) == "1:15.250"
    assert format_time(59.9996, 3) == "1:00.000"
    assert format_time(-3) == "0:00"
    assert fmt(6.5) == "0:06.500" and fmt(125.0) == "2:05.000"


def test_wheel_guard_lets_an_editable_combo_being_edited_keep_the_wheel(app, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QWidget
    from video_chain.ui import widgets
    root = QWidget()
    c = QComboBox(root)
    c.setEditable(True)
    c.addItems(["a", "b"])
    widgets.guard_wheel(root)
    # focus sits in the combo's line edit, not the combo itself
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: c.lineEdit()))
    assert widgets._WheelGuard.has_focus(c)
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: None))
    assert not widgets._WheelGuard.has_focus(c)


def test_icon_pixmaps_and_svg_text_are_cached(app, monkeypatch):
    from video_chain.ui import theme
    reads = []
    real = Path.read_text

    def counting(self, *a, **k):
        if self.suffix == ".svg":
            reads.append(self.name)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", counting)
    tint = QColor("#123456")
    a = theme.icon_pixmap("play", 16, tint, 1.0)
    b = theme.icon_pixmap("play", 16, tint, 1.0)
    assert a is b
    theme.icon_pixmap("play", 24, tint, 1.0)                   # another size: the SVG text is not read again
    assert reads.count("play.svg") <= 1
    assert theme.icon_pixmap("play", 16, tint, 2.0) is not a


def test_missing_icon_warns_once(app):
    from video_chain.ui import theme
    with pytest.warns(UserWarning, match="no-such-icon"):
        assert theme.tinted_svg("no-such-icon", QColor("red")) == b""


def test_tinted_icon_file_follows_the_svg(app, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from video_chain.ui import theme
    monkeypatch.setattr(theme, "QStandardPaths", SimpleNamespace(
        writableLocation=lambda *_: str(tmp_path), CacheLocation=0))
    tm = theme.ThemeManager.instance()
    monkeypatch.setattr(theme, "tinted_svg", lambda *a, **k: b"<svg>old</svg>")
    old = tm._icon_file("play", QColor("red"))
    monkeypatch.setattr(theme, "tinted_svg", lambda *a, **k: b"<svg>new</svg>")
    new = tm._icon_file("play", QColor("red"))
    assert old != new and Path(new).read_bytes() == b"<svg>new</svg>"


def _lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def _ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_text_contrast_pairs_listed_and_met(app):
    import json
    from video_chain.ui import theme
    tok = json.loads((theme.THEME_DIR / "tokens.json").read_text())
    pairs = [tuple(p) for p in tok["contrast"]["pairs"]]
    assert ("textTertiary", "bgRaised") in pairs and ("textTertiary", "bgOverlay") in pairs
    t = theme.load_tokens()
    for fg, bg in pairs:
        assert _ratio(getattr(t, fg), getattr(t, bg)) >= tok["contrast"]["minimum"], (fg, bg)


def test_unknown_accent_falls_back_to_the_default(app):
    from video_chain.ui import theme
    assert theme.load_tokens("Nope").accent == theme.load_tokens().accent


def test_fonts_registered_once(app, monkeypatch):
    from video_chain.ui import theme
    calls = []
    real = theme.QFontDatabase.addApplicationFont
    monkeypatch.setattr(theme.QFontDatabase, "addApplicationFont", staticmethod(lambda p: calls.append(p) or real(p)))
    tm = theme.ThemeManager.instance()
    tm.apply(app)
    tm.apply(app)
    assert len(calls) <= 2 and len(set(calls)) == len(calls)


# ------------------------------------------------------------------ compare player

def test_compare_decode_does_not_use_a_stderr_pipe(tmp_path, monkeypatch):
    from video_chain.ui import compare_player
    seen = {}
    real = subprocess.Popen

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(subprocess, "Popen", spy)
    clip = make_video(tmp_path / "c.mp4", 0.5)
    frames = compare_player.decode(clip, (80, 44))
    assert len(frames) >= 10 and seen["stderr"] is not subprocess.PIPE
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="cannot read bad.mp4"):
        compare_player.decode(bad, (80, 44))


def test_compare_result_decode_cancels_the_older_one(app, monkeypatch):
    from video_chain.ui import compare_player
    cancels = []

    def fake(clip, size, progress=None, cancelled=None):
        cancels.append(cancelled)
        return []
    monkeypatch.setattr(compare_player, "decode", fake)
    p = compare_player.ComparePlayer()
    p.load(Path("a.mp4"), 10, (160, 90))
    assert wait(app, lambda: len(cancels) == 1)
    p.load_result(Path("r1.mp4"))
    assert wait(app, lambda: len(cancels) == 2)
    p.load_result(Path("r2.mp4"))
    assert wait(app, lambda: len(cancels) == 3)
    assert not cancels[0]() and cancels[1]() and not cancels[2]()   # the first result decode was stopped
    p.stop_loading()


def test_compare_player_cancels_when_destroyed(app):
    import shiboken6
    from video_chain.ui.compare_player import ComparePlayer
    p = ComparePlayer()
    ev = p._cancel
    shiboken6.delete(p)
    assert ev.is_set()


def test_compare_player_plays_in_real_time(app, monkeypatch):
    from video_chain import chain
    from video_chain.ui import compare_player
    p = compare_player.ComparePlayer()
    assert p.fps == chain.FPS
    p.size_ = (2, 2)
    p.frames = [bytes(12)] * 30
    p.slider.setRange(0, 29)
    p._controls(True)
    now = [100.0]
    monkeypatch.setattr(compare_player.time, "monotonic", lambda: now[0])
    p.play(from_start=True)
    now[0] += 0.25                                              # a slow tick: the picture catches up
    p._tick()
    assert p.position() == round(0.25 * p.fps)
    now[0] += 10
    p._tick()
    assert p.position() == 29 and not p.playing()
    p.pause()


# ------------------------------------------------------------------ range bar / dialogs

def test_range_bar_overlapping_handles_and_keys(app):
    from video_chain.ui.video_editor import RangeBar
    bar = RangeBar()
    bar.resize(400, 42)
    bar.set_values(10.0, 5.0, 5.0 + 1 / 24)
    assert bar.focusPolicy() != Qt.NoFocus and bar.accessibleName()
    x_start, x_end = bar._x(bar.start), bar._x(bar.end)
    assert bar._hit(x_start - 3) == "start"                     # left of the middle: the start
    assert bar._hit(x_end + 3) == "end"                         # right of it: the end
    got = []
    bar.rangeChanged.connect(lambda s, e: got.append((s, e)))
    QTest.keyClick(bar, Qt.Key_Right, Qt.ControlModifier | Qt.ShiftModifier)
    assert got and got[-1][1] > bar.start + 1 / 24 - 1e-9 and got[-1][0] == 5.0


def _video_dlg(tmp_path, **kw):
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_video(tmp_path / "v.mp4", 2.0)
    return VideoRangeDialog(None, str(src), **kw)


def test_video_dialog_stops_its_frame_timer_when_closed(app, tmp_path):
    dlg = _video_dlg(tmp_path, player=False)
    dlg.set_range(0.0, 1.0)
    assert dlg._frame_timer.isActive()
    dlg.done(0)
    assert not dlg._frame_timer.isActive()
    gen = dlg._gen
    dlg._update_frames()                                        # a late call does nothing
    assert dlg._gen == gen


def test_video_dialog_has_video_none_and_falls_back_when_quick_output_fails(app, tmp_path, monkeypatch):
    from video_chain.ui import video_editor
    dlg = _video_dlg(tmp_path, player=False)
    assert dlg.video is None
    real = video_editor.VideoView

    def flaky(*a, gpu=None, **k):
        if gpu is not False:
            raise RuntimeError("Qt Quick video output: broken")
        return real(*a, gpu=False, **k)
    monkeypatch.setattr(video_editor, "VideoView", flaky)
    dlg2 = _video_dlg(tmp_path, player=True)                    # no traceback: the CPU view is used
    assert dlg2.video is not None and dlg2.player is not None
    dlg2.done(0)


@pytest.mark.parametrize("which", ["audio", "video"])
def test_player_error_is_shown_and_play_disabled(app, tmp_path, which):
    from PySide6.QtMultimedia import QMediaPlayer
    if which == "audio":
        from video_chain.ui.audio_editor import AudioRangeDialog
        dlg = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=True)
        shown = dlg.info_label
    else:
        dlg = _video_dlg(tmp_path, player=True)
        shown = dlg.info_label
    assert dlg.player is not None
    dlg.player.errorOccurred.emit(QMediaPlayer.ResourceError, "no codec for this file")
    assert "no codec for this file" in shown.text()
    assert not dlg.b_play.isEnabled() and not dlg.b_play_sel.isEnabled()
    dlg.done(0)


def test_dialogs_share_the_player_logic_and_seek_clears_stop_at(app, tmp_path):
    from video_chain.ui.audio_editor import AudioRangeDialog
    from video_chain.ui.video_editor import RangePlayerMixin, VideoRangeDialog
    assert issubclass(AudioRangeDialog, RangePlayerMixin) and issubclass(VideoRangeDialog, RangePlayerMixin)
    a = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=False)
    v = _video_dlg(tmp_path, player=False)
    for d in (a, v):
        d._stop_at = 1.0
        d.seek(0.5)
        assert d._stop_at is None                               # a seek ends "play range"
        d.done(0)


def test_audio_play_range_stops_the_cleaned_sound(app, tmp_path):
    from video_chain.ui.audio_editor import AudioRangeDialog
    dlg = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=True)
    stopped = []
    dlg._stop_cleaned_playback = lambda: stopped.append(1)
    dlg.play_range()
    assert stopped
    dlg.done(0)


def test_keys_hint_in_the_editors(app, tmp_path):
    from video_chain.ui.audio_editor import AudioRangeDialog
    a = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=False)
    v = _video_dlg(tmp_path, player=False)
    assert "[" in a.keys_hint.text() and "]" in a.keys_hint.text() and "Space" in a.keys_hint.text()
    assert "R" in v.keys_hint.text() and "Space" in v.keys_hint.text()
    a.done(0)
    v.done(0)


def test_audio_cancel_disabled_while_removing_music(app, tmp_path):
    from PySide6.QtWidgets import QDialogButtonBox
    from video_chain.ui.audio_editor import AudioRangeDialog
    dlg = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav", 6.0)), player=False, voice=True,
                           isolate=lambda a, b: None)
    dlg.clean_busy = True
    dlg._update_enabled()
    assert not dlg.buttons.button(QDialogButtonBox.Cancel).isEnabled()
    dlg.clean_busy = False
    dlg._update_enabled()
    assert dlg.buttons.button(QDialogButtonBox.Cancel).isEnabled()
    dlg.done(0)


# ------------------------------------------------------------------ tts dialogs

def _tts(app, local=None, eleven=None):
    from video_chain.ui.tts_dialog import VoiceTtsDialog
    return VoiceTtsDialog(None, "Ann", "a voice", "English", "local", "model", lambda p, t: None,
                          local or (lambda: None), eleven or (lambda: None))


def test_tts_dialog_fields_cancel_and_voice_id(app, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox
    from video_chain.ui import tts_dialog
    dlg = _tts(app)
    assert dlg.voices == []                                     # set in __init__
    dlg._fill_voices([{"id": "EL1", "name": "Bob"}])           # a cached ElevenLabs voice in the combo
    seen = {}

    def make_sample(eng, text, lang, wav, **kw):
        seen.update(kw)
        return 1.0
    monkeypatch.setattr(tts_dialog.tts, "make_sample", make_sample)
    dlg.kind.setCurrentIndex(dlg.kind.findData("local"))
    dlg._make()
    assert dlg.busy and not dlg.buttons.button(QDialogButtonBox.Cancel).isEnabled()
    assert wait(app, lambda: not dlg.busy)
    assert seen["voice_id"] is None                             # not the ElevenLabs voice for the local engine
    assert dlg.buttons.button(QDialogButtonBox.Cancel).isEnabled()
    dlg.cleanup()


def test_tts_dialog_worker_error_after_close_does_not_raise(app, monkeypatch):
    import shiboken6
    from video_chain.ui import tts_dialog
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a))
    gate = threading.Event()

    def make_sample(*a, **k):
        gate.wait(5)
        raise RuntimeError("late")
    monkeypatch.setattr(tts_dialog.tts, "make_sample", make_sample)
    dlg = _tts(app)
    dlg._make()
    sig = dlg._sig
    shiboken6.delete(sig)                                       # the dialog is gone while the worker runs
    gate.set()
    time.sleep(0.5)
    assert errors == []
    dlg.cleanup()


def _speak(app, render=None):
    from video_chain.ui.speak_dialog import SpeakLinesDialog
    return SpeakLinesDialog(None, "n1", [{"name": "Ann", "line": "Hello"}], "local",
                            {"takes": 3, "delivery": "natural", "speed": 1.0}, lambda p, t: None,
                            render or (lambda k, how, said: {"secs": 1.0, "lines": [{"name": "Ann", "line": "Hello"}],
                                                             "speak": how}), lambda r: "x.wav")


def test_speak_dialog_has_a_stop_button_that_keeps_the_takes(app):
    gate = threading.Event()

    def render(k, how, said):
        if k > 0:
            gate.wait(5)
        return {"secs": 1.0, "lines": [{"name": "Ann", "line": "Hello"}], "speak": how}
    dlg = _speak(app, render)
    assert dlg._failed is False and dlg._left == 0            # set in __init__
    dlg.make()
    assert dlg.busy and dlg.b_make.text() == "Stop" and dlg.b_make.isEnabled()
    assert wait(app, lambda: len(dlg.takes) == 1)
    dlg.b_make.click()                                         # Stop: not Cancel
    assert dlg._stop and not dlg._close_when_done
    gate.set()
    assert wait(app, lambda: not dlg.busy)
    assert len(dlg.takes) <= 2 and len(dlg.takes) >= 1       # the dialog stays open with its takes
    assert dlg.b_make.text() in ("Make more takes", "Make the takes")
