"""GUI tests: real videos and video clips: the range editor, rotation, hand over, compare, split, end images, keyframes.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import os
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402

from video_chain import chain  # noqa: E402
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    _chips, _row_hidden, app, crop_accepted, flush, image, make_real_video, new_project, run_and_wait,
    silent_player, type_prompt, wait, wait_for_joined_path, win)


def test_video_first_then_generated_clip(app, win, tmp_path):
    from video_chain.ui.main_window import PromptCard, VideoCard
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.5)
    assert v and win.live.get(v) == "converting" and not win.b_gen.isEnabled()
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.tree.nodes[v].made["frames"] == 36
    assert isinstance(win.cards[0], VideoCard) and "real.mp4" in win.cards[0].file.text()
    assert "like the first video" in win.size_info.text()
    assert win.graph.items_by_id[v].info.kind == "video"
    # no start image: the reference button takes the selected video's first frame
    assert not win.b_ref_start.isEnabled() and "no start image" in win.b_ref_start.toolTip()
    win.select(v)
    assert win.b_ref_clip.isEnabled() and f"first frame of video #{v[1:]}" in win.b_ref_clip.toolTip()
    win.b_ref_clip.click()
    win.reference_from_node(v, "last")
    assert win.refs.count() == 2 and win.char_lead.isChecked()                 # both: faces of the lead
    assert "last frame of" in win.status.text()
    win.char_lead.setChecked(False)                                            # clips without a face again
    win.node_action("continue", v)
    a = win.selected_id()
    assert isinstance(win.cards[1], PromptCard)
    type_prompt(win, a, "after the real video")
    run_and_wait(app, win, win.generate_path)
    assert win.state(a) == "done", win.status.text()
    assert chain.probe(wait_for_joined_path(app, win, a))["frames"] == 36 + 16
    # a later start, same last frame: converted again, but the clip after it stays (it starts from that frame)
    win.set_video_range(v, str(src), 0.5, 1.5)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.state(a) == "done"
    assert win.tree.nodes[v].made["frames"] == 24
    # another last frame (still red): the clip after it must be made again
    win.set_video_range(v, str(src), 0.0, 0.8)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.state(a) == "changed"


def test_clip_into_video_and_variants(app, win, tmp_path, image):
    src = make_real_video(tmp_path / "real.mp4")
    ids = new_project(win, tmp_path, image, ("walks toward the door",))
    a = ids[0]
    v = win.add_video(a, str(src), 1.0, 2.0)
    wait(app, lambda: not win._converts)
    assert win.add_video(a, str(src), 0.0, 1.0) is None                 # one video per generated clip
    assert "already leads into" in win.status.text()
    assert "ends at video" in win.cards[0].span.text()
    run_and_wait(app, win, win.generate_path)                             # path start -> a -> v
    assert win.state(a) == "done" and win.state(v) == "done", win.status.text()
    assert win.tree.nodes[a].made["end_key"] == win.tree.nodes[v].made["key"]
    assert chain.probe(wait_for_joined_path(app, win, v))["frames"] == 16 + 24              # the clip's repeated last frame is cut
    new = win.create_variants(a, 2, generate=False)
    for nid in new:
        vc = win.tree.video_child(nid)
        assert vc is not None and vc.id != v and vc.source == win.tree.nodes[v].source
        assert win.state(vc.id) == "done" and vc.made["video"] != win.tree.nodes[v].made["video"]
    win.node_action("variant", v)                                         # not for real videos
    assert "Variants are for generated clips" in win.status.text()


def test_end_image_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert card.end_text.text() == "free" and card.b_end.isEnabled() and not card.b_end_clear.isEnabled()
    end = tmp_path / "end.png"
    Image.new("RGB", (40, 30), (255, 255, 0)).save(end)
    assert win.set_end_image(ids[0], str(end))
    card = win.cards[0]
    assert card.end_text.text() == "own end image" and card.b_end_clear.isEnabled()
    assert "end image" in win.badge(win.tree.nodes[ids[0]])
    run_and_wait(app, win, win.generate_path)
    assert win.state(ids[0]) == "done", win.status.text()
    new = win.create_variants(ids[0], 1, generate=False)                    # a variant gets its own copy
    assert win.tree.nodes[new[0]].end_image not in (None, win.tree.nodes[ids[0]].end_image)
    win.node_action("end_clear", ids[0])
    flush(app, win)
    assert win.tree.nodes[ids[0]].end_image is None and win.state(ids[0]) == "changed"
    assert win.cards[0].end_text.text() == "free"


def test_motion_guide_choice_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("x", "y"))
    card = win.cards[1]
    card.guide.setCurrentIndex(card.guide.findData(22))
    assert win.tree.nodes[ids[1]].overrides["guide"] == 22
    assert "carry-over 22" in win.badge(win.tree.nodes[ids[1]])
    flush(app, win)
    assert win.cards[1].span.text().startswith("0:03.0 – 0:05.2")          # 22 guide frames left out


def test_video_range_dialog(app, tmp_path):
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_real_video(tmp_path / "real.mp4", secs=2.0)
    from PySide6.QtGui import QImage
    got = []
    dlg = VideoRangeDialog(None, str(src), canvas=(672, 384), player=False,
                           on_reference=lambda p: got.append(QImage(p).size().toTuple()) or True)
    assert dlg.b_ref.isEnabled() and dlg.reference_here()
    assert got == [(dlg.info["width"], dlg.info["height"])]  # the whole frame, not cut to the chain's size
    assert "was added (see the Cast" in dlg.summary.text()
    assert dlg.info["fps"] == 30 and abs(dlg.end - 2.0) < 0.05 and dlg.frames() == 48
    dlg.set_range(0.5, 1.0)
    assert dlg.frames() == 12 and "12 frames" in dlg.summary.text() and "672×384" in dlg.summary.text()
    dlg.set_range(1.0, 0.2)                                   # never shorter than one frame
    assert dlg.end - dlg.start >= 1 / 24 - 1e-6
    dlg.end_spin.setValue(0.1)                                # moving the end below the start moves the start
    assert dlg.end == 0.1 and dlg.start < 0.1
    dlg.seek(1.5)
    dlg.step_frames(3)
    assert abs(dlg.position() - 1.6) < 1e-6
    dlg.set_range(0.0, 1.0)
    dlg._update_frames()
    wait(app, lambda: dlg.first_tile.has_image() and dlg.last_tile.has_image(), timeout=30)
    from PySide6.QtWidgets import QDialogButtonBox
    assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.close()


def test_video_range_dialog_rotation(app, tmp_path):
    """A quarter turn in the range window: the size it reports, the chain's size worked out again, the face
    picture and the first / last frame are turned; R and Shift+R turn it back."""
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_real_video(tmp_path / "real.mp4", secs=2.0)              # 320x180
    asked, got = [], []

    def canvas_for(file, rot):
        asked.append(rot)
        w, h = chain.rotated_size(320, 180, rot)
        return chain.canvas_size(w, h, 512)
    dlg = VideoRangeDialog(None, str(src), canvas=(672, 384), player=False, canvas_for=canvas_for,
                           on_reference=lambda p: got.append(QImage(p).size().toTuple()) or True)
    from PySide6.QtGui import QImage
    assert dlg.rotate == 0 and dlg.b_rot_right.isEnabled() and dlg.info_label.text().startswith("320×180 ·")
    dlg.b_rot_right.click()
    assert dlg.rotate == 90 and asked[-1] == 90 and dlg.canvas == (384, 672)
    assert dlg.info_label.text().startswith("180×320 · turned 90° right")
    assert "turned 90° right, cut to 384×672" in dlg.summary.text()
    assert dlg.reference_here() and got == [(180, 320)]
    dlg._update_frames()
    wait(app, lambda: dlg.first_tile.has_image() and dlg.last_tile.has_image(), timeout=30)
    dlg.b_rot_left.click()
    dlg.b_rot_left.click()
    assert dlg.rotate == 270 and dlg.canvas == (384, 672)
    dlg.set_rotation(360)
    assert dlg.rotate == 0 and dlg.canvas == (672, 384) and "turned" not in dlg.summary.text()
    dlg.close()
    again = VideoRangeDialog(None, str(src), 0.0, 1.0, canvas=(384, 672), player=False, rotate=180)
    assert again.rotate == 180 and "turned 180° right" in again.info_label.text()
    again.close()


def test_video_view_rotation(app, tmp_path):
    """The preview turns on top of what the player gives; a phone video's own rotation flag is applied as ffmpeg
    applies it (red left / blue right, flagged 90° counter-clockwise: blue on top)."""
    import subprocess

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtMultimedia import QMediaPlayer, QtVideo, QVideoFrame, QVideoSink

    from video_chain.ui.widgets import VideoView, frame_image
    cpu = VideoView(gpu=False, turnable=True)
    cpu._raw = QImage(320, 180, QImage.Format_RGB888)
    cpu.set_rotation(90)
    assert cpu.image().size().toTuple() == (180, 320)
    cpu.set_rotation(180)
    assert cpu.image().size().toTuple() == (320, 180)
    with pytest.raises(RuntimeError):
        VideoView(gpu=False).set_rotation(90)
    plain = VideoView(gpu=True)                                            # the other players: as they were
    assert plain.sink is plain.widget.videoSink()
    gpu = VideoView(gpu=True, turnable=True)          # Qt Quick's VideoOutput, turned by its orientation
    assert gpu._out.property("videoSink") is not None and gpu.sink is gpu._out.property("videoSink")
    img = QImage(32, 18, QImage.Format_RGB32)
    img.fill(0)
    gpu.sink.setVideoFrame(QVideoFrame(img))
    for deg in (90, 270, 0):
        gpu.set_rotation(deg)
        assert gpu._out.property("orientation") == deg
    assert gpu.sink.videoFrame().rotation() == QtVideo.Rotation.None_     # (the frames themselves are left alone)
    gpu.clear()
    assert not gpu.sink.videoFrame().isValid()
    split, phone = tmp_path / "split.mp4", tmp_path / "phone.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=32x36:r=30:d=0.5", "-f", "lavfi",
                    "-i", "color=blue:s=32x36:r=30:d=0.5", "-filter_complex", "[0:v][1:v]hstack", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(split)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(split), "-c", "copy",
                    str(phone)], check=True)
    sink, player, got = QVideoSink(), QMediaPlayer(), []
    player.setVideoOutput(sink)
    sink.videoFrameChanged.connect(lambda f: f.isValid() and not got and got.append(QVideoFrame(f)))
    player.setSource(QUrl.fromLocalFile(str(phone)))
    player.pause()                                  # (the first frame, without playing: the tests never play)
    wait(app, lambda: got, timeout=15)
    player.stop()
    img = frame_image(got[0])
    assert img.size().toTuple() == (36, 64) == got[0].toImage().size().toTuple()
    assert img.pixelColor(18, 3).blue() > 200 and img.pixelColor(18, 60).red() > 200


def test_rotated_real_video_in_window(app, win, tmp_path):
    """A video added turned: the clips follow its portrait size; turning it back converts it again, unturned."""
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.0, rotate=90)
    wait(app, lambda: not win._converts)
    node = win.tree.nodes[v]
    assert win.state(v) == "done" and node.source["rotate"] == 90 and node.made["key"]["rotate"] == 90
    info = chain.probe(win.project / node.made["video"])
    assert info["height"] > info["width"]
    assert "turned 90° right" in win.cards[0].range.text()
    assert win._canvas_for(str(src), 0, v)[0] > win._canvas_for(str(src), 0, v)[1]   # as if it were not turned
    win.set_video_range(v, str(src), 0.0, 1.0)                 # (no rotation given: kept)
    assert win.state(v) == "done" and win.tree.nodes[v].source["rotate"] == 90
    win.set_video_range(v, str(src), 0.0, 1.0, 0)
    wait(app, lambda: not win._converts)
    node = win.tree.nodes[v]
    assert "rotate" not in node.source and win.state(v) == "done"
    info = chain.probe(win.project / node.made["video"])
    assert info["width"] > info["height"]


def test_hand_over_frame_in_the_window(app, win, tmp_path, image):
    from video_chain.ui.clip_dialogs import HandoverDialog
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert not card.b_hand.isEnabled()                                  # not made yet
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    a = win.tree.nodes[ids[0]]
    dlg = HandoverDialog(None, win.project / a.made["video"], "x", None)
    wait(app, lambda: dlg.frames, timeout=20)
    assert len(dlg.frames) == 17 and dlg.frame() is None               # starts at the last frame
    dlg.slider.setValue(9)
    assert dlg.frame() == 9 and "7 frames" in dlg.summary.text()
    dlg.reject()
    assert win.cards[0].b_hand.isEnabled()
    wait(app, lambda: not win._joins, timeout=30)                       # the path played after generating
    assert win.set_handover(ids[0], 9)
    assert a.end_at == 9 and [win.state(i) for i in ids] == ["done", "changed"]
    assert "1 made clip is now CHANGED" in win.status.text()
    assert win.cards[0].hand_text.text() == "frame 10 of 17 (0.38 s)" and "hands over at 0.4 s" in win.badge(a)
    assert win._node_frame(a).name.endswith("_f9.png")
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "done"]
    wait(app, lambda: not win._joins, timeout=30)
    win.node_action("handover_clear", ids[0])                           # back to the last frame
    assert a.end_at is None and win.state(ids[1]) == "changed"


def test_compare_variants_and_keep_one(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    new = win.create_variants(ids[1], 2, generate=True)
    wait(app, lambda: not win.running)
    run_and_wait(app, win, win.generate_all)
    flush(app, win)
    wait(app, lambda: not win._joins, timeout=30)
    assert win.compare_candidates(ids[1]) == [ids[1]] + new
    win.select(ids[1])
    assert win.cards[1].b_compare.isEnabled() and not win.cards[0].b_compare.isEnabled()
    dlg = win.compare_variants(ids[1], player=False)
    assert len(dlg.keep_buttons) == 3 and "seed" in dlg.items[1][2]
    dlg.keep_buttons[new[0]].click()                                    # keep the first variant
    assert dlg.kept == new[0] and set(win.tree.nodes) == {ids[0], new[0]}
    assert win.selected_id() == new[0]
    assert win.compare_variants(new[0], player=False) is None and "Nothing to compare" in win.status.text()


def test_compare_player_frames_and_divider(app, tmp_path):
    """The before / after player: every frame of both clips, exact; the divider follows the mouse."""
    import subprocess

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from video_chain.ui.compare_player import ComparePlayer, store_size
    assert store_size(832, 480, 107) == (832, 480)                       # small clips at their own size
    w, h = store_size(1920, 1080, 2000)
    assert w <= 1280 and 2 * 2000 * w * h * 3 <= 900 * 2**20 and abs(w / h - 16 / 9) < 0.02
    clips = []
    for name, colour in (("before", "red"), ("after", "blue")):
        f = tmp_path / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color={colour}:s=64x36:r=24:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f)], check=True)
        clips.append(f)
    p = ComparePlayer()
    p.resize(640, 420)
    p.load(clips[0], 24, (64, 36))
    wait(app, lambda: p.count() == 24, timeout=20)
    assert p.b_play.isEnabled() and p.view.after is None
    p.load_result(clips[1])
    wait(app, lambda: p.result is not None, timeout=20)
    assert len(p.result) == 24
    before, after = p.image(5), p.image(5, True)
    assert before.pixelColor(32, 18).red() > 200 and after.pixelColor(32, 18).blue() > 200
    p.show()
    r = p.view.picture_rect()
    QTest.mouseClick(p.view, Qt.LeftButton, pos=QPoint(int(r.left() + r.width() * 0.25), int(r.center().y())))
    assert abs(p.view.split - 0.25) < 0.02
    p.seek(23)
    p.play()                                                             # at the end: from the start again
    assert p.position() == 0 and p.playing()
    wait(app, lambda: not p.playing(), timeout=10)                       # plays through and stops at the end
    assert p.position() == 23
    p.load(clips[1], 24, (64, 36))                                       # another clip: the result is gone
    assert p.result is None and p.count() == 0
    wait(app, lambda: p.count() == 24, timeout=20)
    p.stop_loading()
    p.close()
    p.deleteLater()


def test_preview_paints_frames_without_a_native_video_window(app, tmp_path, monkeypatch):
    """The players show frames through a QVideoSink painted in a normal widget: QVideoWidget's own GPU window
    crashed in NVIDIA's EGL driver (segfault) while ComfyUI held the GPU memory."""
    import subprocess

    from video_chain.ui.main_window import Preview
    from video_chain.ui.widgets import VideoView
    monkeypatch.undo()                                                   # (this test plays: a clip without sound,
    clip = tmp_path / "c.mp4"                                            # the player muted as well)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x180:r=24:d=2", "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    p = Preview()
    p.resize(640, 420)
    p.show()
    p._ensure_player()
    p.audio.setMuted(True)
    p.open(clip, "clip")
    assert isinstance(p.video, VideoView)
    wait(app, lambda: not p.video.image().isNull(), timeout=20)
    assert p.video.image().width() == 320
    p.video.repaint()                                                    # (paints the frame scaled into the view)
    wait(app, lambda: p.player.position() >= 1900, timeout=20)
    for _ in range(50):
        app.processEvents()
    assert not p.video.image().isNull()                                  # the end: the last frame stays
    p.stop()
    assert p.video.image().isNull()
    p.close()
    p.deleteLater()
    import video_chain.ui as ui
    for f in Path(ui.__file__).parent.glob("*.py"):                      # no QVideoWidget anywhere in the app
        assert "QVideoWidget()" not in f.read_text(encoding="utf-8"), f.name


def test_video_view_shrinks_frames_without_aliasing(app):
    """A 2K portrait in a small preview: shrunk by area averaging, so fine detail does not turn into shimmering
    grain (a painter's bilinear scaling skips most source pixels)."""
    import numpy as np
    from PySide6.QtGui import QImage

    from video_chain.ui.widgets import VideoView
    board = (np.indices((2112, 1696)).sum(axis=0) % 2 * 255).astype(np.uint8)       # 1-px checkerboard
    rgb = np.ascontiguousarray(np.repeat(board[..., None], 3, axis=2))
    img = QImage(rgb.data, 1696, 2112, 1696 * 3, QImage.Format_RGB888).copy()
    v = VideoView()
    v.resize(200, 250)
    v._img = img
    small = v.scaled_frame()
    assert small.height() <= 250 * v.devicePixelRatioF() and small.width() < 1696
    g = small.convertToFormat(QImage.Format_Grayscale8)
    a = np.frombuffer(g.constBits(), np.uint8).reshape(g.height(), g.bytesPerLine())[:, :g.width()].astype(float)
    assert abs(a.mean() - 127.5) < 6 and a.std() < 8                                  # an even grey, no false pattern
    assert v.scaled_frame() is small                                                    # (once per frame and size)
    v.deleteLater()


def test_camera_row_and_hash_tokens(app, win, tmp_path, image):
    from PySide6.QtGui import QTextCursor
    ids = new_project(win, tmp_path, image, ("She waits",))
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.cam_move.currentData() is None and not card.cam_amp.isEnabled()
    card.cam_move.setCurrentIndex(card.cam_move.findData("push-in"))
    assert card.cam_amp.isEnabled() and card.cam_speed.isEnabled()
    card.cam_amp.setCurrentIndex(card.cam_amp.findData("small"))
    card.cam_speed.setCurrentIndex(card.cam_speed.findData("slow"))
    flush(app, win)
    node = win.tree.nodes[ids[0]]
    assert node.overrides["camera"] == {"move": "push-in", "amplitude": "small", "speed": "slow"}
    assert "🎥 push in · small · slow" in win.badge(node)
    card = next(c for c in win.cards if c.nid == ids[0])
    assert "camera pushes in with small amplitude at slow speed" in card.edit.toolTip()   # (sent to the model)
    card.cam_move.setCurrentIndex(card.cam_move.findData("static"))                     # no amplitude / speed
    assert not card.cam_amp.isEnabled() and not card.cam_speed.isEnabled()
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["camera"] == {"move": "static"}
    # typing # offers the camera tokens; the chosen one goes in whole
    card = next(c for c in win.cards if c.nid == ids[0])
    card.edit.setFocus()
    card.edit.setPlainText("She waits, #pan-ri")
    card.edit.moveCursor(QTextCursor.End)
    kind, m = card.mentions._typed()
    assert m is not None and kind == "#"
    card.mentions._kind = kind                      # (what _update sets when it shows the list; offscreen: no focus)
    card.mentions._insert("pan-right-slow")
    assert card.edit.toPlainText() == "She waits, #pan-right-slow "
    card.edit.setPlainText("Hi @Ann")                                                   # (@ still the cast)
    card.edit.moveCursor(QTextCursor.End)
    assert card.mentions._typed()[0] == "@" and card.mentions._typed()[1] is not None


def test_file_browser_for_videos_and_sounds(app, win, tmp_path, image):
    """The app's own browser also for videos (a frame as thumbnail, length and size) and sounds (their icon)."""
    import subprocess

    from PIL import Image as PImage

    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    d = tmp_path / "media"
    d.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=96x54:r=24:d=2", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(d / "clip.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    str(d / "voice.wav")], check=True)
    PImage.new("RGB", (40, 30), (0, 0, 255)).save(d / "pic.png")
    q = settings(tmp_path, "picker.ini")

    def names(dlg):
        return sorted(dlg.grid.item(i).text() for i in range(dlg.grid.count()))
    vids = ImagePickerDialog(None, "Real video", str(d), q, cache_dir=None, desktop_thumbs=None, kind="video")
    assert names(vids) == ["clip.mp4"] and "(1 videos)" in vids.windowTitle()
    item = vids.grid.item(0)
    wait(app, lambda: item.data(Qt.UserRole + 1) == "96×54", timeout=20)          # a frame became the thumbnail
    vids.grid.setCurrentItem(item)
    wait(app, lambda: "0:02.0" in vids.info.text(), timeout=20)                     # (the details load in the pool)
    assert "96×54" in vids.info.text() and "24 fps" in vids.info.text()
    vids._accept()
    assert vids.chosen() == str(d / "clip.mp4") and q.value("videoDir", type=str) == str(d)   # (its own folder)
    vids.deleteLater()
    media = ImagePickerDialog(None, "Voice", str(d), q, cache_dir=None, desktop_thumbs=None, kind="media")
    assert names(media) == ["clip.mp4", "voice.wav"]
    media.grid.setCurrentItem(next(media.grid.item(i) for i in range(media.grid.count())
                                   if media.grid.item(i).text() == "voice.wav"))
    wait(app, lambda: "0:01.0" in media.info.text(), timeout=20)                   # (the details load in the pool)
    assert "sound" in media.info.text()
    media.deleteLater()
    # the project's own real videos are offered first
    win.open_project(tmp_path / "proj")
    win.set_image(str(image))
    v = win.add_video(None, str(d / "clip.mp4"), 0.0, 1.0)
    cols = win.project_media("video")
    assert cols == [("Real videos of this project", [(str(d / "clip.mp4"), f"Video #{v[1:]} · clip.mp4")])]


def test_file_browser_favorites(app, tmp_path):
    """Folders as favorites: the star adds / removes the open folder; shared by every kind; right-click removes."""
    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    q = settings(tmp_path, "fav.ini")
    a, b = tmp_path / "shots", tmp_path / "music"
    a.mkdir()
    b.mkdir()
    dlg = ImagePickerDialog(None, "Choose", str(a), q, cache_dir=None, desktop_thumbs=None)

    def places(d):
        return [(d.places.item(i).text(), d.places.item(i).data(Qt.UserRole)) for i in range(d.places.count())]
    assert "Favorites" not in [t for t, _d in places(dlg)] and dlg.b_fav.toolTip().startswith("Add")
    dlg.b_fav.click()                                                          # the open folder: a favorite
    assert dlg.favorites() == [str(a)] and dlg.b_fav.toolTip().startswith("Remove")
    texts = [t for t, _d in places(dlg)]
    assert texts.index("Favorites") < texts.index("shots") < texts.index("Folders")
    dlg.open_folder(b)
    dlg.b_fav.click()
    assert dlg.favorites() == [str(a), str(b)]
    dlg.deleteLater()
    vids = ImagePickerDialog(None, "Video", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="video")
    assert vids.favorites() == [str(a), str(b)]                                 # shared by every kind
    row = next(i for i in range(vids.places.count()) if vids.places.item(i).data(Qt.UserRole) == str(a))
    vids._place_clicked(vids.places.item(row))
    assert vids.folder == a.resolve()
    pos = vids.places.visualItemRect(vids.places.item(row)).center()
    vids._exec_menu = lambda m, p: m.actions()[0]                               # right-click → Remove from favorites
    vids._places_menu(pos)
    assert vids.favorites() == [str(b)]
    vids.deleteLater()


def test_split_a_video_and_insert_a_clip_in_the_window(app, win, tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint

    from video_chain.ui import main_window as M
    src = make_real_video(tmp_path / "real.mp4", secs=4.0)
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")                                            # (a new project's empty first clip)
    v = win.add_video(None, str(src), 0.0, 3.0)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done"
    # the menu entry opens the dialog; the dialog's frame becomes the moment in the file
    seen = {}

    def fake_exec(dlg):
        wait(app, lambda: dlg.player.count() > 0, timeout=20)
        dlg.player.seek(48)                                                    # frame 48 = 2.0 s into the range
        seen["text"] = dlg.summary.text()
        seen["at"] = dlg.at()
        return QDialog.Accepted
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(M.SplitDialog, "exec", fake_exec)
    monkeypatch.setattr(win, "_exec_menu", lambda m, pos: next(a for a in m.actions()
                                                                if a.text() == "Split here and insert a clip…"))
    win._context(v, QPoint(0, 0))
    wait(app, lambda: len(win.tree.nodes) == 3, timeout=10)
    assert seen["at"] == 2.0 and "Part 1: 0.00–2.00 s" in seen["text"]
    clip = win.selected_id()
    rest = win.tree.video_child(clip)
    assert win.tree.nodes[v].source["end"] == 2.0 and rest.source == {"file": str(src), "start": 2.0, "end": 3.0}
    assert "write the prompt of the new clip" in win.status.text()
    wait(app, lambda: not win._converts, timeout=30)
    assert win.state(v) == "done" and win.state(rest.id) == "done" and win.state(clip) == "new"


def test_end_image_from_a_video_frame(app, win, tmp_path, image, monkeypatch):
    """A frame of any video as a clip's end image: exactly the chosen frame, at the video's full size."""
    import subprocess

    from PIL import Image as PImage
    from PySide6.QtWidgets import QDialog

    from video_chain.ui import main_window as M
    from video_chain.ui.clip_dialogs import VideoFrameDialog
    vid = tmp_path / "ramp.mp4"                                             # frame k has the brightness 5k
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=128x72:r=24:d=2",
                    "-vf", "geq=lum='5*N':cb=128:cr=128", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
                    str(vid)], check=True)
    dlg = VideoFrameDialog(None, vid, "frame")
    assert wait(app, lambda: dlg.ready)                                    # (a picked file is probed in a worker)
    assert dlg.frames == 48 and dlg.fps == 24
    def exact(k):                                                          # ffmpeg's decode of exactly frame k
        ref = tmp_path / f"ref{k}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
                        str(ref)], check=True)
        return PImage.open(ref).convert("L").getpixel((64, 36))
    for k in (0, 1, 13, 24, 47):
        dlg.slider.setValue(k)
        out = tmp_path / f"f{k}.png"
        assert dlg.frame_file(out)
        im = PImage.open(out).convert("L")
        assert im.size == (128, 72) and im.getpixel((64, 36)) == exact(k), (k, im.getpixel((64, 36)), exact(k))
    dlg.deleteLater()
    ids = new_project(win, tmp_path, image, ("a",))
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.b_end_video.isEnabled()

    def fake_exec(d):
        assert wait(app, lambda: d.ready)
        d.slider.setValue(20)
        return QDialog.Accepted
    monkeypatch.setattr(M.VideoFrameDialog, "exec", fake_exec)
    win.end_from_video(ids[0], video=str(vid))
    end = win.tree.nodes[ids[0]].end_image
    assert end and PImage.open(win.project / end).convert("L").getpixel((64, 36)) == exact(20)
    assert "ends at frame 21 (0.83 s) of ramp.mp4" in win.status.text()


def test_video_view_gpu_or_cpu(app, win):
    """The player view draws through Qt's video widget (GPU) or paints frames itself (CPU); the setting is kept
    at once (read at the next start, also after a crash)."""
    from video_chain.ui.widgets import VideoView
    gpu, cpu = VideoView(gpu=True), VideoView(gpu=False)
    assert gpu.widget is not None and gpu.sink is gpu.widget.videoSink() and gpu.image().isNull()
    assert cpu.widget is None and cpu.sink is not None
    gpu.clear()
    gpu.deleteLater()
    cpu.deleteLater()
    win.video_gpu.setChecked(False)
    assert win.qs.value("videoGpu", True, type=bool) is False and "CPU" in win.status.text()
    win.video_gpu.setChecked(True)
    assert win.qs.value("videoGpu", False, type=bool) is True


def test_middle_keyframes_on_the_card(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import image_picker as IP
    """The keyframe strip: add (picture; the strip's menu), move, note, keep a moment of the made clip, remove;
    the clip counts as changed and the made clip's hard cuts show as a warning."""
    from PIL import Image as PImage
    from PySide6.QtWidgets import QInputDialog

    from video_chain import tree as T
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a",))
    nid = ids[0]
    card = next(c for c in win.cards if c.nid == nid)
    assert card.keys.isEnabled() and card.keys.items == [] and not card.keys.made
    pic = tmp_path / "key.png"
    PImage.new("RGB", (192, 108), (0, 200, 0)).save(pic)
    monkeypatch.setattr(IP, "pick_image", lambda *a, **k: str(pic))
    monkeypatch.setattr(M.PromptCard, "_exec_menu", lambda self, m: m.actions()[0])     # "A picture at …"
    card.keys.addAt.emit(1.0)                                     # a click on the strip at 1 s
    node = win.tree.nodes[nid]
    assert [k["at"] for k in node.keyframes] == [1.0] and "keyframe at 1.00 s set" in win.status.text()
    card = next(c for c in win.cards if c.nid == nid)
    assert len(card.keys.items) == 1 and card.keys.items[0]["ok"] and "1 keyframe" in win.badge(node)
    win.keyframe_action("move", nid, {"index": 0, "at": 1.5})
    assert node.keyframes[0]["at"] == 1.5
    win.keyframe_action("move", nid, {"index": 0, "at": 99.0})    # never past the clip's second-to-last frame
    assert round(node.keyframes[0]["at"] * 24) == T.clip_length(win.tree, node) - 2
    win.keyframe_action("move", nid, {"index": 0, "at": 1.5})
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("she turns", True)))
    win.keyframe_action("note", nid, {"index": 0})
    assert node.keyframes[0]["note"] == "she turns"
    run_and_wait(app, win, win.generate_path)
    node = win.tree.nodes[nid]
    assert node.made["keyframes"] == node.keyframes and T.Checker(win.project, win.tree).status(nid) == "done"
    card = next(c for c in win.cards if c.nid == nid)
    assert card.keys.made
    if node.made.get("cuts"):                                     # (the fake clip's coloured last frame is a jump)
        assert card.key_warn.isVisibleTo(card) and "cuts at" in card.key_warn.text()
    # keep a moment of the made clip: its frame, pinned at its time
    win.modal_dialogs = False
    dlg = win._keyframe_from_video(nid, 0.5, None, own=True)
    assert dlg is not None and dlg.slider.value() == 12
    dlg.slider.setValue(6)
    assert win._keyframe_from_dialog(nid, dlg, 0.5, None, True)
    dlg.deleteLater()
    assert [k["at"] for k in node.keyframes] == [0.25, 1.5]
    assert T.changed_inputs(win.tree, nid) == ["keyframes"]
    win.keyframe_action("remove", nid, {"index": 0})
    assert [k["at"] for k in node.keyframes] == [1.5] and "removed" in win.status.text()
    win.node_action("revert", nid)
    assert node.keyframes == node.made["keyframes"]


def test_freeze_a_clip_on_its_card(app, win, tmp_path, image):
    """The lock button freezes a made clip and the ones before it: FROZEN, locked inputs, done after a size change;
    the edit actions are refused; the start image stays; clicking again unfreezes it (and the ones after it)."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    a, b = ids
    card = next(c for c in win.cards if c.nid == b)
    assert not card.b_freeze.isEnabled()                    # not made yet
    win.node_action("freeze_toggle", b)
    assert "not made yet" in win.status.text() and not win.tree.nodes[b].frozen
    run_and_wait(app, win, win.generate_path)
    card = next(c for c in win.cards if c.nid == b)
    assert card.b_freeze.isEnabled()
    card.b_freeze.click()
    assert win.tree.nodes[a].frozen and win.tree.nodes[b].frozen and "frozen, with 1 before it" in win.status.text()
    card = next(c for c in win.cards if c.nid == b)
    assert card.frozen and card.b_freeze.isChecked() and card.tag.text() == "FROZEN"
    assert card.edit.isReadOnly() and not card.seed.isEnabled() and not card.keys.isEnabled()
    assert not card.b_end.isEnabled() and card.buttons[0].isEnabled()          # a variant can still be made
    assert "🔒 frozen" in win.badge(win.tree.nodes[b])
    win.resolution.setCurrentIndex(win.resolution.findData(384))   # another size: still done
    win.refresh()
    assert win.state(a) == win.state(b) == "done"
    win.node_action("handover", b)
    assert "is frozen" in win.status.text() and win.tree.nodes[b].end_at is None
    win.keyframe_action("add", b, {"at": 1.0})
    assert win.tree.nodes[b].keyframes == []
    assert not win.set_image(str(image)) and "unfreeze it first" in win.status.text()
    win.node_action("freeze_toggle", a)                     # unfreezing a takes b along
    assert not win.tree.nodes[a].frozen and not win.tree.nodes[b].frozen
    card = next(c for c in win.cards if c.nid == b)
    assert not card.frozen and not card.edit.isReadOnly()
    assert win.state(a) == "changed" and "CHANGED" in win.status.text()


def test_continue_the_clip_before_on_the_card(app, win, tmp_path, image):
    """The card's Continue combo: only in the official format and not on a first clip (disabled, the tooltip says
    why); the lengths that fit both clips; it sets the override, shows a chip and the <Video 1> entry of the Model
    prompt window, is advanced-only and counts for the card's note; a value that no longer applies can be taken back."""
    from video_chain.ui import main_window as MW
    ids = new_project(win, tmp_path, image)
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert not card.cont.isEnabled() and card.cont.toolTip() == MW.CONTINUE_CLASSIC_TIP   # classic (tests' default)
    win.prompt_format.setCurrentIndex(win.prompt_format.findData("h3"))
    flush(app, win)
    first, card = (next(c for c in win.cards if c.nid == i) for i in ids)
    assert not first.cont.isEnabled() and first.cont.toolTip() == MW.CONTINUE_FIRST_TIP
    assert card.cont.isEnabled() and card.cont.toolTip() == MW.CONTINUE_TIP
    assert [card.cont.itemData(k) for k in range(card.cont.count())] == [None, 22, 39, 56, 73]   # 73-frame clips
    assert card.cont.itemText(0) == "off" and card.cont.itemText(2) == "last 1.6 s (39 frames)"
    assert "Carry-over is different: it fixes the first frames" in MW.CONTINUE_TIP and "takes longer" in MW.CONTINUE_TIP
    card.cont.setCurrentIndex(card.cont.findData(39))
    assert win.tree.nodes[ids[1]].overrides == {"continue": 39} and card.advanced_active() == ["Continue"]
    flush(app, win)
    card, chips = _chips(win, ids[1])
    assert chips == ["Start frame · Picture 1", "Continues clip before · Video 1", "With reference pictures"]
    assert card.chips.chips[1].toolTip().startswith(f"The end of clip #{ids[0][1:]} (its last 1.6 s, 39 frames")
    assert T.Checker(win.project, win.tree).settings(ids[1])["continue"] == {"frames": 39}
    win.modal_dialogs = False
    card.chips.b_prompt.click()
    dlg = win._model_prompt_dialog
    assert dlg.inputs.text().splitlines() == ["<Picture 1> — start frame (" + Path(str(T.clip_inputs(
        win.project, win.tree, ids[1])["start"]["file"])).name + ")", f"<Video 1> — end of clip #{ids[0][1:]} (39 frames)"]
    assert "The shot continues directly from the end of <Video 1>." in dlg.text.toPlainText()
    dlg.deleteLater()
    win.set_simple(True)                                                 # Simple: hidden, the note counts it
    card = next(c for c in win.cards if c.nid == ids[1])
    assert _row_hidden(card.link_form, card.cont) and card.adv_note.text.text() == "1 advanced setting is active:"
    win.set_simple(False)
    win.prompt_format.setCurrentIndex(win.prompt_format.findData("classic"))   # set, but classic: no effect
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.cont.isEnabled() and card.cont.currentData() == 39 and "choose 'off'" in card.cont.toolTip()
    assert "continue" not in T.Checker(win.project, win.tree).settings(ids[1])
    card.cont.setCurrentIndex(0)
    assert win.tree.nodes[ids[1]].overrides == {}
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert not card.cont.isEnabled() and card.cont.count() == 1


def test_hash_completion_offers_camera_moves_in_plain_words(app):
    """Typing # offers the camera moves in plain words; the chosen one goes in as a token camera.expand knows."""
    from video_chain import camera as CAM
    from video_chain.ui.main_window import PromptCard
    card = PromptCard(T.Node("n1", None, ""))
    card.show()
    card.activateWindow()
    card.edit.setFocus()
    app.processEvents()
    card.edit.insertPlainText("A boat drifts while #push-in")
    app.processEvents()
    offered = card.mentions.offered()
    assert ("Push in, slow", "push-in-slow") in offered and ("Push in", "push-in") not in offered[1:]
    assert all(CAM.is_token(t) for _w, t in offered)
    cm = card.mentions.completer.completionModel()
    idx = cm.index([t for _w, t in offered].index("push-in-slow"), 0)
    assert card.mentions.completer.pathFromIndex(cm.mapToSource(idx)) == "push-in-slow"   # what activated() gives
    popup = card.mentions.completer.popup()
    if popup.isVisible():                                 # Enter in the list, as the user does
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        popup.setCurrentIndex(idx)
        QTest.keyClick(card.edit, Qt.Key_Return)
    else:
        card.mentions._insert("push-in-slow")
    text = card.edit.toPlainText()
    assert text.startswith("A boat drifts while #push-in-slow")
    assert "while the camera pushes in at slow speed" in CAM.expand(text)
    assert CAM.label("pan-left-large-fast") == "Pan left, large, fast" and CAM.label("nope") is None
    card.close()
    card.deleteLater()