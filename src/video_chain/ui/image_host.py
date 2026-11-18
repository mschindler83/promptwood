"""What the image dialog's mixins (image_*.py) use of the dialog they are mixed into (ImageEditDialog, image_edit.py).

At run time ImageEditHost is an empty class. For the type checker it is the dialog: it declares the attributes the
mixins share (the widgets built in image_edit.py and its state) and falls back to Any for the methods of the other
mixins (those are checked in the mixin that defines them). A new attribute that two mixins share is declared here;
mypy says so ("... has no attribute")."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QLabel, QLineEdit, QListWidget, QPlainTextEdit,
        QPushButton, QSlider, QSpinBox, QStackedWidget, QToolButton, QWidget)
    from pathlib import Path
    from ..edits import ResultRecord
    from .image_common import PictureSlot
    from .prompt_refiner import RefineResult
    from .lora_stack import LoraStack
    from .results_gallery import InfoCard, ResultGallery, ResultView, ThumbLoader
    from .variations_panel import PlaceholderHighlighter, VariationsPanel
    from .widgets import FrameTile

    class ImageEditHost(QDialog):
        _rehighlighting: bool
        _sampling_family: str | None
        _was_down: bool
        _built: bool
        _busy: bool
        _failed: set[tuple[str, int]]
        _highlight: PlaceholderHighlighter
        _loader: ThumbLoader
        _requested: set[tuple[str, int]]
        _retry_families: int
        _thumbs: dict[str, tuple[int, QPixmap]]
        _var_timer: QTimer
        aspect: QComboBox
        b_before: QPushButton
        b_fav: QPushButton
        b_gallery: QPushButton
        b_gen: QPushButton
        b_mask: QPushButton
        b_mask_clear: QPushButton
        b_random: QPushButton
        b_refine: QToolButton
        b_single: QPushButton
        b_start_comfy: QPushButton
        b_undo: QPushButton
        batch: list[str]
        check_lora: Callable[[str | None], bool | None] | None
        collections: Callable[[], list] | None
        connect_create: Callable[[Callable | None], object] | None
        cpics: list[PictureSlot]
        create_box: QWidget
        edit_box: QWidget
        edit_model: QComboBox
        edit_pics: QWidget
        edit_shape: QComboBox
        edit_shape_label: QLabel
        f_count: QLabel
        f_model: QComboBox
        f_search: QLineEdit
        f_show: QComboBox
        families: dict
        family: QComboBox
        family_loras: dict[str, list]
        fresh_seed: QCheckBox
        gallery: ResultGallery
        keep_size: QCheckBox
        known_loras: list[tuple[str, str]]
        lora_each: QCheckBox
        lora_text: QLabel
        loras: LoraStack
        m_create: QPushButton
        m_edit: QPushButton
        mask_info: QLabel
        model_file: QComboBox
        negative: QLineEdit
        pic2: PictureSlot
        pic3: PictureSlot
        places: list[tuple[str, str]]
        project: Path
        prompt: QPlainTextEdit
        prompt_title: QLabel
        qs: QSettings | None
        quality: QComboBox
        ref_strength: QDoubleSpinBox
        ref_strength_label: QLabel
        ref_text: QLabel
        refine: Callable[..., RefineResult | None] | None
        res_info: InfoCard
        results: QListWidget
        seed: QSpinBox
        size_text: QLabel
        source: str | None
        src_combo: QComboBox
        src_info: QLabel
        src_tile: FrameTile
        stack: QStackedWidget
        start_comfy: Callable[[], bool] | None
        state: QLabel
        mask: QImage | None  # type: ignore[assignment]   # (shadows QWidget.mask(), which the dialog never uses)
        tags: QCheckBox
        thumb_size: QSlider
        tries: QSpinBox
        use_refs: QCheckBox
        use_triggers: bool
        uses: Callable[[ResultRecord], list[str]]
        variations: VariationsPanel
        view: ResultView

        def __getattr__(self, name: str) -> Any: ...
else:
    class ImageEditHost:
        pass
