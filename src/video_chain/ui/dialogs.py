"""One way to ask, tell and close: the questions and messages every window shows, and the button row of a dialog.

Convention (the words are in widgets.TERMS):
- A question about something that cannot be taken back is `confirm()`: Yes / No, and No is the default button.
- `tell` is for information, `warn` for a problem, `failed` for "Could not <what>: <reason>".
- A dialog that makes a choice has `button_row(dialog, "Use this crop")`: the action says what it does, then Cancel.
  A dialog that holds nothing back has `button_row(dialog)`: just Close. Extra buttons (Other file…) are action
  buttons (`add_action`), a destructive one says so (`destructive=True`)."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QPushButton


def confirm(parent, title: str, text: str) -> bool:
    """Ask a yes / no question; No is the default button (Enter does not confirm)."""
    return QMessageBox.question(parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def tell(parent, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def warn(parent, title: str, text: str) -> None:
    QMessageBox.warning(parent, title, text)


def failed(parent, title: str, what: str, exc) -> None:
    """"Could not <what>: <exc>" as a warning."""
    warn(parent, title, f"Could not {what}: {exc}")


def button_row(dialog: QDialog, ok_text: str | None = None, *, cancel_text: str = "Cancel",
               enabled: bool = True, accept=None) -> tuple[QDialogButtonBox, QPushButton | None]:
    """The buttons of a dialog, wired to it. With ok_text: that action button (Ok role: what it does, in words) and
    Cancel; the box and the action button come back (accept: what the action does instead of dialog.accept).
    Without ok_text: a Close button, and None for the action."""
    if ok_text is None:
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(dialog.reject)
        return box, None
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    ok = box.button(QDialogButtonBox.StandardButton.Ok)
    ok.setText(ok_text)
    ok.setEnabled(enabled)
    box.button(QDialogButtonBox.StandardButton.Cancel).setText(cancel_text)
    box.accepted.connect(accept or dialog.accept)
    box.rejected.connect(dialog.reject)
    return box, ok


def add_action(box: QDialogButtonBox, text: str, *, destructive: bool = False) -> QPushButton:
    """An extra button of the row: an action of its own (not accept / reject)."""
    return box.addButton(text, QDialogButtonBox.ButtonRole.DestructiveRole if destructive else QDialogButtonBox.ButtonRole.ActionRole)
