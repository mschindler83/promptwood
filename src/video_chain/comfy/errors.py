"""The errors of the ComfyUI client."""
from __future__ import annotations


class ComfyError(RuntimeError):
    pass


class LocalWriteError(Exception):
    """Wraps an OSError of writing a download to the local disk (told apart from a network error)."""


class ComfyUnavailable(ComfyError):
    pass


class ComfyJobFailed(ComfyError):
    """ComfyUI itself ended the job with an error (or it was interrupted): nothing is left to cancel."""
