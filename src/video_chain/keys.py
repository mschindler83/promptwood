"""API keys in the system keyring (CivitAI, RunPod, ElevenLabs): one reader and one writer, so a fix is made once."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def read_key(service: str, user: str) -> str | None:
    """The key kept in the keyring ("" when none is kept); None when there is no keyring to ask (missing, locked)."""
    try:
        import keyring
        return (keyring.get_password(service, user) or "").strip()
    except Exception as e:  # noqa: BLE001  (a keyring backend can fail in many ways; there is just no key then)
        logger.warning("the system keyring could not be read: %s", e)
        return None


def get_key(service: str, user: str) -> str:
    return read_key(service, user) or ""


def set_key(service: str, user: str, key: str | None) -> None:
    """Keep the key (empty / None: remove it). RuntimeError: there is no keyring to keep it in."""
    key = (key or "").strip()
    try:
        import keyring
        import keyring.errors
        if key:
            keyring.set_password(service, user, key)
        else:
            try:
                keyring.delete_password(service, user)
            except keyring.errors.PasswordDeleteError:
                pass                                  # (nothing was kept)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the system keyring is not available ({e})") from e
