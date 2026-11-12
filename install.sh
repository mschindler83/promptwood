#!/usr/bin/env bash
# Promptwood: one command from nothing to a working app.
#
#   ./install.sh                      everything the app needs (profile "standard", about 126 GB of models)
#   ./install.sh --profile core       video clips only (about 40 GB)
#   ./install.sh --yes --accept-licenses        without questions
#   ./install.sh --comfy-dir ~/ComfyUI          use a ComfyUI you already have
#   ./install.sh --help               every option
#
# It needs no root and changes nothing outside two folders: this one (the app and its .venv) and the home folder
# ($VIDEO_CHAIN_HOME, default ~/.local/share/video-chain: ComfyUI, models, tools). What it cannot install itself
# (the NVIDIA driver, a few system libraries) it names, with the command for this system.
# Running it again continues where it stopped.
set -euo pipefail

say() { printf '%s\n' "$*"; }
die() { printf '\nInstall stopped: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
    Linux) ;;
    *) say "Note: Promptwood is tested on Linux (x86-64, NVIDIA) only; on $(uname -s) parts may not work." ;;
esac

HOME_DIR="${VIDEO_CHAIN_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/video-chain}"
export VIDEO_CHAIN_HOME="$HOME_DIR"
mkdir -p "$HOME_DIR/tools/bin" || die "cannot create $HOME_DIR"

fetch() {   # fetch URL  -> stdout
    if command -v curl >/dev/null 2>&1; then curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys, shutil, urllib.request as u; shutil.copyfileobj(u.urlopen(u.Request(sys.argv[1], headers={"User-Agent": "video-chain-setup"}), timeout=60), sys.stdout.buffer)' "$1"
    else die "curl, wget or python3 is needed to download (e.g. sudo apt install curl / sudo dnf install curl)"; fi
}

# ---- the app's source: this folder, or (when the script was piped into a shell) a download
SRC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -z "$SRC" ] || [ ! -f "$SRC/pyproject.toml" ] || ! grep -q 'name = "video-chain"' "$SRC/pyproject.toml"; then
    [ -n "${VIDEO_CHAIN_REPO:-}" ] || die "run install.sh from Promptwood's folder, or set VIDEO_CHAIN_REPO to its repository address (https://…)"
    SRC="$HOME_DIR/app"
    if [ ! -f "$SRC/pyproject.toml" ]; then
        say "==> Downloading Promptwood (${VIDEO_CHAIN_REF:-master}) into $SRC"
        mkdir -p "$SRC"
        command -v tar >/dev/null 2>&1 || die "tar is needed to unpack the download"
        fetch "${VIDEO_CHAIN_REPO%.git}/archive/${VIDEO_CHAIN_REF:-master}.tar.gz" | tar -xz --strip-components=1 -C "$SRC" \
            || die "could not download ${VIDEO_CHAIN_REPO}"
    fi
fi

# ---- uv (makes the Python environments and brings Python 3.12 itself when the system has none)
UV="$(command -v uv 2>/dev/null || true)"
[ -n "$UV" ] || { [ -x "$HOME_DIR/tools/bin/uv" ] && UV="$HOME_DIR/tools/bin/uv"; } || true
[ -n "$UV" ] || { [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"; } || true
if [ -z "$UV" ]; then
    say "==> Installing uv into $HOME_DIR/tools/bin (nothing else on the system is changed)"
    fetch https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME_DIR/tools/bin" sh >/dev/null \
        || die "could not install uv (https://docs.astral.sh/uv/getting-started/installation/)"
    UV="$HOME_DIR/tools/bin/uv"
    [ -x "$UV" ] || die "uv was not installed to $UV"
fi
export PATH="$(dirname "$UV"):$PATH"

# ---- the app in its own environment (the versions the app was tested with: uv.lock)
say "==> Installing Promptwood into $SRC/.venv"
( cd "$SRC" && "$UV" sync --locked --extra faces --python 3.12 ) \
    || die "could not install the app's Python packages (see above)"

# ---- everything else: ComfyUI, models, text to speech, llama.cpp, ffmpeg; then the checks and a test clip
exec "$SRC/.venv/bin/python" -m video_chain setup "$@"
