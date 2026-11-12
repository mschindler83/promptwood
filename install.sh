#!/usr/bin/env bash
# Promptwood: one command from nothing to a working app.
#
#   ./install.sh                      asks what to install, shows the plan, then installs it
#   ./install.sh --profile core       video clips only (the smallest set)
#   ./install.sh --dry-run            only show what would be installed and downloaded
#   ./install.sh --yes --accept-licenses        without questions
#   ./install.sh --models-from ~/ComfyUI        a new ComfyUI that reads the models you already have, in place
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
        say "Downloading Promptwood (${VIDEO_CHAIN_REF:-master}) into $SRC …"
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
    say "Getting uv, the tool that sets up Python environments (into $HOME_DIR/tools/bin; nothing else on the system is changed) …"
    fetch https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME_DIR/tools/bin" sh >/dev/null 2>&1 \
        || die "could not install uv (https://docs.astral.sh/uv/getting-started/installation/)"
    UV="$HOME_DIR/tools/bin/uv"
    [ -x "$UV" ] || die "uv was not installed to $UV"
fi
export PATH="$(dirname "$UV"):$PATH"

# ---- first the questions, the plan and the checks: before anything big is installed
# (run with a bare Python 3.12 from uv; the installer itself needs nothing but the standard library)
ask() {
    ( cd "$SRC" && PYTHONPATH="$SRC/src" "$UV" run -q --no-project --python 3.12 --with pyyaml \
        python -m video_chain.setup setup --phase ask "$@" )
}
for arg in "$@"; do
    case "$arg" in
        --help|-h|--dry-run) ask "$@"; exit $? ;;
    esac
done
say "Preparing (uv fetches a small Python for the installer the first time) …"
ask "$@" || exit $?

# ---- the app in its own environment (the versions the app was tested with: uv.lock)
mkdir -p "$HOME_DIR/logs"
say ""
say "Installing Promptwood itself into $SRC/.venv (about 1 minute; details in $HOME_DIR/logs/app.log) …"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
sync_app() { ( cd "$SRC" && "$UV" sync --locked --extra faces --python 3.12 ) >>"$HOME_DIR/logs/app.log" 2>&1; }
: >"$HOME_DIR/logs/app.log"
sync_app || { say "A download did not get through; trying again …"; sleep 10; sync_app; } \
    || { say "Trying once more …"; sleep 30; sync_app; } \
    || { tail -n 25 "$HOME_DIR/logs/app.log" >&2; die "could not install the app's Python packages (see above; all of it is in $HOME_DIR/logs/app.log)"; }

# ---- everything else: ComfyUI, models, text to speech, llama.cpp, ffmpeg; then the checks and the tests
exec "$SRC/.venv/bin/python" -m video_chain setup --answers "$@"
