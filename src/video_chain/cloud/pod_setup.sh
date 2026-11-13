#!/usr/bin/env bash
# Prepares a RunPod pod for video-chain's SeedVR2 upscale and starts ComfyUI on it. Sent by runpod.py over SSH
# (bash -s, as root) after the pod is up; safe to run again on the same pod or network volume.
#
#   COMFY_REF   ComfyUI git tag to install (pinned, so the workflow and the models match the app)
#   MODELS      which SeedVR2 sizes to fetch: "7b", "3b" or "7b 3b"
#   IDLE_MIN    the pod removes itself after this many minutes without an SSH connection (0: never)
#   MIN_RAM_GB  the RAM area (/dev/shm) must have at least this much room, or nothing is started
#
# Privacy: nothing of the user's is ever written to a RunPod disk. RunPod cannot encrypt container disks or network
# volumes (only a pod's volume disk, with a key RunPod holds, and not through the API), and a container may not
# mount dm-crypt or FUSE, so instead every private byte stays in RAM: uploads, results, ComfyUI's temp files, its
# database, user folder and log, and every program's temporary files (TMPDIR) live in /dev/shm/vc-io (tmpfs), which
# the app empties after each job and which is gone when the pod is terminated. The disks (container disk, network
# volume) hold only public files: ComfyUI's code, its Python packages and the public SeedVR2 weights.
# ComfyUI listens on 127.0.0.1 only: reached through the SSH tunnel, never through RunPod's public HTTP proxy.
# Lines "@@ <text>" are progress notes for the app.
set -euo pipefail

COMFY_REF="${COMFY_REF:-v0.45.3}"
MODELS="${MODELS:-7b}"
IDLE_MIN="${IDLE_MIN:-20}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
HF=https://huggingface.co/Comfy-Org/SeedVR2/resolve/main
note() { echo "@@ $*"; }
# an SSH session does not inherit the container's environment: take RunPod's variables from the init process
if [ -r /proc/1/environ ]; then
    while IFS= read -r -d '' kv; do
        case "$kv" in RUNPOD_*=*) export "$kv" ;; esac
    done < /proc/1/environ
fi

# the network volume (if any) keeps ComfyUI, its Python packages and the models between pods: later boots are fast
if [ -n "${RUNPOD_VOLUME_ID:-}" ] && [ -d /workspace ] && [ -w /workspace ]; then
    BASE=/workspace/video-chain
else
    BASE=/root/video-chain
fi
# ---- the private area: RAM only (tmpfs), or nothing runs
IO=/dev/shm/vc-io
if [ "$(stat -f -c %T /dev/shm 2>/dev/null)" != "tmpfs" ]; then
    echo "/dev/shm is not a RAM file system here: refusing to put private files on a disk" >&2
    exit 5
fi
room=$(( $(df -B1 --output=avail /dev/shm | tail -1) / 1073741824 ))
if [ "$room" -lt "$MIN_RAM_GB" ]; then
    echo "/dev/shm has only ${room} GB (needs ${MIN_RAM_GB} GB): refusing to put private files on a disk" >&2
    exit 5
fi
if [ -s /proc/swaps ] && [ "$(wc -l < /proc/swaps)" -gt 1 ]; then
    note "warning: the host has swap, RAM pages could be swapped out to its disk"
fi
umask 077
mkdir -p "$BASE" "$IO/input" "$IO/output" "$IO/temp" "$IO/user" "$IO/tmp"
chmod 700 "$IO"
export TMPDIR="$IO/tmp" UV_CACHE_DIR="$BASE/uv-cache" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1

# ---- ComfyUI at the pinned tag
COMFY="$BASE/ComfyUI"
if [ ! -d "$COMFY/.git" ]; then
    note "installing ComfyUI $COMFY_REF"
    git clone -q --depth 1 --branch "$COMFY_REF" https://github.com/comfyanonymous/ComfyUI "$COMFY"
elif [ "$(git -C "$COMFY" describe --tags --exact-match 2>/dev/null || true)" != "$COMFY_REF" ]; then
    note "switching ComfyUI to $COMFY_REF"
    git -C "$COMFY" fetch -q --depth 1 origin tag "$COMFY_REF"
    git -C "$COMFY" checkout -q --force "$COMFY_REF"
fi

# ---- its Python packages (on top of the image's CUDA PyTorch)
VENV="$BASE/venv"
STAMP="$VENV/.video-chain-$COMFY_REF-$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
if [ ! -f "$STAMP" ]; then
    note "installing ComfyUI's Python packages"
    command -v uv >/dev/null 2>&1 || python3 -m pip install -q --root-user-action=ignore uv
    rm -rf "$VENV"
    uv venv -q --system-site-packages --python "$(command -v python3)" "$VENV"
    # torch / torchvision / torchaudio come from the image (built for its CUDA); everything else from the tag
    grep -viE '^(torch|torchvision|torchaudio)([<>=~! ].*)?$' "$COMFY/requirements.txt" > "$BASE/requirements.txt"
    uv pip install -q --python "$VENV/bin/python" -r "$BASE/requirements.txt"
    touch "$STAMP"
fi

# ---- SeedVR2 models (Comfy-Org/SeedVR2), checked against Hugging Face's SHA-256 once
fetch() {   # fetch <repo path> <ComfyUI models subfolder>
    local src="$1" dir="$COMFY/models/$2" name
    name="$(basename "$1")"
    mkdir -p "$dir"
    [ -f "$dir/$name" ] && [ -f "$dir/.$name.ok" ] && return 0
    note "downloading $name"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$dir/$name.part" "$HF/$src?download=true"
    local want got
    want="$(curl -fsI -L "$HF/$src" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-etag"{gsub(/"/,"",$2); print $2}' | tail -1)"
    note "checking $name"
    got="$(sha256sum "$dir/$name.part" | cut -d' ' -f1)"
    if [ -n "$want" ] && [ "$want" != "$got" ]; then
        rm -f "$dir/$name.part"
        echo "checksum mismatch for $name (expected $want, got $got)" >&2
        exit 3
    fi
    mv "$dir/$name.part" "$dir/$name"
    touch "$dir/.$name.ok"
}
for m in $MODELS; do
    case "$m" in
        7b|3b) fetch "diffusion_models/seedvr2_${m}_int8_convrot.safetensors" diffusion_models ;;
        *) echo "unknown SeedVR2 size $m" >&2; exit 2 ;;
    esac
done
fetch vae/seedvr2_ema_vae_fp16.safetensors vae

# ---- ComfyUI on 127.0.0.1:8188 (only through the SSH tunnel)
if [ -f "$IO/comfy.pid" ] && kill -0 "$(cat "$IO/comfy.pid")" 2>/dev/null; then
    note "ComfyUI is already running"
else
    note "starting ComfyUI"
    cd "$COMFY"
    TMPDIR="$IO/tmp" nohup "$VENV/bin/python" main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
        --input-directory "$IO/input" --output-directory "$IO/output" --temp-directory "$IO/temp" \
        --user-directory "$IO/user" --database-url "sqlite:///$IO/user/comfyui.db" \
        > "$IO/comfy.log" 2>&1 < /dev/null &
    echo $! > "$IO/comfy.pid"
fi
for _ in $(seq 1 600); do
    if curl -fs -o /dev/null http://127.0.0.1:8188/system_stats; then break; fi
    if ! kill -0 "$(cat "$IO/comfy.pid")" 2>/dev/null; then
        echo "ComfyUI exited while starting; its last lines:" >&2
        tail -n 40 "$IO/comfy.log" >&2
        exit 4
    fi
    sleep 1
done
curl -fs -o /dev/null http://127.0.0.1:8188/system_stats || { echo "ComfyUI did not answer within 10 min" >&2; exit 4; }

# ---- watchdog: no SSH connection for IDLE_MIN minutes (the app was killed, the laptop went to sleep …) -> the
# pod removes itself, so a lost connection never keeps a GPU billed. The app terminates the pod itself normally.
if [ "$IDLE_MIN" -gt 0 ] && ! { [ -f "$IO/watchdog.pid" ] && kill -0 "$(cat "$IO/watchdog.pid")" 2>/dev/null; }; then
    cat > "$IO/watchdog.sh" <<'EOF'
#!/usr/bin/env bash
# established TCP connections to local port 22 (0x0016), IPv4 and IPv6: /proc needs no extra tools
ssh_up() { awk 'NR>1 && $4=="01" { split($2,a,":"); if (a[length(a)]=="0016") n++ } END { exit !n }' \
               /proc/net/tcp /proc/net/tcp6 2>/dev/null; }
idle=0
while sleep 60; do
    if ssh_up; then idle=0; else idle=$((idle + 1)); fi
    if [ "$idle" -ge "$IDLE_MIN" ]; then
        echo "$(date -Is) no SSH connection for $IDLE_MIN min: removing the pod" >> /dev/shm/vc-io/watchdog.log
        command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "$RUNPOD_POD_ID" >> /dev/shm/vc-io/watchdog.log 2>&1
        curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >> /dev/shm/vc-io/watchdog.log 2>&1
        sleep 300
    fi
done
EOF
    chmod 700 "$IO/watchdog.sh"
    IDLE_MIN="$IDLE_MIN" nohup "$IO/watchdog.sh" > /dev/null 2>&1 < /dev/null &
    echo $! > "$IO/watchdog.pid"
fi
note "ready"
