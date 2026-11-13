#!/usr/bin/env bash
# Prepares a RunPod pod for video-chain's SeedVR2 upscale and starts ComfyUI on it. Sent by runpod.py over SSH
# (bash -s, as root) after the pod is up; safe to run again on the same pod or network volume.
#
#   COMFY_REF   ComfyUI git tag to install (pinned, so the workflow and the models match the app)
#   MODELS      which SeedVR2 sizes to fetch: "7b", "3b" or "7b 3b"
#   IDLE_MIN    the pod removes itself after this many minutes without an SSH connection (0: never)
#   MIN_RAM_GB  the RAM area (/dev/shm) must have at least this much room, or nothing is started
#   VOLUME_PATH where the app mounted the pod's network or global volume ("" without one)
#   TORCH_PINS  the PyTorch packages to install, pinned (e.g. "torch==2.15.0 torchvision==0.29.2")
#
# Privacy: nothing of the user's is ever written to a RunPod disk. RunPod cannot encrypt container disks or network
# volumes (only a pod's volume disk, with a key RunPod holds, and not through the API), and a container may not
# mount dm-crypt or FUSE, so instead every private byte stays in RAM: uploads, results, ComfyUI's temp files, its
# database, user folder and log, and every program's temporary files (TMPDIR) live in /dev/shm/vc-io (tmpfs), which
# the app empties after each job and which is gone when the pod is terminated. The disks (container disk, network
# volume) hold only public files: ComfyUI's code and its Python packages (container disk), the public SeedVR2
# weights (the volume, if any).
# ComfyUI listens on 127.0.0.1 only: reached through the SSH tunnel, never through RunPod's public HTTP proxy.
# Lines "@@ <text>" are progress notes for the app.
set -euo pipefail

COMFY_REF="${COMFY_REF:-v0.45.3}"
MODELS="${MODELS:-7b}"
IDLE_MIN="${IDLE_MIN:-20}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
HF="${HF_BASE:-https://huggingface.co/Comfy-Org/SeedVR2/resolve/main}"   # (HF_BASE, CODE_DIR, IO_DIR: tests)
note() { echo "@@ $*"; }
# an SSH session does not inherit the container's environment: take RunPod's variables from the init process
if [ -r /proc/1/environ ]; then
    while IFS= read -r -d '' kv; do
        case "$kv" in RUNPOD_*=*) export "$kv" ;; esac
    done < /proc/1/environ
fi

# Everything goes on the container disk on every boot: ComfyUI's code and its Python packages (about a minute: a
# shallow clone and uv) and the SeedVR2 weights straight from Hugging Face. Measured on pods (2026-11-13): Hugging
# Face 111-182 MB/s; a global volume took writes at 7-9 MB/s and its VAE loaded in 16 s against seconds from the
# disk; it allows no chmod either (git and a venv need it). A volume is not used for anything any more.
CODE="${CODE_DIR:-/root/video-chain}"
VOLUME_PATH="${VOLUME_PATH:-}"
if [ -z "$VOLUME_PATH" ] && [ -n "${RUNPOD_VOLUME_ID:-}" ] && [ -d /workspace ]; then
    VOLUME_PATH=/workspace
fi
if [ -n "$VOLUME_PATH" ] && [ -d "$VOLUME_PATH/video-chain" ]; then
    # an earlier layout kept ComfyUI, its venv and the uv cache there: not any more (billed storage). The weights
    # an earlier pod put there stay (they are public; delete the volume when it is not wanted any more)
    rm -rf "$VOLUME_PATH/video-chain/ComfyUI" "$VOLUME_PATH/video-chain/venv" "$VOLUME_PATH/video-chain/uv-cache" \
        "$VOLUME_PATH/video-chain/requirements.txt" 2>/dev/null || true
fi

# the GPU: from 40 GB of VRAM the full-precision weights (fp16, nothing quantized) and ComfyUI keeps its models on
# the GPU (--highvram); smaller GPUs the int8 copy, as at home (the app picks the same way from ComfyUI's VRAM)
VRAM_MB="${VRAM_MB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)}"
VRAM_MB="${VRAM_MB:-0}"
if [ "$VRAM_MB" -ge 40000 ]; then PRECISION=fp16; HIGHVRAM=--highvram; else PRECISION=int8_convrot; HIGHVRAM=""; fi
note "GPU with $((VRAM_MB / 1024)) GB VRAM: SeedVR2 $PRECISION"
# ---- the private area: RAM only (tmpfs), or nothing runs
IO="${IO_DIR:-/dev/shm/vc-io}"
if [ "$(stat -f -c %T "$(dirname "$IO")" 2>/dev/null)" != "tmpfs" ]; then
    echo "/dev/shm is not a RAM file system here: refusing to put private files on a disk" >&2
    exit 5
fi
room=$(( $(df -B1 --output=avail "$(dirname "$IO")" | tail -1) / 1073741824 ))
if [ "$room" -lt "$MIN_RAM_GB" ]; then
    echo "/dev/shm has only ${room} GB (needs ${MIN_RAM_GB} GB): refusing to put private files on a disk" >&2
    exit 5
fi
if [ -s /proc/swaps ] && [ "$(wc -l < /proc/swaps)" -gt 1 ]; then
    note "warning: the host has swap, RAM pages could be swapped out to its disk"
fi
umask 077
mkdir -p "$CODE" "$IO/input" "$IO/output" "$IO/temp" "$IO/user" "$IO/tmp"
chmod 700 "$IO"
export TMPDIR="$IO/tmp" UV_CACHE_DIR="$CODE/uv-cache" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1

# ---- ComfyUI at the pinned tag
COMFY="$CODE/ComfyUI"
if [ ! -d "$COMFY/.git" ]; then
    note "installing ComfyUI $COMFY_REF"
    git clone -q --depth 1 --branch "$COMFY_REF" https://github.com/comfyanonymous/ComfyUI "$COMFY"
elif [ "$(git -C "$COMFY" describe --tags --exact-match 2>/dev/null || true)" != "$COMFY_REF" ]; then
    note "switching ComfyUI to $COMFY_REF"
    git -C "$COMFY" fetch -q --depth 1 origin tag "$COMFY_REF"
    git -C "$COMFY" checkout -q --force "$COMFY_REF"
fi

# ---- video-chain's nodes the app copied here (the lanczos resize on all cores)
if [ -d "$CODE/nodes" ]; then
    mkdir -p "$COMFY/custom_nodes"
    cp -r "$CODE/nodes/." "$COMFY/custom_nodes/"
fi

# ---- its Python packages (on top of the image's CUDA PyTorch)
VENV="$CODE/venv"
TORCH_PINS="${TORCH_PINS:-}"
STAMP="$VENV/.video-chain-$COMFY_REF-$(echo "$TORCH_PINS" | tr ' =' '_-')"
if [ ! -f "$STAMP" ]; then
    note "installing ComfyUI's Python packages"
    command -v uv >/dev/null 2>&1 || python3 -m pip install -q --root-user-action=ignore uv
    rm -rf "$VENV"
    uv venv -q --system-site-packages --python "$(command -v python3)" "$VENV"
    # PyTorch pinned (TORCH_PINS, also as constraints: nothing may pull another one in), the rest from the tag
    grep -viE '^(torch|torchvision|torchaudio)([<>=~! ].*)?$' "$COMFY/requirements.txt" > "$CODE/requirements.txt"
    printf '%s\n' $TORCH_PINS > "$CODE/constraints.txt"
    uv pip install -q --python "$VENV/bin/python" -c "$CODE/constraints.txt" -r "$CODE/constraints.txt" \
        -r "$CODE/requirements.txt"
    touch "$STAMP"
fi

# ---- SeedVR2 models (Comfy-Org/SeedVR2), checked against Hugging Face's SHA-256 once
fetch() {   # fetch <repo path> <models subfolder>, to the container disk
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
        7b|3b) fetch "diffusion_models/seedvr2_${m}_${PRECISION}.safetensors" diffusion_models ;;
        *) echo "unknown SeedVR2 size $m" >&2; exit 2 ;;
    esac
done
fetch vae/seedvr2_ema_vae_fp16.safetensors vae

# ---- CPU threads: as many as the container may use. A pod sees all of the host's CPUs (224 on an H100 host) but
# its cgroup allows only its share (cpu.max 2380000/100000: 23.8 CPUs); PyTorch then ran 112 threads per operation,
# the container was throttled, and a tensor conversion took 0.28 s instead of 0.007 s with one thread (measured
# 2026-11-13, during a 13 min single-core phase before the GPU started; whether this was its cause is not proven).
# Same results, no accuracy cost.
CPUS="${CPUS:-}"
if [ -z "$CPUS" ] && [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max || true
    if [ "${quota:-max}" != max ] && [ "${period:-0}" -gt 0 ]; then CPUS=$(( quota / period )); fi
fi
if [ -z "$CPUS" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then          # (cgroup v1)
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$quota" -gt 0 ] && [ "$period" -gt 0 ]; then CPUS=$(( quota / period )); fi
fi
CPUS="${CPUS:-$(nproc)}"
[ "$CPUS" -ge 1 ] || CPUS=1
note "CPU threads: $CPUS (the container's share of $(nproc))"
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS" NUMEXPR_NUM_THREADS="$CPUS" \
    TORCH_NUM_THREADS="$CPUS"

# ---- ComfyUI on 127.0.0.1:8188 (only through the SSH tunnel)
if [ -f "$IO/comfy.pid" ] && kill -0 "$(cat "$IO/comfy.pid")" 2>/dev/null; then
    note "ComfyUI is already running"
else
    note "starting ComfyUI"
    cd "$COMFY"
    TMPDIR="$IO/tmp" nohup "$VENV/bin/python" main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch $HIGHVRAM \
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
