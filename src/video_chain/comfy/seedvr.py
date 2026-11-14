"""SeedVR2 upscale (video restoration): the workflow, the pieces a long clip goes in and ComfyGenerator.seedvr
(SeedvrMixin)."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .. import media
from ..chain import probe
from .errors import ComfyError
from .ffutil import ffmpeg, with_audio
from .wfb import (FPS, SEEDVR_SAVE_ID, SUBFOLDER, combo_options, loader_unet, loader_vae, save_video)

# ----------------------------------------------------------------------------- SeedVR2 upscale (video restoration)
# ComfyUI's core SeedVR2 nodes, as its blueprint "Upscale Video (SeedVR2 3B Int8)": the clip scaled to the export
# size (lanczos), padded, encoded with SeedVR2's VAE, restored in one step, decoded and colour-matched to the input.
# Unlike the refine it restores what is there instead of drawing it anew: measured on a 1.6 s face clip at 1080p
# (RTX 4090, ~33 s each) it changed the clip least of all (7B 4.1, 3B 4.7, refine "natural" 7.0), closed eyes
# stayed closed, 7B was the cleanest, 3B crisper but harder. Long clips are split in time (SeedVR2TemporalChunk,
# auto: the largest chunk that fits the free GPU memory) and crossfaded back together.
SEEDVR_NODES = ("SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2PostProcessing", "SeedVR2TemporalChunk",
                "SeedVR2TemporalMerge", "VAEEncodeTiled", "VAEDecodeTiled", "ImageScale", "LoadVideo",
                "GetVideoComponents", "CreateVideo", "SaveVideo")
SEEDVR_METHODS = {"seedvr2_7b": "7b", "seedvr2_3b": "3b"}       # export method -> model size
SEEDVR_DEFAULT_METHOD = "seedvr2_7b"
SEEDVR_OVERLAP = 2                         # latent frames (8 frames) crossfaded between the chunks of a long clip
SEEDVR_DOWNLOAD = "https://huggingface.co/Comfy-Org/SeedVR2 (diffusion_models/seedvr2_{}_int8_convrot.safetensors)"


SEEDVR_PRECISION_LOCAL = ("int8", "fp8", "fp16")   # 24 GB: the int8 copy fits with the frames (measured on a 4090)
SEEDVR_PRECISION_FULL = ("fp16", "int8", "fp8")    # 40 GB and more: the full-precision weights, nothing quantized
SEEDVR_FULL_VRAM = 40 * 2**30                      # from this much VRAM: fp16 weights and untiled VAE passes


def pick_seedvr_files(unets: list[str], vaes: list[str], size: str,
                      precision: tuple[str, ...] = SEEDVR_PRECISION_LOCAL) -> dict:
    """SeedVR2's model of this size ('7b' / '3b'; not the "sharp" variant; in the order of precision, then any;
    never mxfp8 / nvfp4, made for newer GPUs) and its VAE."""
    cands = [u for u in unets if f"seedvr2_{size}" in Path(u).name.lower() and "sharp" not in u.lower()
             and not any(k in u.lower() for k in ("mxfp8", "nvfp4"))]

    def order(u: str) -> int:
        return next((i for i, k in enumerate(precision) if k in u.lower()), 9)
    vae = next((v for v in vaes if "seedvr2" in Path(v).name.lower() and "vae" in Path(v).name.lower()), None)
    return {"unet": min(cands, key=order) if cands else None, "vae": vae}


SEEDVR_TILES = {"tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
SCALE_NODE = "VideoChainLanczosScale"      # comfy_nodes/video_chain_nodes/scale.py: ImageScale's lanczos on all cores


def build_seedvr_workflow(*, draft: str, width: int, height: int, unet: str, vae: str, prefix: str,
                          seed: int = 42, tiled: bool = True, fast_scale: bool = False) -> dict:
    """Upscale the uploaded clip (with its sound) to width x height with SeedVR2 (see SEEDVR_NODES).
    tiled: the VAE passes in 512 px tiles (as the blueprint; for 24 GB). Off (40 GB and more): the plain
    VAEEncode / VAEDecode, which run SeedVR2's VAE on whole frames, split in time only as far as the free VRAM
    needs, with its causal cache carried between the slices: exact, no tile seams, the GPU kept busy (ComfyUI
    falls back to tiles by itself on an out-of-memory).
    fast_scale: the lanczos resize by SCALE_NODE (the same frames, bit for bit, on all CPU cores) instead of
    ImageScale (one frame after another on one core)."""
    tiles = SEEDVR_TILES
    enc = ({"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0], **tiles}} if tiled
           else {"class_type": "VAEEncode", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0]}})
    dec = ({"class_type": "VAEDecodeTiled", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0], **tiles}} if tiled
           else {"class_type": "VAEDecode", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0]}})
    return {
        "load": {"class_type": "LoadVideo", "inputs": {"file": draft}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "resize": ({"class_type": SCALE_NODE, "inputs": {"image": ["parts", 0], "width": int(width),
                                                         "height": int(height)}} if fast_scale else
                   {"class_type": "ImageScale", "inputs": {"image": ["parts", 0], "upscale_method": "lanczos",
                                                           "width": int(width), "height": int(height),
                                                           "crop": "disabled"}}),
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}},
        "vae": loader_vae(vae),
        "unet": loader_unet(unet),
        "enc": enc,
        "chunk": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["enc", 0],
                                                                   "temporal_overlap": SEEDVR_OVERLAP,
                                                                   "chunking_mode": "auto"}},
        "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["unet", 0],
                                                                 "vae_conditioning": ["chunk", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["cond", 0],
                                                        "negative": ["cond", 1], "latent_image": ["chunk", 0],
                                                        "seed": int(seed), "steps": 1, "cfg": 1.0,
                                                        "sampler_name": "euler", "scheduler": "simple",
                                                        "denoise": 1.0}},
        "merge": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["sample", 0],
                                                                   "temporal_overlap": ["chunk", 1]}},
        "dec": dec,
        "post": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["dec", 0],
                                                                   "original_resized_images": ["resize", 0],
                                                                   "color_correction_method": "lab"}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["post", 0], "fps": FPS,
                                                          "audio": ["parts", 1]}},
        SEEDVR_SAVE_ID: save_video(prefix),
    }


SEEDVR_RAM_BUDGET = 12 * 2**30           # frame RAM for one SeedVR2 job when ComfyUI does not say what is free
SEEDVR_RAM_RESERVE = 10 * 2**30          # RAM left for the system, this app and the browser
SEEDVR_RAM_MIN = 4 * 2**30               # never less frame RAM than this per job (very short pieces otherwise)
SEEDVR_FRAME_COPIES = 7                  # copies of the clip ComfyUI holds at once (scaled, padded, decoded, matched,
                                         # the video; float32): a 10 s 2K clip as one job took 65 GB RAM + 36 GB swap
                                         # (about 5-6 copies besides the cached models) and systemd-oomd killed it
SEEDVR_PIECE_OVERLAP = 8                 # frames two neighbouring pieces share, crossfaded
# a light denoise before SeedVR2 (optional): mostly temporal (noise that changes from frame to frame), little
# spatial blur, so SeedVR2 does not sharpen the draft's fine noise into texture. Measured on synthetic noise at the
# drafts' level (about 1.5 grey levels frame to frame): 1.56 -> 0.66; strong noise (8.5) it leaves (7.8): hqdn3d
# takes large changes for motion, so real motion is not smeared either
SEEDVR_DENOISE = "hqdn3d=1:1:4:4"
SEEDVR_INTERMEDIATE_CRF = "6"            # x264 quality of the pieces cut for SeedVR2 and of the denoised clip (nearly lossless)
SEEDVR_JOIN_CRF = "12"                   # x264 quality of the joined clip (as VIDEO_CRF)




def seedvr_budget(ram_free: int | None) -> int:
    """The frame RAM one SeedVR2 job may use: what ComfyUI has free less SEEDVR_RAM_RESERVE (SEEDVR_RAM_BUDGET when
    it does not say; never below SEEDVR_RAM_MIN)."""
    if not ram_free:
        return SEEDVR_RAM_BUDGET
    return max(SEEDVR_RAM_MIN, int(ram_free) - SEEDVR_RAM_RESERVE)


def seedvr_piece_frames(width: int, height: int, budget: int | None = None) -> int:
    """How many frames of this size one SeedVR2 job gets within budget bytes (4n+1, at least 17)."""
    budget = SEEDVR_RAM_BUDGET if budget is None else int(budget)
    n = budget // (SEEDVR_FRAME_COPIES * int(width) * int(height) * 3 * 4)
    return max(17, 4 * ((int(n) - 1) // 4) + 1)


class SeedvrProgress:
    """One steady progress for a SeedVR2 upscale: ComfyUI reports each node of each piece on its own (tiled encode,
    the step, tiled decode: bars restarting many times). Here a piece counts 1 and its phases a third each: a phase
    starts when ComfyUI's counter goes back or changes its total. Reported as (done, total) in 1000ths, never back."""
    PHASES = 3

    def __init__(self, pieces: int, out=None):
        self.pieces, self.out = max(1, int(pieces)), out
        self.k, self.phase, self.last, self.best = 0, 0, None, 0

    def piece(self, k: int, note=None):
        self.k, self.phase, self.last = int(k), 0, None
        if note is not None:
            note(f"SeedVR2 piece {k + 1} of {self.pieces}" if self.pieces > 1 else "SeedVR2")
        self._send(0.0)

    def step(self, i: int, n: int):
        if not n:
            return
        if self.last is not None and (n != self.last[1] or i < self.last[0]):
            self.phase = min(self.phase + 1, self.PHASES - 1)
        self.last = (i, n)
        self._send((self.phase + min(1.0, i / n)) / self.PHASES)

    def finish(self):
        self.k, self.phase, self.last = self.pieces - 1, self.PHASES - 1, None
        self._send(1.0)

    def _send(self, sub: float):
        done = round(1000 * (self.k + min(1.0, max(0.0, sub))) / self.pieces)
        self.best = max(self.best, done)                 # never back
        if self.out is not None:
            self.out(self.best, 1000)


def seedvr_pieces(total: int, per: int, overlap: int = SEEDVR_PIECE_OVERLAP) -> list[tuple[int, int]]:
    """(first frame, frame count) of the pieces a clip of `total` frames is upscaled in: at most `per` frames each,
    neighbours sharing `overlap` frames; one piece when it fits."""
    total, per = int(total), int(per)
    if per <= overlap:
        raise ValueError(f"a piece of {per} frames cannot overlap its neighbour by {overlap}")
    if total <= per:
        return [(0, total)]
    out, start = [], 0
    while True:
        if start + per >= total:
            out.append((max(0, total - per), min(per, total)))   # (the last one full length, ending at the end)
            break
        out.append((start, per))
        start += per - overlap
    return out



def _denoise(src: Path, out: Path) -> None:
    """The clip with SEEDVR_DENOISE applied, nearly lossless, its sound copied."""
    ffmpeg(["-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-vf", SEEDVR_DENOISE, "-c:v", "libx264",
            "-crf", SEEDVR_INTERMEDIATE_CRF, "-pix_fmt", "yuv420p", "-c:a", "copy", out])


def _join_pieces(made: list[Path], pieces: list[tuple[int, int]], draft: Path, out: Path, total: int) -> None:
    """The upscaled pieces as one clip: each overlap crossfaded (ffmpeg xfade, frame-exact at 24 fps), the draft's
    own sound; checked to have the draft's frame count."""
    args, filt, last, length = [], [], None, 0
    for i, (f, (first, count)) in enumerate(zip(made, pieces)):
        args += ["-i", str(f)]
        filt.append(f"[{i}:v]settb=1/24,setpts=N,fps=24,format=yuv420p[p{i}]")
        if last is None:
            last, length = f"[p{i}]", count
            continue
        overlap = length - first                   # frames this piece shares with the clip so far
        if overlap <= 0:
            raise ComfyError("the upscaled pieces do not overlap, they cannot be crossfaded")
        filt.append(f"{last}[p{i}]xfade=transition=fade:duration={overlap / 24:.6f}:"
                    f"offset={(length - overlap) / 24:.6f}[x{i}]")
        last, length = f"[x{i}]", first + count
    args += ["-i", str(draft)]
    with media.part_file(out) as part:
        ffmpeg([*args, "-filter_complex", ";".join(filt), "-map", last, "-map", f"{len(made)}:a:0?",
                "-frames:v", int(total), "-c:v", "libx264", "-crf", SEEDVR_JOIN_CRF, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "256k", "-r", "24", part])
        got = probe(part)["frames"]
        if got != int(total):
            raise ComfyError(f"the upscaled pieces joined to {got} frames instead of {total}")



def seedvr_timing(size, unet, up_bytes, t_up, t_run, down_bytes, t_down, node_times: dict, wf: dict) -> str:
    """One log line: where a SeedVR2 job's time went (upload, ComfyUI by node type, download)."""
    def rate(b, t):
        return f"{b / 1e6:.1f} MB in {t:.1f} s ({b / 1e6 / t:.1f} MB/s)" if t > 0 else f"{b / 1e6:.1f} MB"
    by_type: dict[str, float] = {}
    for node, secs in node_times.items():
        kind = (wf.get(str(node)) or {}).get("class_type", str(node))
        by_type[kind] = by_type.get(kind, 0.0) + secs
    nodes = ", ".join(f"{k} {v:.1f} s" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]) if v >= 0.05)
    return (f"SeedVR2 {size[0]}x{size[1]} ({unet}): upload {rate(up_bytes, t_up)}; ComfyUI {t_run:.1f} s"
            + (f" [{nodes}]" if nodes else "") + f"; download {rate(down_bytes, t_down)}")


class SeedvrMixin:
    """ComfyGenerator.seedvr (needs the generator's upload / _run_workflow / _save / free ...)."""

    def seedvr(self, draft: Path, out: Path, size: tuple[int, int], method: str = SEEDVR_DEFAULT_METHOD,
               seed: int = 42, denoise: bool = False) -> None:
        """Upscale one finished clip (or video) to size with SeedVR2 (build_seedvr_workflow).
        The first SeedVR2 job unloads ComfyUI's other models (their RAM); a long clip goes in pieces that fit
        SEEDVR_RAM_BUDGET, crossfaded over SEEDVR_PIECE_OVERLAP frames, with the clip's own sound. Progress: one bar
        for the whole clip (SeedvrProgress) and a note per piece. denoise: SEEDVR_DENOISE on the clip first."""
        self._begin_job()
        if denoise:
            with media.scratch("denoise") as tmp:
                clean = Path(tmp) / "denoised.mp4"
                _denoise(Path(draft), clean)
                return self.seedvr(clean, out, size, method, seed)
        files = self._seedvr_files(method)
        if not self._last_seedvr:  # the job before was another model: its RAM goes to SeedVR2
            if self.free():
                self.last_hyperflow = False
        self._last_seedvr = True
        total = probe(Path(draft))["frames"]
        ram_free = self._ram_free()                    # (after the unload: what SeedVR2 can really use)
        pieces = seedvr_pieces(total, seedvr_piece_frames(*size, seedvr_budget(ram_free)))
        outer = self.progress
        track = SeedvrProgress(len(pieces), outer)
        self.progress = track.step                     # (ComfyUI's per-node bars become one bar for the whole clip)
        try:
            if len(pieces) == 1:
                track.piece(0, self.note)
                self._seedvr_job(files, Path(draft), Path(out), size, seed)
                if not self.seedvr_keep_cache:
                    self.free_cache()
                track.finish()
                return
            self._seedvr_pieces(files, Path(draft), Path(out), size, seed, pieces, total, track)
        finally:
            self.progress = outer

    def _ram_free(self) -> int:
        """The RAM ComfyUI has free now, in bytes (0: it does not say)."""
        try:
            return int((self._json("GET", "/system_stats").get("system") or {}).get("ram_free") or 0)
        except (ComfyError, ValueError, TypeError):
            return 0

    def _seedvr_pieces(self, files, draft, out, size, seed, pieces, total, track) -> None:
        with media.scratch("seedvr-pieces") as tmp:
            tmp, made = Path(tmp), []
            for i, (first, count) in enumerate(pieces):
                track.piece(i, self.note)
                part = tmp / f"in{i}.mp4"
                ffmpeg(["-i", draft, "-vf", f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS",
                        "-an", "-c:v", "libx264", "-crf", SEEDVR_INTERMEDIATE_CRF, "-pix_fmt", "yuv420p", part])
                done = tmp / f"out{i}.mp4"
                self._seedvr_job(files, part, done, size, seed)
                if not self.seedvr_keep_cache:
                    self.free_cache()                  # (ComfyUI keeps the last job's frames cached otherwise)
                made.append(done)
            self.note("joining the upscaled pieces")
            _join_pieces(made, pieces, Path(draft), Path(out), total)
            track.finish()

    def _seedvr_files(self, method: str) -> dict:
        """The SeedVR2 model and VAE ComfyUI lists for this method (clear errors when something is missing)."""
        for node in SEEDVR_NODES:
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI: SeedVR2 is in its core nodes)")
        model = SEEDVR_METHODS[method]
        unets = combo_options(self._object_info("UNETLoader").get("input", {}).get("required", {}).get("unet_name"))
        vaes = combo_options(self._object_info("VAELoader").get("input", {}).get("required", {}).get("vae_name"))
        files = pick_seedvr_files(unets, vaes, model, self.seedvr_precision)
        files["fast_scale"] = bool(self._object_info(SCALE_NODE))   # (video-chain's node, when ComfyUI has it)
        if not files["unet"]:
            raise ComfyError(f"ComfyUI lists no SeedVR2 {model.upper()} model in its diffusion_models: "
                             + SEEDVR_DOWNLOAD.format(model))
        if not files["vae"]:
            raise ComfyError("ComfyUI lists no SeedVR2 VAE (seedvr2_ema_vae_fp16.safetensors in models/vae, "
                             "https://huggingface.co/Comfy-Org/SeedVR2)")
        return files

    def _seedvr_job(self, files: dict, draft: Path, out: Path, size: tuple[int, int], seed: int) -> None:
        """One SeedVR2 job in ComfyUI: draft (with its sound, silence if none) upscaled to size into out."""
        uploads: list[tuple[str, Path]] = []
        with media.scratch("seedvr") as tmp:
            src = with_audio(Path(draft), Path(tmp) / "draft.mp4")
            clock = time.monotonic()
            try:
                name = self.upload(src)
                uploads.append((name, src))
                t_up = time.monotonic() - clock
                prefix = f"{SUBFOLDER}/seedvr_{uuid.uuid4().hex[:12]}"
                wf = build_seedvr_workflow(draft=name, width=size[0], height=size[1], unet=files["unet"],
                                           vae=files["vae"], prefix=prefix, seed=seed,
                                           tiled=self.seedvr_tiled, fast_scale=files.get("fast_scale", False))
                clock = time.monotonic()
                entry = self._run_workflow(wf)
                t_run = time.monotonic() - clock
                video = self._first_file((entry.get("outputs") or {}).get(SEEDVR_SAVE_ID))
                if video is None:
                    raise ComfyError("ComfyUI finished without the upscaled clip")
                clock = time.monotonic()
                self._save(video, Path(out))
                t_down = time.monotonic() - clock
                self.log(seedvr_timing(size, files["unet"], src.stat().st_size, t_up, t_run, Path(out).stat().st_size,
                                       t_down, self.node_times, wf))
                if self.cleanup:
                    self._remove(self.output_dir, video, Path(out))
            finally:
                self._drop_uploads(uploads)

