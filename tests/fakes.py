"""Fake clip generator: synthetic ffmpeg clips, no model."""
import subprocess
import time
from pathlib import Path

from PIL import Image

from video_chain import chain

COLOURS = ["red", "green", "blue", "yellow", "magenta", "cyan"]


class FakeGenerator:
    """Clip k: N-1 grey frames, then one frame in a colour that depends on the call; optional sine sound."""

    def __init__(self, frames=17, audio=True, size=(64, 64), progress=None, step_delay=0.0, steps=4):
        self.frames, self.audio, self.size = frames, audio, size
        self.progress, self.step_delay, self.steps = progress, step_delay, steps
        self.calls = []

    def refine(self, draft, prompt, out, seed, size, frames, loras, first_frame=None, last_frame=None, sparse=None,
               strength="natural"):
        """Fake 2K refine: the draft scaled to size (frames and sound kept)."""
        self.refines = getattr(self, "refines", [])
        self.refines.append({"draft": Path(draft).name, "prompt": prompt, "size": tuple(size), "frames": frames,
                             "loras": loras, "first": first_frame and Path(first_frame).name,
                             "last": last_frame and Path(last_frame).name, "sparse": sparse,
                             "strength": strength})
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(draft), "-vf", f"scale={size[0]}:{size[1]},setsar=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)

    def seedvr(self, draft, out, size, method="seedvr2_7b", seed=42):
        """Fake SeedVR2 upscale: the draft scaled to size (frames and sound kept)."""
        self.seedvrs = getattr(self, "seedvrs", [])
        self.seedvrs.append({"draft": Path(draft).name, "size": tuple(size), "method": method})
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(draft), "-vf", f"scale={size[0]}:{size[1]},setsar=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)

    def generate(self, prompt, start_image, out_video, out_last_frame, seed, settings=None):
        k = len(self.calls)
        if self.progress is not None:
            for i in range(1, self.steps + 1):
                time.sleep(self.step_delay)
                self.progress(i, self.steps)
        s = settings or {}
        self.calls.append({"prompt": prompt, "start": Path(start_image).name if start_image else None, "seed": seed,
                           "settings": settings,
                           "start_rgb": Image.open(start_image).convert("RGB").getpixel((5, 5)) if start_image
                           else None,
                           "end_rgb": Image.open(s["end_image"]).convert("RGB").getpixel((5, 5))
                           if s.get("end_image") else None,
                           "guides": [{**g, "probe": chain.probe(Path(g["video"]))} for g in s.get("guides") or []]})
        w, h = self.size or (s.get("width", 64), s.get("height", 64))   # None: the real clip size
        colour = COLOURS[k % len(COLOURS)]
        n = self.frames
        args = ["-f", "lavfi", "-i", f"color=gray:s={w}x{h}:r=24:d={(n - 1) / 24:.6f}",
                "-f", "lavfi", "-i", f"color={colour}:s={w}x{h}:r=24:d={1 / 24:.6f}"]
        filt = "[0:v][1:v]concat=n=2:v=1:a=0[v]"
        maps = ["-map", "[v]"]
        if self.audio:
            args += ["-f", "lavfi", "-i", f"sine=f={300 + 100 * k}:d={n / 24:.6f}"]
            maps += ["-map", "2:a", "-c:a", "aac"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", filt, *maps, "-frames:v", str(n),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", str(out_video)], check=True)
        chain.last_frame(Path(out_video), Path(out_last_frame))


