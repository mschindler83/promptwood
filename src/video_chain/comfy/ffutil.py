"""ffmpeg helpers of the clip pipelines (refine, SeedVR2)."""
from __future__ import annotations

from pathlib import Path

from .. import media
from ..chain import probe
from .errors import ComfyError

FFMPEG_TIMEOUT = 3600                  # seconds an ffmpeg job on a clip may take


def ffmpeg(args: list, timeout: float = FFMPEG_TIMEOUT) -> None:
    """Run ffmpeg quietly; a failure (with ffmpeg's own reason), a missing ffmpeg or a hang is a ComfyError."""
    try:
        media.ffmpeg(["-y", *map(str, args)], timeout=timeout)
    except media.MediaError as e:
        raise ComfyError(str(e)) from e


def with_audio(video: Path, out: Path) -> Path:
    """The clip with a sound track (silence if it has none): the refine encodes its sound into the AV latent."""
    if probe(video)["audio"]:
        return video
    ffmpeg(["-i", video, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-shortest", out])
    return out
