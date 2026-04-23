"""
Video Assembly
==============
Assembles a final short video by extracting clips at given timestamps from a
source video and stitching them together with optional crossfade transitions
and caption overlays.

Public API
----------
assemble_video_from_segments(video_path, segment_times, output_path, ...)
    → output_path (str)
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_duration(path: str) -> float:
    """Return the duration of a video/audio file in seconds via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def _extract_clip(
    video_path: str,
    start_s: float,
    end_s: float,
    out_path: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Extract, crop-to-portrait, rescale and re-encode a single clip."""
    dur = end_s - start_s
    vf = f"crop=ih*{width}/{height}:ih,scale={width}:{height}:flags=lanczos,fps={fps}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", video_path,
        "-t", str(dur),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def _add_caption(
    clip_path: str,
    caption: str,
    out_path: str,
    width: int,
    height: int,
) -> None:
    """Burn a caption subtitle into the bottom of a clip."""
    safe = caption.replace("'", "").replace(":", " ").replace("%", "pct")[:80]
    fs = max(24, width // 32)
    mg = height // 20
    vf = (
        f"drawtext=text='{safe}':fontsize={fs}:fontcolor=white"
        f":bordercolor=black:borderw=3:x=(w-text_w)/2:y=h-text_h-{mg}"
        f":box=1:boxcolor=black@0.5:boxborderw=10"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        out_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def _assemble_xfade(
    clip_paths: List[str],
    fade_dur: float,
    output_path: str,
) -> None:
    """Concatenate clips with xfade/acrossfade transitions."""
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], output_path)
        return

    durations = [_get_duration(p) for p in clip_paths]
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    fv, fa = [], []
    pv, pa = "[0:v]", "[0:a]"
    offset = 0.0
    for i in range(1, len(clip_paths)):
        offset += durations[i - 1] - fade_dur
        is_last = i == len(clip_paths) - 1
        ov = "[vout]" if is_last else f"[v{i}]"
        oa = "[aout]" if is_last else f"[a{i}]"
        fv.append(
            f"{pv}[{i}:v]xfade=transition=fade"
            f":duration={fade_dur}:offset={offset:.3f}{ov}"
        )
        fa.append(f"{pa}[{i}:a]acrossfade=d={fade_dur}{oa}")
        pv, pa = ov, oa

    fc = ";".join(fv + fa)
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", fc,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
            "-loglevel", "error",
        ]
    )
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_video_from_segments(
    video_path: str,
    segment_times: Union[Sequence[Tuple[float, float]], np.ndarray],
    output_path: str,
    fade_dur: float = 0.5,
    resolution: str = "1080x1920",
    fps: int = 30,
    add_captions: bool = False,
    captions: Optional[List[str]] = None,
    target_duration: Optional[float] = None,
) -> str:
    """
    Assemble a short video by concatenating clips extracted at the given
    timestamps from a source video.

    Clips are cropped to portrait aspect ratio, rescaled to ``resolution``,
    and joined with optional crossfade transitions.  Captions can be burned
    into the bottom of each clip.

    Args:
        video_path       : Path to the source video file.
        segment_times    : Sequence of (start_s, end_s) tuples defining which
                           portions of the source video to use.  Clips are
                           assembled in the order they appear in this list.
        output_path      : Destination path for the assembled video (MP4).
        fade_dur         : Duration of the crossfade transition between clips,
                           in seconds (default 0.5).
        resolution       : Output resolution as ``"WIDTHxHEIGHT"``
                           (default ``"1080x1920"`` — portrait 9:16).
        fps              : Output frame rate (default 30).
        add_captions     : If ``True``, burn text captions into each clip.
        captions         : List of caption strings, one per segment.  Required
                           when ``add_captions=True``.
        target_duration  : If provided, clips are trimmed so that the total
                           assembled duration does not exceed this value
                           (seconds).

    Returns:
        output_path : Absolute path to the assembled video file.

    Raises:
        ValueError  : If ``add_captions`` is True but ``captions`` is not
                      provided or has the wrong length.
        subprocess.CalledProcessError : If ffmpeg/ffprobe fails.
    """
    if add_captions:
        if captions is None:
            raise ValueError(
                "`captions` must be provided when `add_captions=True`."
            )
        if len(captions) != len(segment_times):
            raise ValueError(
                f"Length of `captions` ({len(captions)}) must match "
                f"`segment_times` ({len(segment_times)})."
            )

    out_w, out_h = map(int, resolution.split("x"))

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: List[str] = []
        total_dur = 0.0

        for rank, (start_s, end_s) in enumerate(segment_times):
            start_s = float(start_s)
            end_s = float(end_s)

            if target_duration is not None:
                remaining = target_duration - total_dur
                if remaining <= 0:
                    break
                end_s = min(end_s, start_s + remaining)

            actual_dur = end_s - start_s
            raw_clip = os.path.join(tmpdir, f"clip_{rank + 1:02d}.mp4")
            _extract_clip(video_path, start_s, end_s, raw_clip, out_w, out_h, fps)

            if add_captions and captions is not None:
                cap_clip = os.path.join(tmpdir, f"clip_{rank + 1:02d}_cap.mp4")
                caption_text = str(captions[rank]).split(" | ")[0]
                _add_caption(raw_clip, caption_text, cap_clip, out_w, out_h)
                clip_paths.append(cap_clip)
            else:
                clip_paths.append(raw_clip)

            total_dur += actual_dur

        _assemble_xfade(clip_paths, fade_dur, output_path)

    return output_path
