"""
download_dataset.py
Reusable helper functions for VAST-150K trimodal segment extraction.
"""

import os, json, subprocess, warnings
import numpy as np
import librosa
import cv2
from decord import VideoReader, cpu

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEGMENT_DURATION = 2
FPS_TARGET       = 2
AUDIO_SR         = 16000


def parse_time_string(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"index": 0, "segments": 0}


def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, "w") as f:
        json.dump(state, f)


def download_video(url, output_path):
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", output_path, url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60
        )
        return (result.returncode == 0
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0)
    except Exception:
        return False


def extract_frames(video_path, start):
    """Returns list of RGB uint8 arrays at native resolution."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        indices = [
            min(int((start + t) * fps), total_frames - 1)
            for t in np.linspace(0, SEGMENT_DURATION, FPS_TARGET, endpoint=False)
        ]
        frames = vr.get_batch(indices).asnumpy()
        return [frames[k] for k in range(len(indices))]
    except Exception:
        return None


def extract_audio(video_path, start):
    try:
        y, _ = librosa.load(video_path, sr=AUDIO_SR, offset=start,
                            duration=SEGMENT_DURATION, mono=True)
        target_len = AUDIO_SR * SEGMENT_DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y.astype(np.float32)
    except Exception:
        return None


def save_segment(output_dir, seg_id, frames, audio, text):
    """
    segments/
      seg_000000/
        frame_00.jpg
        frame_01.jpg
        audio.npy
        text.txt
    """
    seg_dir = os.path.join(output_dir, f"seg_{seg_id:06d}")
    os.makedirs(seg_dir, exist_ok=True)

    for k, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(seg_dir, f"frame_{k:02d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    np.save(os.path.join(seg_dir, "audio.npy"), audio)

    with open(os.path.join(seg_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text or "")


def count_segments_on_disk(output_dir):
    if not os.path.isdir(output_dir):
        return 0
    return sum(
        1 for d in os.listdir(output_dir)
        if d.startswith("seg_") and os.path.isdir(os.path.join(output_dir, d))
    )