"""
Universal video processing utilities.
Used across ImageBind, VideoLLaMA2, CLAP, etc.
"""
from decord import VideoReader, cpu
import numpy as np
import subprocess
import os
import json

# KEPT for backwards compatibility but now delegates to compute_segments
def segment_video(video_path, window_size=2, stride=1):
    """
    Segment video into overlapping windows.
    
    Used by:
    - ImageBind embedding extraction
    - VideoLLaMA2 inference
    - Any future video processing
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    
    segments = []
    t = 0
    while t + window_size <= duration:
        segments.append((t, t + window_size))
        t += stride
    
    return segments, fps, vr

   
def compute_segments(video_path, window_size=2, stride=1):
    """
    replaces segment_video()
    Single source of truth for segment boundaries.
    ALL modalities must use these exact (t_start, t_end) tuples.
 
    Uses ffprobe duration + fps (stable across calls) instead of
    decord avg_fps (non-deterministic).
 
    Used by:
    - ImageBind embedding extraction
    - VideoLLaMA2 inference
    - Audio segmentation
    - Text alignment
    - Any future video processing
    """
    info = get_video_info(video_path)
    fps = info['fps']
    duration = info['duration']
 
    segments = []
    t = 0.0
    while t + window_size <= duration:
        # CHANGE 2: round() prevents floating-point drift accumulating
        # over hundreds of segments (e.g. t=0.9999999 instead of 1.0).
        segments.append((round(t, 6), round(t + window_size, 6)))
        t += stride
 
    return segments, fps, duration


def extract_frames(vr, start_time, end_time, fps, num_frames=8):
    """
    Extract uniformly sampled frames.
    Used by:
    - ImageBind (8 frames)
    - VideoLLaMA2 (8-16 frames)
    - Custom pipelines
    """
    
    start_frame = round(start_time * fps) # int() truncates != round() gives the nearest frame: Prevents the same float time from mapping to different frame
    end_frame = round(end_time * fps)
    
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()
    
    return frames


def segment_video_ffmpeg(video_path, output_dir="segments", window_size=2, stride=1):
    """
    Create actual video clips using FFmpeg.
 
    Used by:
    - Final highlight extraction
    - Clip generation
    """
    # CHANGE 4: Use compute_segments() so this function uses the same
    # duration source as all other modalities (was: ffprobe format duration,
    # which differs from stream duration used in get_video_info).
    segments_bounds, _, _ = compute_segments(video_path, window_size, stride)
 
    os.makedirs(output_dir, exist_ok=True)
 
    segments = []
    for idx, (t, t_end) in enumerate(segments_bounds):
        seg_path = os.path.join(output_dir, f"segment_{idx:04d}.mp4")
 
        # CHANGE 5: Replace '-c copy' with explicit re-encode.
        # '-c copy' cuts at the nearest keyframe (not the exact timestamp),
        # so clip content varies depending on keyframe placement.
        # Re-encoding guarantees frame-accurate cuts every time.
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(t),
            '-i', video_path,
            '-t', str(window_size),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-avoid_negative_ts', '1',
            seg_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
        segments.append((seg_path, t, t_end))
 
    return segments


def get_video_info(video_path):
    """
    Get video metadata (duration, fps, resolution).
    
    Used everywhere for preprocessing checks.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    
    stream = data['streams'][0]
    fps_parts = stream['r_frame_rate'].split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1])
    
    return {
        'width': stream['width'],
        'height': stream['height'],
        'fps': fps,
        'duration': float(stream.get('duration', 0))
    }
    
    
def save_video_segment(video_path: str, start_time: float,
                       end_time: float, output_path: str) -> None:
    """
    Save a single video segment as .mp4.
    Thin wrapper around ffmpeg stream copy  used by EmbeddingEngine.
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'libx264', '-c:a', 'aac',
        '-avoid_negative_ts', '1',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)