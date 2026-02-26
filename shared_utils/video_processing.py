"""
Universal video processing utilities.
Used across ImageBind, VideoLLaMA2, CLAP, etc.
"""
from decord import VideoReader, cpu
import numpy as np
import subprocess
import os


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


def extract_frames(vr, start_time, end_time, fps, num_frames=8):
    """
    Extract uniformly sampled frames.
    
    Used by:
    - ImageBind (8 frames)
    - VideoLLaMA2 (8-16 frames)
    - Custom pipelines
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
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
    # Get duration
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    os.makedirs(output_dir, exist_ok=True)
    
    segments = []
    t = 0
    idx = 0
    
    while t + window_size <= duration:
        seg_path = os.path.join(output_dir, f"segment_{idx:04d}.mp4")
        
        cmd = ['ffmpeg', '-y', '-ss', str(t), '-i', video_path,
               '-t', str(window_size), '-c', 'copy', seg_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        segments.append((seg_path, t, t + window_size))
        t += stride
        idx += 1
    
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
    Thin wrapper around ffmpeg stream copy — used by EmbeddingEngine.
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', '1',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)