"""
Universal audio processing utilities.
Used across ImageBind, CLAP, Whisper pipelines.
"""
import librosa
import soundfile as sf
import subprocess
import os
import numpy as np


def extract_audio_from_video(video_path, sr=16000):
    """
    Extract full audio track from video.
    
    Used by:
    - Whisper transcription
    - CLAP embedding
    - Audio analysis
    """
    audio_file = video_path.replace('.mp4', '.wav')
    
    if not os.path.exists(audio_file):
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sr), '-ac', '1',
            audio_file, '-y'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return audio_file


def extract_audio_segment(video_path, start_time, end_time, sr=16000):
    """
    Extract audio segment.
    
    Used by:
    - ImageBind per-segment extraction
    - CLAP per-segment extraction
    - Audio feature analysis
    """
    audio_file = extract_audio_from_video(video_path, sr)
    
    audio, _ = librosa.load(
        audio_file,
        sr=sr,
        offset=start_time,
        duration=end_time - start_time
    )
    
    return audio


def compute_audio_features(audio, sr=16000):
    """
    Compute basic audio features (RMS, spectral centroid, ZCR).
    
    Used by:
    - Audio-based scoring
    - Highlight detection
    - Audio analysis
    """
    return {
        'rms': librosa.feature.rms(y=audio)[0].mean(),
        'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean(),
        'zcr': librosa.feature.zero_crossing_rate(audio)[0].mean(),
        'duration': len(audio) / sr
    }