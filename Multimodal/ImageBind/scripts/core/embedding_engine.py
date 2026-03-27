"""
ImageBind Embedding Extraction Engine
Core pipeline for extracting trimodal embeddings (Vision + Audio + Text).
"""
import torch
import numpy as np
import tempfile
import os
from tqdm import tqdm

from imagebind import data
from imagebind.models.imagebind_model import ModalityType

# Import from shared utils
from shared_utils.video_processing import compute_segments, save_video_segment
#from shared_utils.audio_processing import extract_audio_segment
from shared_utils.text_processing import (
    extract_transcript_with_timestamps,
    align_text_to_segments
)


class EmbeddingEngine:
    """
    Main engine for extracting trimodal ImageBind embeddings.
    
    Features:
    - Vision: Processes all frames and averages embeddings
    - Audio: Processes segment waveform
    - Text: Processes aligned transcript
    """
    
    def __init__(self, model, device, whisper_model):
        """
        Initialize embedding engine.
        
        Args:
            model: ImageBind model
            device: Device (cuda:0 or cpu)
            whisper_model: Whisper model for transcription
        """
        self.model = model
        self.device = device
        self.whisper_model = whisper_model
    
    def extract_segment_features(self, video_path, start_time, end_time,
                                 segment_text):

        with tempfile.TemporaryDirectory() as tmpdir:

            # === SAVE 2-SEC SEGMENT AS .mp4 ===
            segment_path = os.path.join(tmpdir, "segment.mp4")
            save_video_segment(video_path, start_time, end_time, segment_path)

            # === AUDIO EMBEDDING ===
            # Same .mp4 file — load_and_transform_audio_data ignores video track
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    [segment_path], self.device
                )
            }
            with torch.no_grad():
                audio_emb = self.model(inputs)[ModalityType.AUDIO].cpu().numpy()

            # === VISION EMBEDDING ===
            # load_and_transform_video_data: clips_per_video=1 (already 2sec),
            # decode_audio=False internally — audio track ignored automatically
            inputs = {
                ModalityType.VISION: data.load_and_transform_video_data(
                    [segment_path], self.device,
                    clip_duration=2,
                    clips_per_video=1,   # segment is already the right duration
                )
            }
            with torch.no_grad():
                vision_emb = self.model(inputs)[ModalityType.VISION].cpu().numpy()

            # === TEXT EMBEDDING === 
            if segment_text:
                inputs = {
                    ModalityType.TEXT: data.load_and_transform_text(
                        [segment_text], self.device
                    )
                }
                with torch.no_grad():
                    text_emb = self.model(inputs)[ModalityType.TEXT].cpu().numpy()
            else:
                text_emb = np.zeros((1, 1024), dtype=np.float32)

            # === COMBINED ===
            combined_emb = (vision_emb + audio_emb + text_emb) / 3

        return {
            'vision': vision_emb,
            'audio': audio_emb,
            'text': text_emb,
            'combined': combined_emb,
            'start': start_time,
            'end': end_time
        }
    def extract_video_features(self, video_path, window_size=2, stride=1, 
                            verbose=True):
        """
        Complete pipeline: Extract trimodal features from entire video.
        
        Args:
            video_path: Path to video file
            window_size: Segment duration (seconds)
            stride: Stride between segments (seconds)
            num_frames: Frames to sample per segment
            verbose: Print progress messages
        
        Returns:
            results: List of dicts with embeddings and metadata
            full_transcript: Complete video transcript
        """
        if verbose:
            print(f"Processing: {video_path}")
            print("=" * 60)
        
        # Step 1: Extract transcript
        if verbose:
            print("\nStep 1: Extracting transcript with Whisper...")
        
        transcript_segments, full_text = extract_transcript_with_timestamps(
            video_path, self.whisper_model
        )
        
        if verbose:
            print(f"✓ Extracted {len(transcript_segments)} text segments")
            print(f"✓ Total transcript length: {len(full_text)} characters")
        
        # Step 2: Segment video
        if verbose:
            print(f"\nStep 2: Segmenting video (window={window_size}s, stride={stride}s)...")
        
        segments, _, _ = compute_segments(video_path, window_size, stride)

        if verbose:
            print(f"✓ Created {len(segments)} segments")
        
        # Step 3: Align text to segments
        if verbose:
            print("\nStep 3: Aligning text to video segments...")
        
        segment_tuples = [(None, start, end) for start, end in segments]
        segment_texts = align_text_to_segments(transcript_segments, segment_tuples)
        
        assert len(segment_texts) == len(segments), (
            f"Text/vision segment mismatch: {len(segment_texts)} vs {len(segments)}"
        )
        text_count = sum(1 for t in segment_texts if t)
        if verbose:
            print(f"✓ {text_count}/{len(segment_texts)} segments have text")
        
        # Step 4: Extract embeddings
        if verbose:
            print(f"\nStep 4: Extracting trimodal embeddings...")
            print(f"Processing {len(segments)} segments (Vision + Audio + Text)...\n")
        
        results = []
        
        iterator = tqdm(enumerate(segments), total=len(segments), 
                       desc="Extracting features") if verbose else enumerate(segments)
        
        for idx, (start, end) in iterator:
            try:
                seg_data = self.extract_segment_features(
                    video_path, start, end,
                    segment_texts[idx]
                )
                
                results.append({
                    'start': start,
                    'end': end,
                    'text': segment_texts[idx],
                    'vision_emb': seg_data['vision'],
                    'audio_emb': seg_data['audio'],
                    'text_emb': seg_data['text'],
                    'combined_emb': seg_data['combined']
                })
                
            except Exception as e:
                if verbose:
                    print(f"\n Error at segment {idx} ({start:.1f}-{end:.1f}s): {e}")
                continue
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✓ Extraction complete!")
            print(f"  Processed: {len(results)}/{len(segments)} segments")
            print(f"  Each segment has:")
            print(f"    - Vision embedding: (1024,)")
            print(f"    - Audio embedding: (1024,)")
            print(f"    - Text embedding: (1024,)")
            print("=" * 60)
        
        return results, full_text
    
    def get_embedding_stats(self, results):
        """
        Compute statistics on extracted embeddings.
        
        Args:
            results: List of result dicts
        
        Returns:
            dict: Statistics on embeddings
        """
        vision_norms = [np.linalg.norm(r['vision_emb']) for r in results]
        audio_norms = [np.linalg.norm(r['audio_emb']) for r in results]
        text_norms = [np.linalg.norm(r['text_emb']) for r in results]
        
        return {
            'num_segments': len(results),
            'segments_with_text': sum(1 for r in results if r['text']),
            'vision_norm_mean': np.mean(vision_norms),
            'vision_norm_std': np.std(vision_norms),
            'audio_norm_mean': np.mean(audio_norms),
            'audio_norm_std': np.std(audio_norms),
            'text_norm_mean': np.mean(text_norms),
            'text_norm_std': np.std(text_norms)
        }


def create_engine(imagebind_model, device, whisper_model):
    """
    Convenience function to create an EmbeddingEngine.
    
    Args:
        imagebind_model: ImageBind model
        device: Device string
        whisper_model: Whisper model
    
    Returns:
        EmbeddingEngine instance
    """
    return EmbeddingEngine(imagebind_model, device, whisper_model)