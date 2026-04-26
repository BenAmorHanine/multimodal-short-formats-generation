"""
I/O Utilities for Pipeline

Centralized save/load functions for:
- Stage 2+3: segment_data (JSON)
- Stage 1: trimodal embeddings (NPZ)
- Stage 4: unified embeddings with weights (NPZ)
"""

import json
import numpy as np
import os
from pathlib import Path


def save_segment_data(segment_data: dict, output_dir: str, video_stem: str, verbose: bool = True) -> str:
    """
    Save preprocessed segment data from Stage 2+3.
    
    Args:
        segment_data: dict[str(index)] -> {text, trust, source, start, end}
        output_dir: Directory to save to
        video_stem: Video filename without extension
        verbose: Print confirmation message
    
    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_stem}_segment_texts.json")
    
    with open(output_path, "w") as f:
        json.dump(segment_data, f, indent=2)
    
    if verbose:
        print(f"✓ Saved {len(segment_data)} segments → {output_path}")
    
    return output_path


def save_trimodal_embeddings(
    results: dict,
    output_dir: str,
    video_stem: str,
    verbose: bool = True
) -> str:
    """
    Save Stage 1 ImageBind embeddings.
    
    Args:
        results: dict from EmbeddingEngine.extract_video_features()
        output_dir: Directory to save to
        video_stem: Video filename without extension
        verbose: Print confirmation message
    
    Returns:
        Path to saved NPZ file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_stem}_embeddings_final.npz")
    
    np.savez(
        output_path,
        vision=results["vision"],
        audio=results["audio"],
        text=results["text"],
        times=results["times"],
        raw_text=results["raw_text"],
        text_trust=results["text_trust"],
        errors=np.array(results["errors"], dtype=np.int32),
    )
    
    if verbose:
        print(f"✓ Saved trimodal embeddings → {output_path}")
        print(f"  Shapes: V={results['vision'].shape}  A={results['audio'].shape}  T={results['text'].shape}")
    
    return output_path


def save_unified_embeddings(
    unified_tensor,
    weights: dict,
    results: dict,
    output_dir: str,
    video_stem: str,
    verbose: bool = True
) -> str:
    """
    Save Stage 4 unified embeddings with per-modality weights.
    
    Args:
        unified_tensor: [N, 512] fused embeddings from ConfidenceGate
        weights: dict with 'vision', 'audio', 'text' weight tensors
        results: dict from EmbeddingEngine.extract_video_features() (for metadata)
        output_dir: Directory to save to
        video_stem: Video filename without extension
        verbose: Print confirmation message
    
    Returns:
        Path to saved NPZ file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_stem}_unified_final.npz")
    
    np.savez(
        output_path,
        unified=unified_tensor.cpu().numpy() if hasattr(unified_tensor, 'cpu') else unified_tensor,
        times=results["times"],
        raw_text=results["raw_text"],
        text_trust=results["text_trust"],
        weights_vision=weights["vision"].cpu().numpy() if hasattr(weights["vision"], 'cpu') else weights["vision"],
        weights_audio=weights["audio"].cpu().numpy() if hasattr(weights["audio"], 'cpu') else weights["audio"],
        weights_text=weights["text"].cpu().numpy() if hasattr(weights["text"], 'cpu') else weights["text"],
    )
    
    if verbose:
        print(f"✓ Saved unified embeddings → {output_path}")
        print(f"  Shape: {unified_tensor.shape}")
    
    return output_path


def load_segment_data(json_path: str) -> dict:
    """Load segment_data from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_trimodal_embeddings(npz_path: str) -> dict:
    """Load trimodal embeddings from NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_unified_embeddings(npz_path: str) -> dict:
    """Load unified embeddings with weights from NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_features(emb_path):
    """Loads the NPZ file back into a dictionary. [Legacy compatibility]"""
    return np.load(emb_path, allow_pickle=True)


def save_features(results, full_transcript, output_dir, video_name):
    """Legacy save function for backward compatibility."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save MetaData
    meta_path = os.path.join(output_dir, f"{video_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({"transcript": full_transcript, "segments": len(results)}, f)

    # Save Embeddings (The heavy part)
    emb_path = os.path.join(output_dir, f"{video_name}_embeddings.npz")
    np.savez_compressed(
        emb_path,
        vision=np.array([r['vision_emb'] for r in results]),
        audio=np.array([r['audio_emb'] for r in results]),
        text=np.array([r['text_emb'] for r in results]),
        times=np.array([(r['start'], r['end']) for r in results])
    )
    print(f"✓ Saved features to {output_dir}")