"""
Full Pipeline Orchestrator: Stages 1-4

Coordinates the complete multimodal short-format generation pipeline:
  - Stage 2+3: Text preprocessing (from video)
  - Stage 1:   Trimodal embedding extraction (ImageBind)
  - Stage 4:   Confidence-gated late fusion

Single entry point for deployment. Handles all I/O via io_utils.
"""

import json
import os
from typing import Any

import torch
import numpy as np

from Multimodal.Text_Handler import TextProducer, run_preprocessing
from .model_loader import quick_load_all
from .embedding_engine import create_engine
from .confidence_gate import ConfidenceGate
from ..utils.io_utils import (
    save_segment_data,
    save_trimodal_embeddings,
    save_unified_embeddings,
)


def run_full_pipeline(
    video_path: str,
    output_dir: str,
    window_size: float = 2.0,
    stride: float = 1.0,
    whisper_size: str = "base",
    stages: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute complete pipeline: Stage 2+3 → Stage 1 → Stage 4.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for outputs (created if needed)
        window_size: Audio/video window length in seconds (default 2.0)
        stride: Window stride in seconds (default 1.0)
        whisper_size: Whisper model size ('base', 'small', etc.)
        stages: List of stages to run, e.g. ['2+3', '1', '4']. Default: all
        verbose: Print progress messages
    
    Returns:
        dict with keys:
            - segment_data: Stage 2+3 output (dict[str(i)] -> {text, trust, source, start, end})
            - trimodal: Stage 1 output (dict with vision, audio, text embeddings + metadata)
            - unified: Stage 4 output (dict with unified embedding + weights)
            - paths: dict with saved file paths
    
    Raises:
        ValueError: If video_path doesn't exist or pipeline fails
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract video stem for output naming
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    
    if stages is None:
        stages = ["2+3", "1", "4"]
    
    results = {"segment_data": None, "trimodal": None, "unified": None, "paths": {}}
    

    # LOAD MODELS (shared across all stages)
    if verbose:
        print("\n" + "="*70)
        print("LOADING MODELS")
        print("="*70)
    
    models = quick_load_all(whisper_size=whisper_size)
    imagebind_mod = models["imagebind"]
    blip_processor = models["blip_processor"]
    blip_model = models["blip"]
    wat_model = models["whisper_at"]
    parse_at_label = models["parse_at_label"]
    device = models["device"]
    
    # STAGE 2+3: TEXT PREPROCESSING
    if "2+3" in stages:
        if verbose:
            print("\n" + "="*70)
            print("STAGE 2+3: VISION + AUDIO + TEXT PREPROCESSING")
            print("="*70)
        
        text_producer = TextProducer(
            wat_model=wat_model,
            blip_model=blip_model,
            blip_processor=blip_processor,
            parse_at_label_fn=parse_at_label,
            device=device,
        )
        
        segment_data = run_preprocessing(
            video_path,
            text_producer,
            window_size=window_size,
            stride=stride,
            verbose=verbose,
        )
        
        results["segment_data"] = segment_data
        
        # Save Stage 2+3
        texts_path = save_segment_data(segment_data, output_dir, video_stem, verbose=verbose)
        results["paths"]["segment_data"] = texts_path
        
        if verbose:
            source_counts = {}
            for item in segment_data.values():
                source = item["source"]
                source_counts[source] = source_counts.get(source, 0) + 1
            print(f"  Sources breakdown: {source_counts}\n")
    else:
        # Load segment_data from disk if skipping Stage 2+3
        texts_path = os.path.join(output_dir, f"{video_stem}_segment_texts.json")
        if os.path.exists(texts_path):
            results["segment_data"] = load_segment_data(texts_path)
    
    # STAGE 1: TRIMODAL EMBEDDING EXTRACTION
    if "1" in stages:
        if verbose:
            print("\n" + "="*70)
            print("STAGE 1: TRIMODAL EMBEDDING EXTRACTION")
            print("="*70)
        
        if results["segment_data"] is None:
            raise ValueError("Segment data required for Stage 1. Run Stage 2+3 first or load from disk.")
        
        engine = create_engine(imagebind_mod, device)
        
        trimodal_results = engine.extract_video_features(
            video_path,
            results["segment_data"],
            verbose=verbose,
        )
        
        results["trimodal"] = trimodal_results
        
        # Save Stage 1
        emb_path = save_trimodal_embeddings(trimodal_results, output_dir, video_stem, verbose=verbose)
        results["paths"]["trimodal"] = emb_path
    else:
        # Load trimodal embeddings from disk if skipping Stage 1
        emb_path = os.path.join(output_dir, f"{video_stem}_embeddings_final.npz")
        if os.path.exists(emb_path):
            data = np.load(emb_path, allow_pickle=True)
            results["trimodal"] = {key: data[key] for key in data.files}
    
    # STAGE 4: CONFIDENCE-GATED LATE FUSION
    if "4" in stages:
        if verbose:
            print("\n" + "="*70)
            print("STAGE 4: CONFIDENCE-GATED LATE FUSION")
            print("="*70)
        
        if results["trimodal"] is None:
            raise ValueError("Trimodal embeddings required for Stage 4. Run Stage 1 first or load from disk.")
        
        # Convert to tensors
        V = torch.tensor(results["trimodal"]["vision"], dtype=torch.float32)
        A = torch.tensor(results["trimodal"]["audio"], dtype=torch.float32)
        T = torch.tensor(results["trimodal"]["text"], dtype=torch.float32)
        trust_tensor = torch.tensor(results["trimodal"]["text_trust"], dtype=torch.float32)
        
        # Apply ConfidenceGate
        gate = ConfidenceGate(input_dim=1024, proj_dim=512).eval()
        
        with torch.no_grad():
            unified, weights = gate(V, A, T, trust_tensor)
        
        if verbose:
            print(f"Unified: {unified.shape}  norm={unified.norm(dim=-1).mean():.4f}")
            for name, w in weights.items():
                print(f"  {name:<8}: mean={w.mean():.3f} ± {w.std():.3f}")
            print("✓ Stage 4 complete — unified embeddings ready\n")
        
        # Save Stage 4
        unified_path = save_unified_embeddings(
            unified,
            weights,
            results["trimodal"],
            output_dir,
            video_stem,
            verbose=verbose,
        )
        
        results["unified"] = {
            "unified": unified.cpu().numpy(),
            "weights": {k: v.cpu().numpy() for k, v in weights.items()},
        }
        results["paths"]["unified"] = unified_path
    
    # SUMMARY
    if verbose:
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print(f"Saved files:")
        for stage, path in results["paths"].items():
            print(f"  [{stage}] {os.path.basename(path)}")
    
    return results


def load_segment_data(json_path: str) -> dict:
    """Load segment_data from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)
