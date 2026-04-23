"""
geometric_filtering
===================
Deployment module for geometric highlight detection and video assembly.

Extracted from:
- notebooks/Geometric_Highlight_Detection.ipynb
- notebooks/Highlight_Pipeline_Geometric.ipynb

Public API
----------
Three main functions cover the full highlight-to-short pipeline:

1. ``compute_geometric_scores``
   Computes pairwise geometric scores (coherence, novelty, saliency) from
   trimodal ImageBind embeddings.
   Returns: (unified, geo_score [N-1], components dict)

2. ``get_highlights_by_window``
   Selects highlight segments using window-based geometric scoring and
   temporal NMS.
   Returns: (results list, scores array)

3. ``assemble_video_from_segments``
   Concatenates video clips extracted at given timestamps into a short,
   with optional crossfade transitions and caption overlays.
   Returns: output_path (str)

Example usage::

    from Multimodal.geometric_filtering import (
        compute_geometric_scores,
        get_highlights_by_window,
        assemble_video_from_segments,
    )

    # --- Step 1: score all segment pairs ---
    unified, geo_score, components = compute_geometric_scores(
        unified, V, A, T, trust, window=3
    )

    # --- Step 2: select highlights ---
    results, scores = get_highlights_by_window(
        unified, V, A, T, trust, times, texts, window=3
    )

    # --- Step 3: assemble final video ---
    segment_times = [r["times"] for r in results]
    captions      = [r["text"]  for r in results]
    output = assemble_video_from_segments(
        video_path, segment_times, "output_short.mp4",
        add_captions=True, captions=captions
    )
"""

from .scoring import compute_geometric_scores
from .filtering import get_highlights_by_window
from .assembly import assemble_video_from_segments

__all__ = [
    "compute_geometric_scores",
    "get_highlights_by_window",
    "assemble_video_from_segments",
]
