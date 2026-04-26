"""
geometric_filtering
===================
Stage 5 — Highlight Detection, Phase 1: Geometric Filter

Highlights are found using three pure-math signals on multimodal embeddings:
  • Cross-modal coherence  — do vision / audio / text agree?
  • Local novelty          — does this moment differ from its neighbours?
  • Temporal saliency      — does this moment stand out from the whole video?

Scores are computed over a sliding window whose size controls the granularity:
  window_size=1  → evaluate every segment individually
  window_size=2  → evaluate consecutive pairs   (scene ≈ 2 segments)
  window_size=3  → evaluate consecutive triplets (scene ≈ 3 segments)
  window_size=k  → evaluate consecutive k-tuples (k >= 1)

Typical usage
-------------
    from geometric_filtering import get_highlights_by_window
    from geometric_filtering import save_geo_scores, save_top_segments
"""

from .scoring import compute_geometric_scores
from .filtering import get_highlights_by_window
from .utils import (
    save_geo_scores,
    save_top_segments,
    load_geo_scores,
    load_top_segments,
)

__all__ = [
    "compute_geometric_scores",
    "get_highlights_by_window",
    "save_geo_scores",
    "save_top_segments",
    "load_geo_scores",
    "load_top_segments",
]