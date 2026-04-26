"""
filtering.py — Geometric Filter, Stage 5 Phase 1
=================================================
computes scores then selects the top-K highlights
using temporal non-maximum suppression (greedy, configurable min gap).

Public API
----------
    results, scores = get_highlights_by_window(
        unified, V, A, T, trust, times, texts,
        window_size=2,       # any integer >= 1 and <= N
        context_window=5,
        top_k=10,
        min_gap_s=2.0,       # min seconds between selected highlights
        w_coherence=0.5,
        w_novelty=0.3,
        w_saliency=0.2,
    )

Returns
-------
results : list[dict]  ranked highlights, each dict contains:
              rank, seg_idx, times [start_s, end_s], text,
              geo_score, coherence, novelty, saliency
scores  : dict
              'geo_score'   torch.Tensor [N]
              'coherence_n' torch.Tensor [N]
              'novelty_n'   torch.Tensor [N]
              'saliency_n'  torch.Tensor [N]
"""

import numpy as np
import torch
from .scoring import compute_geometric_scores


# ─────────────────────────────────────────────────────────────────────────────
# Temporal NMS
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_nms(geo_score: torch.Tensor,
                  times: np.ndarray,
                  top_k: int,
                  min_gap_s: float) -> list[int]:
    """
    Greedy temporal NMS: pick the highest-scoring segment, then suppress
    any segment whose midpoint is within `min_gap_s` seconds of it.

    Returns a list of selected segment indices (length ≤ top_k).
    """
    order      = geo_score.argsort(descending=True).tolist()
    mid_times  = [(times[i][0] + times[i][1]) / 2.0 for i in range(len(times))]
    selected   = []

    for idx in order:
        if len(selected) >= top_k:
            break
        t = mid_times[idx]
        # check against already-selected segments
        too_close = any(abs(t - mid_times[s]) < min_gap_s for s in selected)
        if not too_close:
            selected.append(idx)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_highlights_by_window(
    unified:        torch.Tensor,
    V:              torch.Tensor,
    A:              torch.Tensor,
    T:              torch.Tensor,
    trust:          torch.Tensor,
    times:          np.ndarray,     # [N, 2]  segment start/end in seconds
    texts:          np.ndarray,     # [N]     transcript per segment
    window_size:    int   = 3,      # any integer >= 1 and <= N
    context_window: int   = 5,
    top_k:          int   = 10,
    min_gap_s:      float = 2.0,
    w_coherence:    float = 0.5,
    w_novelty:      float = 0.3,
    w_saliency:     float = 0.2,
) -> tuple[list[dict], dict]:
    """
    Compute geometric scores then return the top-K highlights.

    Parameters
    ----------
    unified, V, A, T, trust : see scoring.compute_geometric_scores
    times          : numpy array [N, 2] — segment [start_s, end_s]
    texts          : numpy array [N]   — transcript text per segment
    window_size    : number of consecutive segments per scoring window
    context_window : neighbours on each side used as baseline
    top_k          : number of highlights to return
    min_gap_s      : minimum midpoint gap between highlights (temporal NMS)
    w_*            : signal weights (must sum to 1)

    Returns
    -------
    results : list[dict] sorted by rank
    scores  : dict of score tensors keyed by signal name
    """
    geo_score, components = compute_geometric_scores(
        unified, V, A, T, trust,
        window_size    = window_size,
        context_window = context_window,
        w_coherence    = w_coherence,
        w_novelty      = w_novelty,
        w_saliency     = w_saliency,
    )

    coherence_n = components["coherence_n"]
    novelty_n   = components["novelty_n"]
    saliency_n  = components["saliency_n"]

    selected_idx = _temporal_nms(geo_score, times, top_k, min_gap_s)

    results = []
    for rank, seg_idx in enumerate(selected_idx, start=1):
        results.append({
            "rank":      rank,
            "seg_idx":   seg_idx,
            "times":     times[seg_idx].tolist(),   # [start_s, end_s]
            "text":      str(texts[seg_idx]),
            "geo_score": geo_score[seg_idx].item(),
            "coherence": coherence_n[seg_idx].item(),
            "novelty":   novelty_n[seg_idx].item(),
            "saliency":  saliency_n[seg_idx].item(),
        })

    scores = {
        "geo_score":   geo_score,
        "coherence_n": coherence_n,
        "novelty_n":   novelty_n,
        "saliency_n":  saliency_n,
    }

    return results, scores