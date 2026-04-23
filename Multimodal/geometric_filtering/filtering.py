"""
Highlight Filtering
===================
Selects highlight segments from a video using window-based geometric scoring
followed by temporal non-maximum suppression (NMS).

Public API
----------
get_highlights_by_window(unified, V, A, T, trust, times, texts, window=3, ...)
    → (results, scores)
"""

import numpy as np
import torch
import torch.nn.functional as F

from .scoring import compute_geometric_scores


# ---------------------------------------------------------------------------
# Temporal NMS
# ---------------------------------------------------------------------------

def _temporal_nms(
    scores: torch.Tensor,
    pair_times: np.ndarray,
    min_gap_s: float,
    top_k: int,
) -> list:
    """
    Greedy temporal NMS: keep high-scoring pairs that do not overlap or
    sit closer than ``min_gap_s`` to an already-kept pair.

    Args:
        scores     : geo scores per pair  [N-1]
        pair_times : timestamps [N-1, 2]  (start_s, end_s)
        min_gap_s  : minimum gap between kept pairs (seconds)
        top_k      : maximum number of pairs to keep

    Returns:
        List of kept pair indices (into pair_times / scores).
    """
    sorted_idx = scores.argsort(descending=True).tolist()
    kept = []
    for idx in sorted_idx:
        if len(kept) >= top_k:
            break
        s_new = (pair_times[idx][0] + pair_times[idx][1]) / 2.0
        too_close = any(
            (
                max(0, min(pair_times[idx][1], pair_times[k][1])
                    - max(pair_times[idx][0], pair_times[k][0])) > 0
            )
            or (
                abs(s_new - (pair_times[k][0] + pair_times[k][1]) / 2.0) < min_gap_s
            )
            for k in kept
        )
        if not too_close:
            kept.append(idx)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_highlights_by_window(
    unified: torch.Tensor,
    V: torch.Tensor,
    A: torch.Tensor,
    T: torch.Tensor,
    trust: torch.Tensor,
    times: np.ndarray,
    texts: np.ndarray,
    window: int = 3,
    top_n: int = 20,
    nms_gap_s: float = 10.0,
    top_k: int = 10,
    w_coherence: float = 0.50,
    w_novelty: float = 0.30,
    w_saliency: float = 0.20,
) -> tuple:
    """
    Select highlight segments from a video using window-based geometric scoring.

    Each consecutive pair of segments (i, i+1) is scored using three
    signals computed over a neighbourhood of ``window`` segments on each
    side (see :func:`~geometric_filtering.scoring.compute_geometric_scores`).
    Temporal NMS is then applied to return a diverse, non-redundant set of
    highlight windows.

    Args:
        unified    : unified per-segment embeddings  [N, D]
        V          : L2-normalised vision embeddings [N, D']
        A          : L2-normalised audio embeddings  [N, D']
        T          : L2-normalised text embeddings   [N, D']
        trust      : text-trust weights per segment  [N]  (range [0, 1])
        times      : segment timestamps              [N, 2]  (start_s, end_s)
        texts      : segment text labels             [N]
        window     : neighbourhood half-size for scoring (default 3)
        top_n      : number of top pairs to consider before NMS (default 20)
        nms_gap_s  : minimum temporal gap between kept highlights (default 10 s)
        top_k      : maximum highlights to return after NMS (default 10)
        w_coherence: weight for coherence signal (default 0.50)
        w_novelty  : weight for novelty signal   (default 0.30)
        w_saliency : weight for saliency signal  (default 0.20)

    Returns:
        results : list of dicts, one per selected highlight, each containing::

                    {
                        "rank"      : int,           # 1-based rank by score
                        "times"     : (float, float),# (start_s, end_s) of the pair
                        "text"      : str,           # combined text of the pair
                        "geo_score" : float,
                        "coherence" : float,
                        "novelty"   : float,
                        "saliency"  : float,
                    }

        scores  : np.ndarray [K] of geo scores for the returned highlights
                  (same order as *results*, sorted descending).
    """
    N = len(times)

    # --- compute scores ---
    _, geo_score, components = compute_geometric_scores(
        unified, V, A, T, trust,
        window=window,
        w_coherence=w_coherence,
        w_novelty=w_novelty,
        w_saliency=w_saliency,
    )

    # pair-level time spans: start of segment i → end of segment i+1
    pair_times = np.array(
        [[times[p][0], times[p + 1][1]] for p in range(N - 1)]
    )
    pair_texts = np.array(
        [f"{texts[p]} | {texts[p + 1]}" for p in range(N - 1)]
    )

    # --- temporal NMS ---
    kept_indices = _temporal_nms(geo_score, pair_times, nms_gap_s, top_k)

    # --- build results ---
    results = []
    scores_list = []
    for rank, idx in enumerate(kept_indices):
        score = geo_score[idx].item()
        results.append(
            {
                "rank": rank + 1,
                "times": (float(pair_times[idx][0]), float(pair_times[idx][1])),
                "text": str(pair_texts[idx]),
                "geo_score": score,
                "coherence": components["coherence"][idx].item(),
                "novelty": components["novelty"][idx].item(),
                "saliency": components["saliency"][idx].item(),
            }
        )
        scores_list.append(score)

    scores = np.array(scores_list, dtype=np.float32)
    return results, scores
