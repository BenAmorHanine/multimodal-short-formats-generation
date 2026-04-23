"""
Geometric Scoring
=================
Computes pairwise geometric signals (coherence, novelty, saliency) from
trimodal ImageBind embeddings and returns a combined geo-score.

Public API
----------
compute_geometric_scores(unified, V, A, T, trust, window=3)
    → (unified, geo_score, components)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal signal functions
# ---------------------------------------------------------------------------

def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _compute_pairwise_coherence(
    V: torch.Tensor,
    A: torch.Tensor,
    T: torch.Tensor,
    trust: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Score each consecutive pair (i, i+1) by the average cross-modal
    coherence relative to their joint neighbourhood.

    Args:
        V, A, T : normalised per-modality embeddings  [N, D]
        trust   : text-trust weights                  [N]
        window  : number of neighbours on each side

    Returns:
        pair_scores [N-1]
    """
    N = V.shape[0]
    pair_scores = torch.zeros(N - 1)

    def _seg_coh(idx: int) -> torch.Tensor:
        w = trust[idx]
        va = (V[idx] * A[idx]).sum()
        vt = (V[idx] * T[idx]).sum() * w
        at = (A[idx] * T[idx]).sum() * w
        return (va + vt + at) / (1.0 + 2.0 * w + 1e-8)

    for i in range(N - 1):
        j = i + 1
        lo = max(0, i - window)
        hi = min(N, j + window + 1)
        ctx = [k for k in range(lo, hi) if k != i and k != j]
        ci = _seg_coh(i)
        cj = _seg_coh(j)
        cc = (
            torch.stack([_seg_coh(k) for k in ctx]).mean()
            if ctx
            else torch.tensor(0.0)
        )
        pair_scores[i] = ((ci + cj) / 2.0) - cc

    return pair_scores


def _compute_pairwise_novelty(
    unified: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Score each pair by how much the two segments diverge from each other
    relative to their left/right neighbourhood context.

    Args:
        unified : unified embeddings [N, D]
        window  : neighbourhood half-size

    Returns:
        pair_scores [N-1]
    """
    N = len(unified)
    pair_scores = torch.zeros(N - 1)

    for i in range(N - 1):
        j = i + 1
        lc = unified[max(0, i - window) : i]
        rc = unified[j + 1 : min(N, j + window + 1)]
        ci = F.normalize(
            (lc.mean(dim=0) if len(lc) > 0 else unified[i]).unsqueeze(0), dim=-1
        ).squeeze(0)
        cj = F.normalize(
            (rc.mean(dim=0) if len(rc) > 0 else unified[j]).unsqueeze(0), dim=-1
        ).squeeze(0)
        si = F.normalize(unified[i].unsqueeze(0), dim=-1).squeeze(0)
        sj = F.normalize(unified[j].unsqueeze(0), dim=-1).squeeze(0)
        pair_scores[i] = 0.6 * (1.0 - (si * sj).sum()) + 0.4 * (1.0 - (ci * cj).sum())

    return pair_scores


def _compute_pairwise_saliency(
    unified: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Score each pair by how much it deviates from both the global video mean
    and its local neighbourhood mean.

    Args:
        unified : unified embeddings [N, D]
        window  : neighbourhood half-size

    Returns:
        pair_scores [N-1]
    """
    N = len(unified)
    gm = F.normalize(unified.mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)
    pair_scores = torch.zeros(N - 1)

    for i in range(N - 1):
        j = i + 1
        lo = max(0, i - window)
        hi = min(N, j + window + 1)
        ctx = [k for k in range(lo, hi) if k != i and k != j]
        pm = F.normalize(
            ((unified[i] + unified[j]) / 2.0).unsqueeze(0), dim=-1
        ).squeeze(0)
        gd = 1.0 - (pm * gm).sum()
        if ctx:
            lm = F.normalize(
                unified[ctx].mean(dim=0).unsqueeze(0), dim=-1
            ).squeeze(0)
            ld = 1.0 - (pm * lm).sum()
        else:
            ld = gd
        pair_scores[i] = 0.5 * gd + 0.5 * ld

    return pair_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_geometric_scores(
    unified: torch.Tensor,
    V: torch.Tensor,
    A: torch.Tensor,
    T: torch.Tensor,
    trust: torch.Tensor,
    window: int = 3,
    w_coherence: float = 0.50,
    w_novelty: float = 0.30,
    w_saliency: float = 0.20,
) -> tuple:
    """
    Compute pairwise geometric scores from trimodal embeddings.

    For every consecutive pair of segments (i, i+1), three signals are
    computed using a local neighbourhood of size ``window`` on each side:

    * **Coherence** – cross-modal agreement (V/A/T cosine similarities,
      text down-weighted by ``trust``).
    * **Novelty**   – divergence between the pair and its context.
    * **Saliency**  – deviation from the global and local video mean.

    The three normalised signals are linearly combined into a single
    ``geo_score``.

    Args:
        unified     : unified per-segment embeddings  [N, D]
        V           : L2-normalised vision embeddings [N, D']
        A           : L2-normalised audio embeddings  [N, D']
        T           : L2-normalised text embeddings   [N, D']
        trust       : text-trust weights per segment  [N]  (range [0, 1])
        window      : neighbourhood half-size for pairwise signals (default 3)
        w_coherence : weight for coherence signal  (default 0.50)
        w_novelty   : weight for novelty signal    (default 0.30)
        w_saliency  : weight for saliency signal   (default 0.20)

    Returns:
        unified     : same input tensor (pass-through for pipeline chaining)
        geo_score   : combined geometric score per pair  [N-1]  (float32 tensor)
        components  : dict with normalised component scores
                      ``{"coherence": Tensor[N-1],
                         "novelty":   Tensor[N-1],
                         "saliency":  Tensor[N-1]}``
    """
    raw_coh = _compute_pairwise_coherence(V, A, T, trust, window)
    raw_nov = _compute_pairwise_novelty(unified, window)
    raw_sal = _compute_pairwise_saliency(unified, window)

    coh_n = _minmax_norm(raw_coh)
    nov_n = _minmax_norm(raw_nov)
    sal_n = _minmax_norm(raw_sal)

    # Normalise weights so they always sum to 1
    total = w_coherence + w_novelty + w_saliency
    wc = w_coherence / total
    wn = w_novelty / total
    ws = w_saliency / total

    geo_score = wc * coh_n + wn * nov_n + ws * sal_n

    components = {
        "coherence": coh_n,
        "novelty": nov_n,
        "saliency": sal_n,
    }

    return unified, geo_score, components
