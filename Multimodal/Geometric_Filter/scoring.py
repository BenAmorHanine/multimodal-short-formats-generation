"""
scoring.py — Geometric Filter, Stage 5 Phase 1
===============================================
Computes cross-modal geometric scores over a configurable sliding window.

Window semantics
----------------
window_size=1  → score each segment individually   (original cells 13-14)
window_size=2  → score consecutive pairs  (i, i+1) (cells 13b-14b)
window_size=3  → score consecutive triplets (i, i+1, i+2)
window_size=k  → score consecutive k-tuples (k >= 1)

All three variants produce a score tensor of length  N - (window_size - 1),
i.e. one score per window position, which is then attributed back equally
to every member segment.

Public API
----------
    geo_score, components = compute_geometric_scores(
        unified, V, A, T, trust,
        window_size=2,          # any integer >= 1 and <= N
        context_window=5,       # neighbours on each side (excluded from window)
        w_coherence=0.5,
        w_novelty=0.3,
        w_saliency=0.2,
    )

Returns
-------
geo_score  : torch.Tensor [N]           final combined score per segment
components : dict with keys
               'coherence_n'  [N]   normalised coherence attribution
               'novelty_n'    [N]   normalised novelty attribution
               'saliency_n'   [N]   normalised saliency attribution
               'raw_coherence'[M]   raw window-level coherence  (M = N - ws + 1)
               'raw_novelty'  [M]
               'raw_saliency' [M]
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _seg_coherence(idx: int, V, A, T, trust) -> torch.Tensor:
    """Cross-modal coherence for a single segment index."""
    w_t = trust[idx]
    va  = (V[idx] * A[idx]).sum()
    vt  = (V[idx] * T[idx]).sum() * w_t
    at  = (A[idx] * T[idx]).sum() * w_t
    return (va + vt + at) / (1.0 + w_t + w_t + 1e-8)


def _context_indices(members: list[int], N: int, context_window: int) -> list[int]:
    """Return neighbour indices around a window group, excluding the members."""
    lo = max(0, members[0]  - context_window)
    hi = min(N, members[-1] + context_window + 1)
    return [k for k in range(lo, hi) if k not in members]


# ─────────────────────────────────────────────────────────────────────────────
# Per-window signal functions
# ─────────────────────────────────────────────────────────────────────────────

def _window_coherence(members: list[int], context_ids: list[int],
                      V, A, T, trust) -> torch.Tensor:
    """
    Average coherence of the window members minus the context baseline.
    Positive  → window is more coherent than its surroundings.
    """
    coh_members = torch.stack([_seg_coherence(k, V, A, T, trust) for k in members]).mean()

    if context_ids:
        coh_ctx = torch.stack([_seg_coherence(k, V, A, T, trust) for k in context_ids]).mean()
    else:
        coh_ctx = coh_members.new_tensor(0.0)

    return coh_members - coh_ctx


def _window_novelty(members: list[int], N: int, context_window: int,
                    unified) -> torch.Tensor:
    """
    How much do the members differ from the left context vs the right context?
    High novelty → clear scene boundary inside or around the window.
    """
    lo = max(0, members[0]  - context_window)
    hi = min(N, members[-1] + context_window + 1)

    left_ctx  = unified[lo : members[0]]
    right_ctx = unified[members[-1] + 1 : hi]

    def mean_norm(t):
        if len(t) == 0:
            return None
        return F.normalize(t.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    win_mean = F.normalize(unified[members].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    l_mean   = mean_norm(left_ctx)
    r_mean   = mean_norm(right_ctx)

    scores = []
    if l_mean is not None:
        scores.append(1.0 - (win_mean * l_mean).sum())
    if r_mean is not None:
        scores.append(1.0 - (win_mean * r_mean).sum())

    # within-window diversity (average pairwise dissimilarity between members)
    if len(members) > 1:
        norms = F.normalize(unified[members], dim=-1)
        sims  = norms @ norms.T
        mask  = ~torch.eye(len(members), dtype=torch.bool)
        internal_div = (1.0 - sims[mask]).mean()
        scores.append(internal_div * 0.5)   # downweighted — boundary matters more

    return torch.stack(scores).mean() if scores else win_mean.new_tensor(0.0)


def _window_saliency(members: list[int], context_ids: list[int],
                     unified, global_mean_norm) -> torch.Tensor:
    """
    How much does the window stand out from (a) the whole video and
    (b) its local neighbourhood?
    """
    win_mean = F.normalize(
        unified[members].mean(dim=0, keepdim=True), dim=-1
    ).squeeze(0)

    global_dev = 1.0 - (win_mean * global_mean_norm).sum()

    if context_ids:
        local_mean = F.normalize(
            unified[context_ids].mean(dim=0, keepdim=True), dim=-1
        ).squeeze(0)
        local_dev = 1.0 - (win_mean * local_mean).sum()
    else:
        local_dev = global_dev

    return 0.5 * global_dev + 0.5 * local_dev


# ─────────────────────────────────────────────────────────────────────────────
# Attribution: spread window scores back to individual segments
# ─────────────────────────────────────────────────────────────────────────────

def _attribute_to_segments(window_scores: torch.Tensor,
                            N: int, window_size: int) -> torch.Tensor:
    """
    Each segment i belongs to all windows that cover it.
    We average those window scores back onto the segment.
    """
    seg_acc   = window_scores.new_zeros(N)
    seg_count = window_scores.new_zeros(N)

    for w_idx in range(len(window_scores)):
        for k in range(window_size):
            seg_idx = w_idx + k
            seg_acc[seg_idx]   += window_scores[w_idx]
            seg_count[seg_idx] += 1.0

    return seg_acc / (seg_count + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_geometric_scores(
    unified:        torch.Tensor,   # [N, D]  fused embedding
    V:              torch.Tensor,   # [N, D]  vision  (L2-normalised)
    A:              torch.Tensor,   # [N, D]  audio   (L2-normalised)
    T:              torch.Tensor,   # [N, D]  text    (L2-normalised)
    trust:          torch.Tensor,   # [N]     text trust weights  ∈ [0, 1]
    window_size:    int   = 2,      # any integer >= 1 and <= N
    context_window: int   = 5,      # neighbours on each side for baseline
    w_coherence:    float = 0.5,
    w_novelty:      float = 0.3,
    w_saliency:     float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Compute geometric highlight scores for every segment.

    Parameters
    ----------
    unified        : fused multimodal embedding, shape [N, D]
    V, A, T        : per-modality embeddings, each [N, D], L2-normalised
    trust          : text trust weight per segment, [N]
    window_size    : k consecutive segments per scoring window (k >= 1)
    context_window : how many neighbours on each side form the baseline
    w_coherence / w_novelty / w_saliency : weights (must sum to 1)

    Returns
    -------
    geo_score  : [N] final score per segment
    components : dict of normalised + raw signal tensors
    """
    N = unified.shape[0]
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f"window_size must be an integer >= 1 — got {window_size}")
    if window_size > N:
        raise ValueError(f"window_size ({window_size}) cannot exceed number of segments ({N})")
    if context_window < 0:
        raise ValueError(f"context_window must be >= 0 — got {context_window}")

    w_sum = w_coherence + w_novelty + w_saliency
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Signal weights must sum to 1.0 (got {w_sum:.6f} from "
            f"{w_coherence}, {w_novelty}, {w_saliency})"
        )

    M = N - window_size + 1          # number of window positions

    global_mean_norm = F.normalize(
        unified.mean(dim=0, keepdim=True), dim=-1
    ).squeeze(0)

    raw_coh = unified.new_zeros(M)
    raw_nov = unified.new_zeros(M)
    raw_sal = unified.new_zeros(M)

    for w in range(M):
        members     = list(range(w, w + window_size))
        context_ids = _context_indices(members, N, context_window)

        raw_coh[w] = _window_coherence(members, context_ids, V, A, T, trust)
        raw_nov[w] = _window_novelty(members, N, context_window, unified)
        raw_sal[w] = _window_saliency(members, context_ids, unified, global_mean_norm)

    # normalise window-level scores
    coh_n = _minmax_normalize(raw_coh)
    nov_n = _minmax_normalize(raw_nov)
    sal_n = _minmax_normalize(raw_sal)

    geo_score_windows = w_coherence * coh_n + w_novelty * nov_n + w_saliency * sal_n

    # attribute back to segments
    geo_score   = _attribute_to_segments(geo_score_windows, N, window_size)
    coherence_n = _attribute_to_segments(coh_n, N, window_size)
    novelty_n   = _attribute_to_segments(nov_n, N, window_size)
    saliency_n  = _attribute_to_segments(sal_n, N, window_size)

    components = {
        "coherence_n":   coherence_n,
        "novelty_n":     novelty_n,
        "saliency_n":    saliency_n,
        "raw_coherence": raw_coh,
        "raw_novelty":   raw_nov,
        "raw_saliency":  raw_sal,
    }

    return geo_score, components