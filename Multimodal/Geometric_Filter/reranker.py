"""
reranker.py — Post-filter highlight reranking
============================================
Reranks already-filtered highlight candidates using embedding-aware criteria.

This module is intended to run AFTER `get_highlights_by_window`.

Signals
-------
    A) Semantic centrality: proximity to candidate centroid
    B) Diversity: inverse of max similarity to other candidates
    C) Narrative emphasis: directional embedding change over time

Final score
-----------
    rerank_score =
        w_geo       * base_geo_score +
        w_centrality* centrality +
        w_diversity * diversity +
        w_narrative * narrative
"""

import numpy as np
import torch
import torch.nn.functional as F


def _norm01(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def rerank_filtered_highlights(
    results: list[dict],
    unified: torch.Tensor,
    w_geo: float = 0.50,
    w_centrality: float = 0.20,
    w_diversity: float = 0.15,
    w_narrative: float = 0.15,
) -> tuple[list[dict], dict]:
    """
    Rerank already-filtered highlight candidates.

    Parameters
    ----------
    results : output list from get_highlights_by_window (already filtered)
    unified : fused embedding tensor [N, D]
    w_*     : reranker weights (must sum to 1)

    Returns
    -------
    reranked_results : list[dict] sorted by rerank score descending
    details          : dict of component score tensors and rank order
    """
    w_sum = w_geo + w_centrality + w_diversity + w_narrative
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Reranker weights must sum to 1.0 (got {w_sum:.6f} from "
            f"{w_geo}, {w_centrality}, {w_diversity}, {w_narrative})"
        )

    if not results:
        empty = torch.tensor([])
        return [], {
            "base_geo": empty,
            "centrality": empty,
            "diversity": empty,
            "narrative": empty,
            "rerank_score": empty,
            "order": [],
        }

    unified_norm = F.normalize(unified, dim=-1)

    cand_emb = []
    base_geo = []
    start_times = []

    for r in results:
        members = r.get("member_seg_idx")
        if not members:
            members = [int(r.get("seg_idx", 0))]

        emb = F.normalize(unified_norm[members].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        cand_emb.append(emb)
        base_geo.append(float(r["geo_score"]))
        start_times.append(float(r["times"][0]))

    cand_emb = torch.stack(cand_emb)  # [K, D]
    base_geo_t = _norm01(torch.tensor(base_geo, dtype=torch.float32))
    K = cand_emb.shape[0]

    # A) Centrality
    centroid = F.normalize(cand_emb.mean(dim=0, keepdim=True), dim=-1)
    centrality = _norm01((cand_emb * centroid).sum(dim=1))

    # B) Diversity
    if K == 1:
        diversity = torch.ones(1)
    else:
        sim_mat = cand_emb @ cand_emb.T
        sim_mat.fill_diagonal_(-1.0)
        diversity = _norm01(1.0 - sim_mat.max(dim=1).values)

    # C) Narrative emphasis
    if K == 1:
        narrative = torch.ones(1)
    else:
        time_order = np.argsort(np.array(start_times))
        emb_sorted = cand_emb[time_order]
        narrative = torch.zeros(K)

        for rank, orig_idx in enumerate(time_order):
            prev_rank = rank - 1 if rank > 0 else rank
            next_rank = rank + 1 if rank < K - 1 else rank

            v_in = F.normalize((emb_sorted[rank] - emb_sorted[prev_rank]).unsqueeze(0), dim=-1).squeeze(0)
            v_out = F.normalize((emb_sorted[next_rank] - emb_sorted[rank]).unsqueeze(0), dim=-1).squeeze(0)
            narrative[orig_idx] = 1.0 - (v_in * v_out).sum()

        narrative = _norm01(narrative)

    rerank_score = (
        w_geo * base_geo_t +
        w_centrality * centrality +
        w_diversity * diversity +
        w_narrative * narrative
    )

    order = rerank_score.argsort(descending=True).tolist()
    reranked = []
    for new_rank, idx in enumerate(order, start=1):
        item = dict(results[idx])
        item["rank"] = new_rank
        item["base_geo_score"] = float(base_geo_t[idx].item())
        item["centrality"] = float(centrality[idx].item())
        item["diversity"] = float(diversity[idx].item())
        item["narrative"] = float(narrative[idx].item())
        item["rerank_score"] = float(rerank_score[idx].item())
        reranked.append(item)

    details = {
        "base_geo": base_geo_t,
        "centrality": centrality,
        "diversity": diversity,
        "narrative": narrative,
        "rerank_score": rerank_score,
        "order": order,
    }
    return reranked, details
