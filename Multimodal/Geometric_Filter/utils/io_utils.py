"""
io_utils.py — Geometric Filter I/O
====================================
Save and reload geometric scores and highlight results.

Public API
----------
    geo_path = save_geo_scores(
        geo_score, components, times, texts,
        output_dir, video_stem, window_size=2
    )

    top_path = save_top_segments(
        results, scores, output_dir, video_stem, window_size=2
    )

    geo_data = load_geo_scores(geo_path)
    top_data = load_top_segments(top_path)
"""

import os
import numpy as np
import torch



def save_geo_scores(
    geo_score:   torch.Tensor,
    components:  dict,
    times:       np.ndarray,
    texts:       np.ndarray,
    output_dir:  str,
    video_stem:  str,
    window_size: int = 2,
) -> str:
    """
    Persist full segment-level scores (geo + 3 signals) to .npz.

    Parameters
    ----------
    geo_score   : [N] final combined score per segment
    components  : dict from compute_geometric_scores
                  must contain 'coherence_n', 'novelty_n', 'saliency_n'
    times       : [N, 2] segment timestamps
    texts       : [N]   transcript text
    output_dir  : directory to write the file
    video_stem  : base filename (no extension)
    window_size : recorded in filename for traceability

    Returns
    -------
    path to the saved .npz file
    """
    suffix = {1: "single", 2: "pairwise", 3: "triplet"}.get(window_size, f"w{window_size}")
    path   = os.path.join(output_dir, f"{video_stem}_geo_scores_{suffix}.npz")

    np.savez(
        path,
        geo_score   = geo_score.numpy(),
        coherence   = components["coherence_n"].numpy(),
        novelty     = components["novelty_n"].numpy(),
        saliency    = components["saliency_n"].numpy(),
        times       = times,
        raw_text    = texts,
        window_size = np.array(window_size),
    )
    print(f"✓ Saved {len(geo_score)} segment scores → {path}")
    return path


def save_top_segments(
    results:     list[dict],
    scores:      dict,
    output_dir:  str,
    video_stem:  str,
    window_size: int = 2,
) -> str:
    """
    Persist top-K highlights (from get_highlights_by_window) to .npz
    and print a ranked summary table.

    Parameters
    ----------
    results     : list[dict] from get_highlights_by_window
    scores      : dict of full [N] score tensors (from same call)
    output_dir  : directory to write the file
    video_stem  : base filename (no extension)
    window_size : recorded in filename

    Returns
    -------
    path to the saved .npz file
    """
    top_n  = len(results)
    suffix = {1: "single", 2: "pairwise", 3: "triplet"}.get(window_size, f"w{window_size}")
    path   = os.path.join(output_dir, f"{video_stem}_top{top_n}_{suffix}.npz")

    np.savez(
        path,
        rank        = np.array([r["rank"]      for r in results]),
        seg_idx     = np.array([r["seg_idx"]   for r in results]),
        geo_score   = np.array([r["geo_score"] for r in results]),
        coherence   = np.array([r["coherence"] for r in results]),
        novelty     = np.array([r["novelty"]   for r in results]),
        saliency    = np.array([r["saliency"]  for r in results]),
        times       = np.array([r["times"]     for r in results]),
        raw_text    = np.array([r["text"]      for r in results]),
        window_size = np.array(window_size),
    )
    print(f"✓ Saved top {top_n} highlights → {path}")

    # summary table
    print(f"\n{'Rank':<6} {'Start':>7} {'End':>7} {'Score':>7} "
          f"{'Coh':>6} {'Nov':>6} {'Sal':>6}  Text")
    print("─" * 92)
    for r in results:
        start, end = r["times"]
        print(f"  #{r['rank']:<4} {start:>6.1f}s {end:>6.1f}s "
              f"{r['geo_score']:>7.3f} {r['coherence']:>6.2f} "
              f"{r['novelty']:>6.2f} {r['saliency']:>6.2f}  "
              f"{r['text'][:45]}")

    return path



def load_geo_scores(path: str) -> dict:
    """
    Load a geo_scores .npz saved by save_geo_scores.

    Returns a plain dict with numpy arrays:
        geo_score, coherence, novelty, saliency, times, raw_text, window_size
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_top_segments(path: str) -> list[dict]:
    """
    Load a top-segments .npz saved by save_top_segments.

    Returns a list[dict] in the same format as get_highlights_by_window results.
    """
    data    = np.load(path, allow_pickle=True)
    n       = len(data["rank"])
    results = []
    for i in range(n):
        results.append({
            "rank":      int(data["rank"][i]),
            "seg_idx":   int(data["seg_idx"][i]),
            "times":     data["times"][i].tolist(),
            "text":      str(data["raw_text"][i]),
            "geo_score": float(data["geo_score"][i]),
            "coherence": float(data["coherence"][i]),
            "novelty":   float(data["novelty"][i]),
            "saliency":  float(data["saliency"][i]),
        })
    return results