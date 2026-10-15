"""Evaluation: utility + fairness metrics of the project plan (full-catalogue ranking).

Utility    HR@K, Recall@K, NDCG@K, MRR@K            (leave-one-out: Recall@K == HR@K; both are reported)
Fairness   tail_coverage@K   fraction of tail items recommended at least once
           exposure_gap@K    normalised head-vs-tail exposure gap (0 = parity, see fairness.py)
           gini@K            Gini index of per-item exposure (0 = equal)
           recall_gap@K      Recall@K of users with a HEAD target minus Recall@K of users with a TAIL target
           tail_share@K, head_share@K, coverage@K, ARP@K, exposure_ratio@K, pop_corr@K   (extra diagnostics)
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import RecData
from .fairness import exposure_gap


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


@torch.no_grad()
def compute_metrics(scores: torch.Tensor, seen: torch.Tensor, target: torch.Tensor,
                    data: RecData, ks: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """scores [U,I]; seen [U,I] bool (excluded from the ranking); target [U] held-out item."""
    U, I = scores.shape
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Model produced non-finite scores (training diverged); lower the learning rate.")
    scores = scores.masked_fill(seen, float("-inf"))
    t_score = scores.gather(1, target.unsqueeze(1))
    greater = (scores > t_score).sum(1).float()
    equal = (scores == t_score).sum(1).float()              # includes the target itself
    rank = greater + (equal + 1.0) / 2.0                      # tie-aware 1-based rank
    is_head_target = data.head_mask[target]
    pop = data.item_pop.cpu().numpy()
    head_np = data.head_mask.cpu().numpy()
    r = float(head_np.mean())

    out: Dict[str, float] = {"MRR": float((1.0 / rank).mean())}
    extras: Dict[str, np.ndarray] = {}
    topk = torch.topk(scores, max(ks), dim=1).indices

    for k in ks:
        hit = (rank <= k).float()
        ndcg = hit / torch.log2(rank + 1.0)
        out[f"HR@{k}"] = float(hit.mean())
        out[f"Recall@{k}"] = out[f"HR@{k}"]
        out[f"NDCG@{k}"] = float(ndcg.mean())
        out[f"MRR@{k}"] = float((hit / rank).mean())
        hr_head = float(hit[is_head_target].mean()) if is_head_target.any() else float("nan")
        hr_tail = float(hit[~is_head_target].mean()) if (~is_head_target).any() else float("nan")
        out[f"HR@{k}_head"], out[f"HR@{k}_tail"] = hr_head, hr_tail
        out[f"recall_gap@{k}"] = hr_head - hr_tail
        for name, m in (("head", is_head_target), ("tail", ~is_head_target)):
            out[f"NDCG@{k}_{name}"] = float(ndcg[m].mean()) if m.any() else float("nan")

        counts = torch.bincount(topk[:, :k].reshape(-1), minlength=I).cpu().numpy().astype(np.float64)
        total = counts.sum()
        n_head, n_tail = head_np.sum(), (~head_np).sum()
        head_exp, tail_exp = counts[head_np].sum(), counts[~head_np].sum()
        hs = float(head_exp / total)
        out[f"head_share@{k}"], out[f"tail_share@{k}"] = hs, 1.0 - hs
        out[f"exposure_gap@{k}"] = exposure_gap(hs, r)
        out[f"coverage@{k}"] = float((counts > 0).mean())
        out[f"tail_coverage@{k}"] = float((counts[~head_np] > 0).mean())
        out[f"gini@{k}"] = gini(counts)
        out[f"ARP@{k}"] = float((counts * pop).sum() / total)
        per_tail = tail_exp / max(n_tail, 1)
        out[f"exposure_ratio@{k}"] = float((head_exp / max(n_head, 1)) / per_tail) if per_tail > 0 else float("inf")
        out[f"pop_corr@{k}"] = float(spearmanr(pop, counts).statistic) if counts.std() > 0 else 0.0
        extras[f"exposure@{k}"] = counts
    return out, extras


def evaluate_scores(scores: torch.Tensor, data: RecData, split: str,
                    ks: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """split = 'val' (mask train) or 'test' (mask train + val item)."""
    seen = data.train_mask
    if split == "val":
        target = data.val_items
    elif split == "test":
        target = data.test_items
        seen = seen.clone()
        seen[torch.arange(data.n_users, device=seen.device), data.val_items] = True
    else:
        raise ValueError(split)
    return compute_metrics(scores, seen, target, data, ks)
