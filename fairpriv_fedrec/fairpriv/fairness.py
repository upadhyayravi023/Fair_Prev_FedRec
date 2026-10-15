"""Fairness components of FairPriv-FedRec (no sensitive demographic attributes are used).

(a) Popularity-aware re-weighting            w_i = 1 + beta * (1 - pop(i) / max_j pop(j))
        L_u = sum_{(u,i,j)} w_i * L_BPR(u,i,j)      (weight of the *positive* item i)
(b) Exposure-gap regularisation (client side)
        p      = softmax(s_u(cand) / tau)           soft exposure over random unseen candidates
        L_gap  = lambda * ( sum_{i in head} p_i  -  share of head items among the candidates )^2
    -> penalises a client's model for concentrating its score mass on head items.
(c) Fairness monitor + adaptive beta (server side)
        gap    = normalised exposure gap of the top-K lists of the reporting (validation) users
        beta  <- clip( beta + beta_lr * (gap - gap_target), 0, beta_max )

Normalised exposure gap (used everywhere, also as an evaluation metric):
        EG@K = (head exposure per head item - tail exposure per tail item) / (uniform exposure per item)
             = hs / r - (1 - hs) / (1 - r),       hs = share of top-K slots given to head items,
                                                  r  = |head| / |catalogue|
    EG = 0 means parity; larger = more head-dominated (max = 1 / r).

ASSUMPTION to state in the paper: item popularity counts / head-tail membership are public aggregate
statistics broadcast by the server. Popularity counts are not protected by the DP noise.
"""
from __future__ import annotations

from typing import List

import torch

from .config import FairnessConfig


def compute_item_weights(item_pop: torch.Tensor, cfg: FairnessConfig, beta: float | None = None) -> torch.Tensor:
    pop = item_pop.float()
    if cfg.weighting == "none":
        return torch.ones_like(pop)
    if cfg.weighting == "popaware":
        b = cfg.beta if beta is None else beta
        return 1.0 + b * (1.0 - pop / pop.max())
    if cfg.weighting == "ips":
        # inverse-propensity style baseline, mean-normalised over train interactions
        raw = pop.clamp(min=1.0).pow(-cfg.ips_gamma)
        return raw / ((pop * raw).sum() / pop.sum())
    raise ValueError(f"unknown weighting {cfg.weighting!r}")


def exposure_gap(head_share: float, head_ratio: float) -> float:
    """Normalised exposure gap from the share of top-K slots given to head items."""
    return head_share / head_ratio - (1.0 - head_share) / (1.0 - head_ratio)


def gap_penalty(cand_scores: torch.Tensor, cand_is_head: torch.Tensor, lam: float, tau: float) -> torch.Tensor:
    """Differentiable exposure-gap penalty of ONE client (works under vmap).

    cand_scores [M] scores of M random unseen items; cand_is_head [M] bool.
    """
    p = torch.softmax(cand_scores / tau, dim=-1)
    head = cand_is_head.to(p.dtype)
    return lam * ((p * head).sum() - head.mean()) ** 2


class BetaController:
    """Server-side fairness monitor: raises beta while the exposure gap is above target."""

    def __init__(self, cfg: FairnessConfig, beta0: float):
        self.cfg = cfg
        self.beta = float(beta0)
        self.history: List[dict] = []

    def update(self, round_: int, gap: float) -> float:
        c = self.cfg
        self.beta = float(min(max(self.beta + c.beta_lr * (gap - c.gap_target), 0.0), c.beta_max))
        self.history.append({"round": round_, "gap": gap, "beta": self.beta})
        return self.beta
