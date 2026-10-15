"""Basic privacy layer: L2 gradient clipping + Gaussian / Laplace noise (+ accounting).

Per-client update of the shared parameters (as in the project plan)::

    clip:      g_bar   = g_u * min(1, C / ||g_u||_2)
    Gaussian:  g_tilde = g_bar + N(0, sigma^2 C^2 I)
    Laplace :  g_tilde = g_bar + Lap(0, C / epsilon)

Server aggregation with data-size weights a_u = n_u / sum_v n_v  (or uniform 1/|U_t|)::

    g = sum_u a_u * g_tilde_u

Noise placement
    "local"       every client perturbs its own vector (local-DP style, as in the plan).
    "distributed" the aggregate receives one N(0, sigma^2 C^2 a_max^2 I) draw (secure-aggregation
                  style, equals central DP on the weighted sum with sensitivity a_max * C).

Accounting (Gaussian): Renyi-DP -> (epsilon, delta) (Canonne et al. 2020 conversion), composed over the
rounds a client participated in, WITHOUT amplification by client sampling (so epsilon is an upper bound).
    local       -> sensitivity of the clipped vector = 2C (any two client datasets)
    distributed -> sensitivity = C (add / remove one client)
No clipping => unbounded sensitivity => epsilon = inf.

CAVEATS
* Laplace with scale C/epsilon (formula from the plan) is an exact epsilon-DP guarantee only if C bounds the
  L1 sensitivity; with L2 clipping in d dimensions the L1 sensitivity can be up to C*sqrt(d).
* With ~10^5-10^6 shared parameters, the epsilon of the noise levels at which the model still works is large;
  report it honestly and complement it with the empirical attack evaluation (see federated.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .config import PrivacyConfig


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def clip_l2(g: torch.Tensor, clip_norm: float, eps: float = 1e-12):
    """Row-wise L2 clipping of a [B, D] matrix. Returns (clipped, per-row norm)."""
    norms = g.norm(dim=1)
    scale = torch.clamp(clip_norm / (norms + eps), max=1.0)
    return g * scale.unsqueeze(1), norms


def gaussian_noise(shape, std: float, device) -> torch.Tensor:
    return torch.randn(shape, device=device) * std


def laplace_noise(shape, scale: float, device) -> torch.Tensor:
    u = torch.rand(shape, device=device) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=0.5 - 1e-7))


@dataclass
class PrivacyStats:
    n: int = 0
    norm_sum: float = 0.0
    clipped: int = 0

    def update(self, norms: torch.Tensor, clip: float) -> None:
        self.n += len(norms)
        self.norm_sum += float(norms.sum())
        self.clipped += int((norms > clip).sum())

    def summary(self) -> Dict[str, float]:
        n = max(self.n, 1)
        return {"grad_norm_mean": self.norm_sum / n, "clip_fraction": self.clipped / n}


# --------------------------------------------------------------------------- #
# privatizer used by the federated trainer
# --------------------------------------------------------------------------- #
class GradientPrivatizer:
    def __init__(self, cfg: PrivacyConfig):
        if cfg.mechanism not in {"none", "gaussian", "laplace"}:
            raise ValueError(cfg.mechanism)
        if cfg.placement not in {"local", "distributed"}:
            raise ValueError(cfg.placement)
        self.cfg = cfg
        self.stats = PrivacyStats()

    @property
    def enabled(self) -> bool:
        return self.cfg.mechanism != "none"

    @property
    def adds_noise(self) -> bool:
        c = self.cfg
        return c.mechanism == "laplace" or (c.mechanism == "gaussian" and c.noise_multiplier > 0)

    def _noise(self, shape, device, scale_mult: float = 1.0) -> torch.Tensor:
        c = self.cfg
        if c.mechanism == "gaussian":
            return gaussian_noise(shape, c.noise_multiplier * c.clip_norm * scale_mult, device)
        return laplace_noise(shape, (c.clip_norm / c.epsilon) * scale_mult, device)

    def _clip(self, g: torch.Tensor) -> torch.Tensor:
        clipped, norms = clip_l2(g, self.cfg.clip_norm)
        self.stats.update(norms, self.cfg.clip_norm)
        return clipped if self.cfg.clip else g

    def upload(self, g: torch.Tensor) -> torch.Tensor:
        """What each client sends: clip (+ local noise). [B,D] -> [B,D]."""
        if not self.enabled:
            return g
        g = self._clip(g)
        if self.cfg.placement == "local" and self.adds_noise:
            g = g + self._noise(g.shape, g.device)
        return g

    def server_view(self, g: torch.Tensor, n_selected: int) -> torch.Tensor:
        """Per-client vectors as seen by a (curious) server, used by the attack evaluation.

        local: exactly the uploads. distributed: uploads plus each client's *share* sigma*C/sqrt(n) of the
        aggregate noise (i.e. an adversary that bypasses secure aggregation).
        """
        if not self.enabled:
            return g
        g = self._clip(g)
        if self.adds_noise:
            mult = 1.0 if self.cfg.placement == "local" else 1.0 / math.sqrt(max(n_selected, 1))
            g = g + self._noise(g.shape, g.device, mult)
        return g

    def process_chunk(self, g: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """[B, D] raw client gradients, [B] aggregation weights -> [D] weighted sum of the clipped uploads.

        Simulation shortcut (exact in distribution): with LOCAL Gaussian noise the server only ever uses
        sum_u a_u * N(0, sigma^2 C^2 I) ~ N(0, sigma^2 C^2 ||a||_2^2 I), so that single draw is added in
        ``finalize``.  Laplace noise is not closed under summation and is added per client here.
        """
        if not self.enabled:
            return (g * weights.unsqueeze(1)).sum(0)
        g = self._clip(g)
        if self.cfg.placement == "local" and self.cfg.mechanism == "laplace":
            g = g + self._noise(g.shape, g.device)
        return (g * weights.unsqueeze(1)).sum(0)

    def finalize(self, total: torch.Tensor, w_max: float, w_l2: float) -> torch.Tensor:
        """Add the aggregate noise (see ``process_chunk``) and return the weighted aggregate."""
        if not (self.enabled and self.adds_noise):
            return total
        if self.cfg.placement == "distributed":
            return total + self._noise(total.shape, total.device, w_max)
        if self.cfg.mechanism == "gaussian":
            return total + self._noise(total.shape, total.device, w_l2)
        return total


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
_ORDERS = np.concatenate([np.linspace(1.05, 10.0, 200), np.arange(11, 513, dtype=float)])


def gaussian_rdp_epsilon(noise_multiplier: float, steps: int, delta: float,
                         sensitivity_ratio: float = 1.0) -> float:
    """(epsilon, delta) after ``steps`` compositions; noise std = noise_multiplier*C, sensitivity = ratio*C."""
    if noise_multiplier <= 0:
        return float("inf")
    if steps <= 0:
        return 0.0
    a = _ORDERS
    rdp = steps * a * (sensitivity_ratio ** 2) / (2.0 * noise_multiplier ** 2)
    eps = rdp + np.log((a - 1.0) / a) - (math.log(delta) + np.log(a)) / (a - 1.0)
    return float(max(np.min(eps), 0.0))


def compute_epsilon(cfg: PrivacyConfig, participations: int) -> float:
    if cfg.mechanism == "none" or not cfg.clip:
        return float("inf")
    if cfg.mechanism == "laplace":
        return participations * cfg.epsilon          # basic composition of the nominal per-round epsilon
    ratio = 2.0 if cfg.placement == "local" else 1.0
    return gaussian_rdp_epsilon(cfg.noise_multiplier, participations, cfg.delta, ratio)


def calibrate_sigma(target_epsilon: float, participations: int, delta: float, placement: str) -> float:
    """Smallest noise multiplier whose composed Gaussian epsilon is <= target (bisection in log-space)."""
    ratio = 2.0 if placement == "local" else 1.0
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gaussian_rdp_epsilon(mid, participations, delta, ratio) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi
