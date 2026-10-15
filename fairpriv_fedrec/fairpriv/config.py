"""Central experiment configuration for FairPriv-FedRec.

Every knob that can change a result lives here; a run is fully described by one
``ExperimentConfig`` (dumped next to the results).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class DataConfig:
    name: str = "ml-1m"                  # "ml-1m" (default, as in the plan) | "ml-100k"
    root: str = "data"
    positive_threshold: float = 4.0      # rating >= 4  ->  implicit positive
    min_user_interactions: int = 5       # users with fewer positives are dropped
    head_ratio: float = 0.2              # top 20 % most popular items = head, rest = tail
    max_users: Optional[int] = None      # random user subset (debugging only)
    subset_seed: int = 0


@dataclass
class ModelConfig:
    emb_dim: int = 32
    mlp_hidden: Tuple[int, ...] = (64, 32, 16)
    init_std: float = 0.1


@dataclass
class FairnessConfig:
    # --- (a) popularity-aware loss re-weighting:  w_i = 1 + beta (1 - pop_i / max pop) ---
    weighting: str = "none"              # "none" | "popaware" | "ips" (extra baseline)
    beta: float = 0.0
    ips_gamma: float = 0.5
    # --- (b) exposure-gap regularisation (client-side, differentiable) -------------------
    #     L_gap = lambda * ( sum_{i in head} softmax(s_u / tau)_i  -  |head ∩ cand| / |cand| )^2
    #     evaluated on `gap_candidates` random not-yet-seen items per client per round.
    gap_lambda: float = 0.0
    gap_candidates: int = 64
    gap_temp: float = 1.0
    # --- (c) fairness monitor: server adapts beta from the head/tail exposure gap ---------
    adaptive_beta: bool = False
    gap_target: float = 2.0              # target normalised exposure gap (0 = parity)
    beta_lr: float = 0.5                 # beta <- clip(beta + beta_lr * (gap - gap_target), 0, beta_max)
    beta_max: float = 16.0
    monitor_every: int = 5               # rounds between monitor reports
    monitor_epsilon: Optional[float] = None   # if set: each client's scalar report gets Lap(1/eps)


@dataclass
class PrivacyConfig:
    mechanism: str = "none"              # "none" | "gaussian" | "laplace"
    clip: bool = True                    # False only for the "no clipping" ablation
    clip_norm: float = 1.0               # C
    noise_multiplier: float = 0.0        # sigma: Gaussian std = sigma * C
    epsilon: float = 1.0                 # per-round epsilon for Laplace (scale = C / epsilon)
    placement: str = "local"             # "local" (as in the plan) | "distributed" (secure-agg style)
    delta: float = 1e-5


@dataclass
class TrainConfig:
    rounds: int = 100
    client_fraction: float = 1.0         # |U_t| / |U|
    aggregation: str = "data_size"       # "data_size": weights n_u / sum n_v (as in the plan) | "uniform"
    local_batch: int = 64
    num_neg: int = 4
    server_optimizer: str = "adam"       # "adam" | "sgd"   (sgd = literal theta <- theta - eta * g)
    server_lr: float = 0.01
    server_momentum: float = 0.0         # only for "sgd"
    user_lr: float = 2.0                 # local step for the private user embedding
    weight_decay: float = 0.0
    client_chunk: int = 128
    # centralized baselines
    central_epochs: int = 30
    central_batch: int = 1024
    central_lr: float = 1e-3
    # evaluation
    eval_every: int = 10
    topk: Tuple[int, ...] = (10, 20)
    attack_clients: int = 500            # clients used for the privacy-attack evaluation
    seed: int = 0
    device: str = "auto"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fair: FairnessConfig = field(default_factory=FairnessConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def but(self, **overrides: Any) -> "ExperimentConfig":
        """Deep copy with dotted overrides, e.g. ``cfg.but(**{"fair.beta": 2.0})``."""
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            section, name = key.split(".")
            sec = getattr(new, section)
            if not hasattr(sec, name):
                raise KeyError(f"unknown config field {key}")
            setattr(sec, name, value)
        return new

    def flat(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for section, values in asdict(self).items():
            for k, v in values.items():
                out[f"{section}.{k}"] = v
        return out
