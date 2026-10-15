"""Method registry + single ``run_method`` entry point.

Names follow the project plan:
    Centralized-MF / Centralized-NCF      centralized upper bounds
    Vanilla-FedRec                        federated, no privacy, no fairness
    DP-FedRec                             privacy only (clip + noise)
    Fair-FedRec                           fairness only (re-weighting + exposure-gap regulariser)
    FairPriv-FedRec (ours)                privacy + fairness
plus references (Random, MostPop), one extra fairness baseline (Fair-FedRec (IPS)) and the ablations
(component removed one at a time).
beta / lambda / sigma / C / epsilon come from the base config (command line).
"""
from __future__ import annotations

from typing import Any, Dict

from .centralized import CentralizedTrainer, PopularityRecommender
from .config import ExperimentConfig
from .data import RecData
from .federated import FederatedTrainer
from .utils import get_device, get_logger, set_seed

log = get_logger()

_NO_FAIR = {"fair.weighting": "none", "fair.gap_lambda": 0.0, "fair.adaptive_beta": False}
_FAIR = {"fair.weighting": "popaware"}
_NO_PRIV = {"privacy.mechanism": "none"}
_PRIV = {"privacy.mechanism": "gaussian"}

METHODS: Dict[str, Dict[str, Any]] = {
    "Random":                  {"kind": "random"},
    "MostPop":                 {"kind": "mostpop"},
    "Centralized-MF":          {"kind": "central_mf"},
    "Centralized-NCF":         {"kind": "central_ncf"},
    "Vanilla-FedRec":          {"kind": "fed", "over": {**_NO_FAIR, **_NO_PRIV}},
    "Fair-FedRec (IPS)":       {"kind": "fed", "over": {**_NO_FAIR, "fair.weighting": "ips", **_NO_PRIV}},
    "DP-FedRec":               {"kind": "fed", "over": {**_NO_FAIR, **_PRIV}},
    "Fair-FedRec":             {"kind": "fed", "over": {**_FAIR, **_NO_PRIV}},
    "FairPriv-FedRec (ours)":  {"kind": "fed", "over": {**_FAIR, **_PRIV}},
    # ---- ablations of the full method: remove one component at a time ----
    "Ours w/o clipping":       {"kind": "fed", "over": {**_FAIR, **_PRIV, "privacy.clip": False}},
    "Ours w/o noise":          {"kind": "fed", "over": {**_FAIR, **_PRIV, "privacy.noise_multiplier": 0.0}},
    "Ours w/o fairness weight": {"kind": "fed", "over": {**_PRIV, "fair.weighting": "none", "fair.adaptive_beta": False}},
    "Ours w/o gap regulariser": {"kind": "fed", "over": {**_FAIR, **_PRIV, "fair.gap_lambda": 0.0}},
    "Ours + adaptive beta":    {"kind": "fed", "over": {**_FAIR, **_PRIV, "fair.adaptive_beta": True}},
}


def run_method(method: str, base_cfg: ExperimentConfig, data: RecData, seed: int) -> Dict[str, Any]:
    spec = METHODS[method]
    cfg = base_cfg.but(**spec.get("over", {}))
    cfg.train.seed = seed
    set_seed(seed)
    device = get_device(cfg.train.device)
    data_d = data.to(device)
    fed = spec["kind"] == "fed"
    log.info("=== %s | seed %d | weighting=%s beta=%.2f lambda=%.3g adaptive=%s | mech=%s sigma=%.3g C=%.3g clip=%s %s",
             method, seed, cfg.fair.weighting, cfg.fair.beta, cfg.fair.gap_lambda, cfg.fair.adaptive_beta,
             cfg.privacy.mechanism, cfg.privacy.noise_multiplier, cfg.privacy.clip_norm, cfg.privacy.clip,
             cfg.privacy.placement)

    if spec["kind"] in {"random", "mostpop"}:
        trainer = PopularityRecommender(cfg, data_d, device, spec["kind"])
    elif spec["kind"] in {"central_mf", "central_ncf"}:
        trainer = CentralizedTrainer(cfg, data_d, device, kind=spec["kind"].split("_")[1])
    else:
        trainer = FederatedTrainer(cfg, data_d, device)
    res = trainer.fit()

    private = fed and cfg.privacy.mechanism != "none"
    row: Dict[str, Any] = {
        "method": method, "seed": seed,
        "weighting": cfg.fair.weighting if fed else "none",
        "beta": res["final_beta"] if (fed and cfg.fair.weighting == "popaware") else 0.0,
        "gap_lambda": cfg.fair.gap_lambda if fed else 0.0,
        "mechanism": cfg.privacy.mechanism if fed else "none",
        "sigma": cfg.privacy.noise_multiplier if private and cfg.privacy.mechanism == "gaussian" else 0.0,
        "clip": cfg.privacy.clip_norm if private and cfg.privacy.clip else float("nan"),
        "placement": cfg.privacy.placement if private else "none",
        "epsilon": res["epsilon"], "max_participations": res["max_participations"], "seconds": res["seconds"],
        **{f"test_{k}": v for k, v in res["test"].items()},
        **{f"val_{k}": v for k, v in res["val"].items()},
        **{f"priv_{k}": v for k, v in res["privacy_stats"].items()},
    }
    return {"row": row, "history": res["history"], "extras": res["extras"], "cfg": cfg,
            "beta_history": res["beta_history"]}
