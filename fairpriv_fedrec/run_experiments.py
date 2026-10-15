#!/usr/bin/env python
"""Command line entry point for FairPriv-FedRec.

Examples
--------
    python run_experiments.py --exp main --seeds 0 1 2          # Tables 1-2 + figures
    python run_experiments.py --exp all  --seeds 0 1 2          # everything
    python run_experiments.py --exp main --seeds 0 --quick      # smoke test
    python run_experiments.py --exp main --seeds 0 --quick --max-users 1000   # very fast debug run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairpriv.config import ExperimentConfig
from fairpriv.data import build_recdata
from fairpriv.experiments import EXPERIMENTS
from fairpriv.utils import get_logger

log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair + privacy-preserving federated recommendation (FairPriv-FedRec)")
    p.add_argument("--exp", nargs="+", default=["main"], choices=list(EXPERIMENTS) + ["all"])
    p.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "ml-100k"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    # headline hyper-parameters of the proposed method
    p.add_argument("--beta", type=float, default=4.0, help="popularity-aware weight strength")
    p.add_argument("--gap-lambda", type=float, default=1.0, help="exposure-gap regulariser strength")
    p.add_argument("--sigma", type=float, default=0.003, help="Gaussian noise multiplier (noise std = sigma*C)")
    p.add_argument("--clip", type=float, default=1.0, help="L2 clipping threshold C")
    p.add_argument("--placement", default="local", choices=["local", "distributed"])
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps-grid", nargs="+", type=float, default=[1e4, 1e5, 1e6, 1e7], help="targets for --exp epsilon")
    # training budget
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--client-fraction", type=float, default=1.0)
    p.add_argument("--aggregation", default="data_size", choices=["data_size", "uniform"])
    p.add_argument("--server-optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--server-lr", type=float, default=None)
    p.add_argument("--central-epochs", type=int, default=30)
    p.add_argument("--client-chunk", type=int, default=128, help="clients per vmap batch (raise on a GPU)")
    p.add_argument("--max-users", type=int, default=None, help="random user subset (debugging)")
    p.add_argument("--quick", action="store_true", help="tiny budget for a smoke test")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = ExperimentConfig()
    cfg.data.name, cfg.data.root, cfg.data.max_users = a.dataset, a.data_root, a.max_users
    t, f, pr = cfg.train, cfg.fair, cfg.privacy
    t.device, t.rounds, t.client_fraction, t.aggregation = a.device, a.rounds, a.client_fraction, a.aggregation
    t.server_optimizer, t.central_epochs, t.client_chunk = a.server_optimizer, a.central_epochs, a.client_chunk
    if a.server_lr is not None:
        t.server_lr = a.server_lr
    f.beta, f.gap_lambda = a.beta, a.gap_lambda
    pr.noise_multiplier, pr.clip_norm, pr.placement, pr.delta = a.sigma, a.clip, a.placement, a.delta
    if a.quick:
        t.rounds, t.central_epochs, t.eval_every, t.attack_clients = 15, 5, 5, 200

    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.flat(), indent=2, default=str))
    data = build_recdata(cfg.data)
    (out / "dataset_stats.json").write_text(json.dumps(data.stats, indent=2))

    names = list(EXPERIMENTS) if "all" in a.exp else a.exp
    for name in names:
        log.info("########## experiment: %s ##########", name)
        kwargs = {"eps_grid": a.eps_grid} if name == "epsilon" else {}
        summ = EXPERIMENTS[name](cfg, data, a.seeds, out, **kwargs)
        keys = [c for c in ("method", "beta_set", "lambda_set", "sigma_set", "target_epsilon") if c in summ.columns]
        cols = [c for c in summ.columns if c.endswith("_mean") and any(
            s in c for s in ("NDCG@10", "exposure_gap@10", "tail_coverage@10", "attack_auc", "epsilon"))]
        log.info("\n%s", summ[keys + cols].round(4).to_string(index=False))
    log.info("Done. Results in %s", out.resolve())


if __name__ == "__main__":
    main()
