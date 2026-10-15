"""Non-federated baselines: centralized MF / NCF (BPR), MostPop and Random."""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import RecData
from .metrics import evaluate_scores
from .models import CentralMF, CentralNCF
from .utils import get_logger

log = get_logger()


class CentralizedTrainer:
    """Mini-batch BPR on pooled data (utility upper bound, no privacy, no fairness)."""

    def __init__(self, cfg: ExperimentConfig, data: RecData, device: torch.device, kind: str = "ncf"):
        self.cfg, self.data, self.device = cfg, data, device
        cls = CentralNCF if kind == "ncf" else CentralMF
        self.model = cls(data.n_users, data.n_items, cfg.model).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.train.central_lr)
        self.pairs = data.train_pairs()

    def fit(self) -> Dict:
        t, d = self.cfg.train, self.data
        k = t.topk[0]
        history, t0 = [], time.time()
        for ep in range(1, t.central_epochs + 1):
            perm = torch.randperm(len(self.pairs), device=self.device)
            total = 0.0
            for s in range(0, len(perm), t.central_batch):
                b = self.pairs[perm[s:s + t.central_batch]]
                u, pos = b[:, 0], b[:, 1]
                neg = d.sample_negatives(u, (t.num_neg,))
                sc = self.model(u, torch.cat([pos.unsqueeze(1), neg], dim=1))
                loss = (-F.logsigmoid(sc[:, :1] - sc[:, 1:])).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += float(loss) * len(b)
            if ep % max(1, t.central_epochs // 10) == 0 or ep == t.central_epochs:
                val, _ = evaluate_scores(self.model.score_matrix(d.n_items), d, "val", t.topk)
                history.append({"round": ep, "train_loss": total / len(self.pairs), f"val_NDCG@{k}": val[f"NDCG@{k}"],
                                f"val_tail_share@{k}": val[f"tail_share@{k}"], f"val_exposure_gap@{k}": val[f"exposure_gap@{k}"]})
                log.info("  epoch %3d | loss %.4f | val NDCG@%d %.4f", ep, total / len(self.pairs), k, val[f"NDCG@{k}"])
        sc = self.model.score_matrix(d.n_items)
        test, extras = evaluate_scores(sc, d, "test", t.topk)
        val, _ = evaluate_scores(sc, d, "val", t.topk)
        return _result(test, val, history, extras, time.time() - t0)


class PopularityRecommender:
    """MostPop (score = train popularity) or Random."""

    def __init__(self, cfg: ExperimentConfig, data: RecData, device: torch.device, kind: str):
        self.cfg, self.data, self.device, self.kind = cfg, data, device, kind

    def fit(self) -> Dict:
        d, t = self.data, self.cfg.train
        if self.kind == "mostpop":
            sc = d.item_pop.to(self.device).unsqueeze(0).repeat(d.n_users, 1)
        elif self.kind == "random":
            sc = torch.rand(d.n_users, d.n_items, device=self.device)
        else:
            raise ValueError(self.kind)
        test, extras = evaluate_scores(sc, d, "test", t.topk)
        val, _ = evaluate_scores(sc, d, "val", t.topk)
        return _result(test, val, [], extras, 0.0)


def _result(test, val, history, extras, seconds) -> Dict:
    return {"test": test, "val": val, "history": history, "extras": extras, "epsilon": float("inf"),
            "max_participations": 0, "privacy_stats": {}, "beta_history": [], "final_beta": 0.0,
            "upload_floats_per_client_per_round": 0, "seconds": seconds}
