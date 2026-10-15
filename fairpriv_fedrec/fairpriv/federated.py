"""FairPriv-FedRec: federated NCF with fairness-aware re-weighting and a basic privacy layer.

Algorithm (project plan, Section 7)
-----------------------------------
1. Server initialises theta^0 (item embeddings, MLP tower, output layer).
2. For every round t = 1..T
   2.1 server samples clients U_t and sends theta^t (and the item weights w).
   2.2 each client trains locally with the weighted BPR / NCF loss
            L_u = mean_{(u,i,j)} w_i * L_BPR(u,i,j)  +  lambda * L_gap(u)          (exposure-gap regulariser)
       -> gradient g_u of the shared parameters + gradient of its private user embedding
       (the private embedding is updated locally and never leaves the device).
   2.3 client clips g_u to norm C and adds Gaussian / Laplace noise -> g~_u  (privacy layer)
   2.4 server aggregates    g = sum_{u in U_t} (n_u / sum_v n_v) * g~_u ;  theta^{t+1} = theta^t - eta * g
       (``server_optimizer="sgd"`` is the literal update; "adam" applies Adam to the same aggregate g -
        pure post-processing, no extra privacy cost - and is the default because it trains far better)
   2.5 fairness monitor: every ``monitor_every`` rounds the reporting clients send the head-share of their
       own validation top-K list (a scalar, optionally Laplace-noised); the server computes the exposure gap
       and (if ``adaptive_beta``) tunes beta.
3. Final top-K lists are produced on the clients from the final global model + private embeddings.

Per-client gradients are computed in parallel with ``torch.func.vmap``.
A simple item-interaction inference attack on the uploads is evaluated at the end of training.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.func import functional_call, grad_and_value, vmap

from .config import ExperimentConfig
from .data import RecData
from .fairness import BetaController, compute_item_weights, exposure_gap, gap_penalty
from .metrics import evaluate_scores
from .models import LocalUserStore, NCFShared, score_all_items
from .privacy import GradientPrivatizer, compute_epsilon
from .utils import get_logger

log = get_logger()


class FederatedTrainer:
    def __init__(self, cfg: ExperimentConfig, data: RecData, device: torch.device):
        self.cfg, self.data, self.device = cfg, data, device
        t, f = cfg.train, cfg.fair
        self.model = NCFShared(data.n_items, cfg.model).to(device)
        self.users = LocalUserStore(data.n_users, cfg.model, device)
        self.privatizer = GradientPrivatizer(cfg.privacy)
        self.head_ratio = float(data.head_mask.float().mean())

        self.beta = float(f.beta)
        self.ctrl = BetaController(f, f.beta) if (f.adaptive_beta and f.weighting == "popaware") else None
        self.item_w = compute_item_weights(data.item_pop, f, self.beta).to(device)
        self.use_gap = f.gap_lambda > 0

        params = list(self.model.parameters())
        if t.server_optimizer == "adam":
            self.opt = torch.optim.Adam(params, lr=t.server_lr, weight_decay=t.weight_decay)
        elif t.server_optimizer == "sgd":
            self.opt = torch.optim.SGD(params, lr=t.server_lr, momentum=t.server_momentum, weight_decay=t.weight_decay)
        else:
            raise ValueError(t.server_optimizer)

        self.param_names = [n for n, _ in self.model.named_parameters()]
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_numel = [p.numel() for p in self.model.parameters()]
        self.dim = int(sum(self.param_numel))
        self.emb_dim = cfg.model.emb_dim
        self.participation = torch.zeros(data.n_users, dtype=torch.long)

        if self.use_gap:
            self._per_client = vmap(grad_and_value(self._loss_gap, argnums=(0, 1, 2)),
                                    in_dims=(None, 0, 0, 0, 0, 0, 0, 0))
        else:
            self._per_client = vmap(grad_and_value(self._loss_plain, argnums=(0, 1, 2)),
                                    in_dims=(None, 0, 0, 0, 0, 0))

    # ------------------------------------------------------------------ #
    # client side
    # ------------------------------------------------------------------ #
    def _bpr(self, params, u_gmf, u_mlp, pos, neg, w):
        k = neg.shape[-1]
        items = torch.cat([pos.unsqueeze(-1), neg], dim=-1).reshape(-1)
        s = functional_call(self.model, params, (u_gmf, u_mlp, items)).reshape(-1, 1 + k)
        bpr = -F.logsigmoid(s[:, :1] - s[:, 1:])            # [B, K]
        return (w.unsqueeze(-1) * bpr).mean()               # weighted BPR of ONE client

    def _loss_plain(self, params, u_gmf, u_mlp, pos, neg, w):
        return self._bpr(params, u_gmf, u_mlp, pos, neg, w)

    def _loss_gap(self, params, u_gmf, u_mlp, pos, neg, w, cand, cand_head):
        base = self._bpr(params, u_gmf, u_mlp, pos, neg, w)
        cs = functional_call(self.model, params, (u_gmf, u_mlp, cand))
        return base + gap_penalty(cs, cand_head, self.cfg.fair.gap_lambda, self.cfg.fair.gap_temp)

    def _client_batch(self, chunk: torch.Tensor):
        """Local training sample of every client in ``chunk`` -> argument tuple for the vmapped loss."""
        t, d = self.cfg.train, self.data
        pos = d.sample_positives(chunk, t.local_batch)
        neg = d.sample_negatives(chunk, (t.local_batch, t.num_neg))
        w = self.item_w[pos]
        ug, um = self.users.gmf[chunk], self.users.mlp[chunk]
        args = [ug, um, pos, neg, w]
        if self.use_gap:
            cand = d.sample_negatives(chunk, (self.cfg.fair.gap_candidates,))
            args += [cand, d.head_mask[cand]]
        return args

    def _flatten(self, grads: Dict[str, torch.Tensor], b: int) -> torch.Tensor:
        return torch.cat([grads[n].reshape(b, -1) for n in self.param_names], dim=1)

    def _unflatten(self, vec: torch.Tensor) -> List[torch.Tensor]:
        return [c.view(s) for c, s in zip(torch.split(vec, self.param_numel), self.param_shapes)]

    # ------------------------------------------------------------------ #
    # one communication round
    # ------------------------------------------------------------------ #
    def run_round(self, r: int) -> Dict[str, float]:
        t, d = self.cfg.train, self.data
        n_sel = max(1, int(round(t.client_fraction * d.n_users)))
        selected = torch.randperm(d.n_users, device=self.device)[:n_sel]
        self.participation[selected.cpu()] += 1
        params = {n: p.detach() for n, p in self.model.named_parameters()}

        n_u = d.train_len[selected].float()
        agg_w = n_u / n_u.sum() if t.aggregation == "data_size" else torch.full_like(n_u, 1.0 / n_sel)
        w_max, w_l2 = float(agg_w.max()), float(agg_w.norm())

        total = torch.zeros(self.dim, device=self.device)
        loss_sum = 0.0
        for chunk, aw in zip(torch.split(selected, t.client_chunk), torch.split(agg_w, t.client_chunk)):
            args = self._client_batch(chunk)
            (g_params, g_ug, g_um), losses = self._per_client(params, *args)
            # private (local-only) user-embedding step
            self.users.gmf[chunk] = args[0] - t.user_lr * g_ug
            self.users.mlp[chunk] = args[1] - t.user_lr * g_um
            # clip + noise + weighted sum
            total += self.privatizer.process_chunk(self._flatten(g_params, len(chunk)), aw)
            loss_sum += float(losses.sum())

        agg = self.privatizer.finalize(total, w_max, w_l2)
        for p, g in zip(self.model.parameters(), self._unflatten(agg)):
            p.grad = g
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        info = {"train_loss": loss_sum / n_sel}
        if self.ctrl is not None and r % self.cfg.fair.monitor_every == 0:
            info.update(self._monitor(selected, r))
        return info

    # ------------------------------------------------------------------ #
    # fairness monitor (Step 2.5)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _monitor(self, selected: torch.Tensor, r: int) -> Dict[str, float]:
        f, d = self.cfg.fair, self.data
        k = self.cfg.train.topk[0]
        rep = selected[torch.randperm(len(selected), device=self.device)[:1000]]
        scores = score_all_items(self.model, self.users.gmf[rep], self.users.mlp[rep], d.n_items)
        scores = scores.masked_fill(d.train_mask[rep], float("-inf"))
        top = torch.topk(scores, k, dim=1).indices
        share = d.head_mask[top].float().mean(1)                     # each client's scalar report in [0,1]
        if f.monitor_epsilon is not None:
            share = share + torch.distributions.Laplace(0.0, 1.0 / f.monitor_epsilon).sample(share.shape).to(share.device)
        hs = float(share.mean().clamp(0.0, 1.0))
        gap = exposure_gap(hs, self.head_ratio)
        if self.ctrl is not None:
            self.beta = self.ctrl.update(r, gap)
            self.item_w = compute_item_weights(d.item_pop, f, self.beta).to(self.device)
        return {"monitor_gap": gap, "beta": self.beta}

    # ------------------------------------------------------------------ #
    # privacy attack: item-interaction inference from one uploaded update
    # ------------------------------------------------------------------ #
    def evaluate_attack(self, n_clients: int) -> Dict[str, float]:
        """A curious server ranks items by the energy of the item-embedding rows of a client's upload.

        Label of an item = it is in the client's TRAIN set. Reports the mean per-client
          attack_auc  : ROC-AUC of the ranking (0.5 = the upload reveals nothing, 1 = perfect leakage)
          attack_prec : precision of the top-|P_u| items (random guess ~ |P_u| / n_items)
        """
        t, d = self.cfg.train, self.data
        saved_stats = copy.deepcopy(self.privatizer.stats)
        users = torch.randperm(d.n_users, device=self.device)[: min(n_clients, d.n_users)]
        n_sel = max(1, int(round(t.client_fraction * d.n_users)))
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        I, dim = d.n_items, self.emb_dim
        aucs, precs = [], []
        for chunk in torch.split(users, t.client_chunk):
            args = self._client_batch(chunk)
            (g_params, _, _), _ = self._per_client(params, *args)
            view = self.privatizer.server_view(self._flatten(g_params, len(chunk)), n_sel)
            gmf = view[:, : I * dim].reshape(len(chunk), I, dim)
            mlp = view[:, I * dim: 2 * I * dim].reshape(len(chunk), I, dim)
            energy = (gmf.pow(2).sum(-1) + mlp.pow(2).sum(-1)).double().cpu().numpy()
            labels = d.train_mask[chunk].cpu().numpy()
            for sc, lab in zip(energy, labels):
                npos, nneg = int(lab.sum()), int((~lab).sum())
                ranks = rankdata(sc)
                aucs.append((ranks[lab].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
                jit = sc + 1e-12 * np.random.rand(len(sc)) * (np.abs(sc).max() + 1e-30)
                top = np.argpartition(-jit, npos - 1)[:npos]
                precs.append(lab[top].mean())
        self.privatizer.stats = saved_stats
        return {"attack_auc": float(np.mean(aucs)), "attack_prec": float(np.mean(precs))}

    # ------------------------------------------------------------------ #
    # evaluation / training loop
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_matrix(self) -> torch.Tensor:
        return score_all_items(self.model, self.users.gmf, self.users.mlp, self.data.n_items)

    def evaluate(self, split: str):
        return evaluate_scores(self.score_matrix(), self.data, split, self.cfg.train.topk)

    def fit(self) -> Dict:
        t = self.cfg.train
        k = t.topk[0]
        history: List[Dict] = []
        t0 = time.time()
        for r in range(1, t.rounds + 1):
            info = self.run_round(r)
            if r % t.eval_every == 0 or r == t.rounds:
                val, _ = self.evaluate("val")
                history.append({"round": r, "train_loss": info["train_loss"], "beta": self.beta,
                                f"val_NDCG@{k}": val[f"NDCG@{k}"], f"val_tail_share@{k}": val[f"tail_share@{k}"],
                                f"val_exposure_gap@{k}": val[f"exposure_gap@{k}"]})
                log.info("  round %3d | loss %.4f | val NDCG@%d %.4f | val exp-gap@%d %.3f | beta %.2f | %.0fs",
                         r, info["train_loss"], k, val[f"NDCG@{k}"], k, val[f"exposure_gap@{k}"], self.beta, time.time() - t0)
        test, extras = self.evaluate("test")
        val, _ = self.evaluate("val")
        attack = self.evaluate_attack(t.attack_clients)
        max_part = int(self.participation.max())
        return {
            "test": test, "val": val, "history": history, "extras": extras,
            "epsilon": compute_epsilon(self.cfg.privacy, max_part), "max_participations": max_part,
            "privacy_stats": {**self.privatizer.stats.summary(), **attack},
            "beta_history": self.ctrl.history if self.ctrl else [], "final_beta": self.beta,
            "upload_floats_per_client_per_round": self.dim, "seconds": time.time() - t0,
        }
