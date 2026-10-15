"""Neural Collaborative Filtering (NeuMF = GMF + MLP) split into private / shared parts, plus MF.

Federated split
---------------
* ``LocalUserStore``  : user embeddings (GMF + MLP). PRIVATE. Stay on the client, never uploaded.
* ``NCFShared``       : item embeddings + MLP tower + output layer. GLOBAL. Only (clipped + noised)
                        gradients of these are uploaded.
Parameter order matters for the privacy attack: the first two blocks of the flattened update are the
GMF and MLP item-embedding gradients ([n_items, d] each).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class NCFShared(nn.Module):
    """Item side + interaction function of NeuMF. ``forward`` works under ``torch.func.vmap``."""

    def __init__(self, n_items: int, cfg: ModelConfig):
        super().__init__()
        d = cfg.emb_dim
        self.item_gmf = nn.Embedding(n_items, d)
        self.item_mlp = nn.Embedding(n_items, d)
        layers, in_dim = [], 2 * d
        for h in cfg.mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d + in_dim, 1)
        nn.init.normal_(self.item_gmf.weight, std=cfg.init_std)
        nn.init.normal_(self.item_mlp.weight, std=cfg.init_std)

    def forward(self, u_gmf: torch.Tensor, u_mlp: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """u_*: [..., d]; items: [..., n]  ->  scores [..., n]."""
        ig, im = self.item_gmf(items), self.item_mlp(items)
        ug = u_gmf.unsqueeze(-2).expand_as(ig)
        um = u_mlp.unsqueeze(-2).expand_as(im)
        gmf = ug * ig
        mlp = self.mlp(torch.cat([um, im], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class LocalUserStore:
    """Private per-client embeddings (simulation container)."""

    def __init__(self, n_users: int, cfg: ModelConfig, device: torch.device):
        d = cfg.emb_dim
        self.gmf = torch.randn(n_users, d, device=device) * cfg.init_std
        self.mlp = torch.randn(n_users, d, device=device) * cfg.init_std


class CentralNCF(nn.Module):
    """Non-federated NeuMF (upper-bound baseline)."""

    def __init__(self, n_users: int, n_items: int, cfg: ModelConfig):
        super().__init__()
        d = cfg.emb_dim
        self.user_gmf = nn.Embedding(n_users, d)
        self.user_mlp = nn.Embedding(n_users, d)
        nn.init.normal_(self.user_gmf.weight, std=cfg.init_std)
        nn.init.normal_(self.user_mlp.weight, std=cfg.init_std)
        self.shared = NCFShared(n_items, cfg)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.shared(self.user_gmf(users), self.user_mlp(users), items)

    @torch.no_grad()
    def score_matrix(self, n_items: int) -> torch.Tensor:
        return score_all_items(self.shared, self.user_gmf.weight, self.user_mlp.weight, n_items)


class CentralMF(nn.Module):
    """Centralized matrix factorisation with BPR:  r_ui = p_u^T q_i."""

    def __init__(self, n_users: int, n_items: int, cfg: ModelConfig):
        super().__init__()
        self.user = nn.Embedding(n_users, cfg.emb_dim)
        self.item = nn.Embedding(n_items, cfg.emb_dim)
        nn.init.normal_(self.user.weight, std=cfg.init_std)
        nn.init.normal_(self.item.weight, std=cfg.init_std)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users).unsqueeze(1) * self.item(items)).sum(-1)

    @torch.no_grad()
    def score_matrix(self, n_items: int) -> torch.Tensor:
        return self.user.weight @ self.item.weight.T


@torch.no_grad()
def score_all_items(shared: NCFShared, u_gmf: torch.Tensor, u_mlp: torch.Tensor,
                    n_items: int, chunk: int = 64) -> torch.Tensor:
    """Full score matrix [n_users, n_items] for the given user embeddings."""
    out = []
    all_items = torch.arange(n_items, device=u_gmf.device)
    for s in range(0, len(u_gmf), chunk):
        ug, um = u_gmf[s:s + chunk], u_mlp[s:s + chunk]
        items = all_items.unsqueeze(0).expand(len(ug), -1)
        out.append(shared(ug, um, items))
    return torch.cat(out, dim=0)
