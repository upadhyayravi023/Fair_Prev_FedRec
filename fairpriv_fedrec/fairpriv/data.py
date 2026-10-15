"""Data loading, preprocessing and sampling for implicit-feedback recommendation.

Protocol (as in the project plan)
---------------------------------
* MovieLens-1M; ratings >= ``positive_threshold`` (4) become implicit positives,
* every user is one client,
* per user, positives are sorted by timestamp:
      last item          -> test
      second-to-last     -> validation
      everything else    -> train
* evaluation ranks the held-out item against the FULL catalogue (no sampled negatives),
  excluding items the user interacted with in train (and the validation item when testing).
"""
from __future__ import annotations

import io
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import DataConfig
from .utils import get_logger

log = get_logger()

_OFFICIAL = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}
_NCF_BASE = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/"
_EXPECTED = {"ml-1m": (6040, 3706, 1_000_209), "ml-100k": (943, 1682, 100_000)}


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #
def _download(name: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    try:
        log.info("Downloading %s from GroupLens ...", name)
        raw = urllib.request.urlopen(_OFFICIAL[name], timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(root)
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("Official download failed (%s).", exc)
    if name == "ml-1m":   # mirror: NCF repo ships the same ratings split into train/test files
        try:
            parts = [pd.read_csv(io.BytesIO(urllib.request.urlopen(_NCF_BASE + f"ml-1m.{s}.rating", timeout=120).read()),
                                 sep="\t", names=["user", "item", "rating", "ts"]) for s in ("train", "test")]
            df = pd.concat(parts)
            df["user"] += 1
            df["item"] += 1
            (root / "ml-1m").mkdir(exist_ok=True)
            lines = ["::".join(map(str, r)) for r in df.itertuples(index=False)]
            (root / "ml-1m" / "ratings.dat").write_text("\n".join(lines) + "\n")
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("Mirror failed (%s).", exc)
    raise FileNotFoundError(
        f"Could not obtain {name}. Download it from {_OFFICIAL[name]} and unzip it into '{root}/'.")


def load_raw(cfg: DataConfig) -> pd.DataFrame:
    root = Path(cfg.root)
    if cfg.name == "ml-1m":
        path = root / "ml-1m" / "ratings.dat"
        if not path.exists():
            _download("ml-1m", root)
        df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "rating", "ts"])
    elif cfg.name == "ml-100k":
        path = root / "ml-100k" / "u.data"
        if not path.exists():
            _download("ml-100k", root)
        df = pd.read_csv(path, sep="\t", names=["user", "item", "rating", "ts"])
    else:
        raise ValueError(f"Unknown dataset {cfg.name!r}. Add a loader in data.load_raw().")
    shape = (df.user.nunique(), df.item.nunique(), len(df))
    if shape != _EXPECTED[cfg.name]:
        log.warning("%s has shape %s, expected %s", cfg.name, shape, _EXPECTED[cfg.name])
    return df


# --------------------------------------------------------------------------- #
# processed container
# --------------------------------------------------------------------------- #
@dataclass
class RecData:
    n_users: int
    n_items: int
    train_padded: torch.Tensor   # [U, Lmax] item ids, 0-padded
    train_len: torch.Tensor      # [U]   (= n_u, the client's data size)
    train_mask: torch.Tensor     # [U, I] bool
    val_items: torch.Tensor      # [U]
    test_items: torch.Tensor     # [U]
    item_pop: torch.Tensor       # [I] float, #train interactions per item
    head_mask: torch.Tensor      # [I] bool
    stats: Dict[str, float]

    def to(self, device: torch.device) -> "RecData":
        return RecData(
            self.n_users, self.n_items,
            self.train_padded.to(device), self.train_len.to(device), self.train_mask.to(device),
            self.val_items.to(device), self.test_items.to(device),
            self.item_pop.to(device), self.head_mask.to(device), self.stats,
        )

    def train_pairs(self) -> torch.Tensor:
        u, i = torch.nonzero(self.train_mask, as_tuple=True)
        return torch.stack([u, i], dim=1)

    def sample_positives(self, users: torch.Tensor, n: int) -> torch.Tensor:
        """Uniformly sample ``n`` train positives (with replacement) for each user."""
        lens = self.train_len[users].unsqueeze(1)
        idx = (torch.rand(len(users), n, device=lens.device) * lens).long()
        return self.train_padded[users.unsqueeze(1), idx]

    def sample_negatives(self, users: torch.Tensor, shape_tail: tuple, tries: int = 10) -> torch.Tensor:
        """Uniform items the user has not interacted with in train."""
        shape = (len(users),) + tuple(shape_tail)
        neg = torch.randint(0, self.n_items, shape, device=users.device)
        u_idx = users.view(-1, *([1] * len(shape_tail)))
        for _ in range(tries):
            bad = self.train_mask[u_idx.expand(shape), neg]
            if not bad.any():
                break
            neg = torch.where(bad, torch.randint_like(neg, self.n_items), neg)
        return neg


def build_recdata(cfg: DataConfig) -> RecData:
    return process_interactions(load_raw(cfg), cfg)


def process_interactions(df: pd.DataFrame, cfg: DataConfig) -> RecData:
    """Raw (user, item, rating, ts) frame -> leave-one-out ``RecData``."""
    n_raw = len(df)
    df = df[df["rating"] >= cfg.positive_threshold]
    df = df.drop_duplicates(["user", "item"])
    counts = df.groupby("user").size()
    df = df[df.user.isin(counts[counts >= cfg.min_user_interactions].index)]
    if cfg.max_users is not None:
        keep = np.random.default_rng(cfg.subset_seed).permutation(df.user.unique())[: cfg.max_users]
        df = df[df.user.isin(keep)]

    df = df.copy()
    df["user"] = df["user"].astype("category").cat.codes.astype(np.int64)
    df["item"] = df["item"].astype("category").cat.codes.astype(np.int64)
    n_users, n_items = int(df.user.max()) + 1, int(df.item.max()) + 1

    df = df.sort_values(["user", "ts", "item"], kind="mergesort")
    train_lists: List[np.ndarray] = []
    val = np.zeros(n_users, dtype=np.int64)
    test = np.zeros(n_users, dtype=np.int64)
    for u, grp in df.groupby("user", sort=True):
        items = grp["item"].to_numpy().copy()
        test[u], val[u] = items[-1], items[-2]
        train_lists.append(items[:-2])

    lens = np.array([len(x) for x in train_lists])
    padded = np.zeros((n_users, lens.max()), dtype=np.int64)
    mask = torch.zeros(n_users, n_items, dtype=torch.bool)
    for u, items in enumerate(train_lists):
        padded[u, : len(items)] = items
        mask[u, torch.from_numpy(items)] = True

    pop = mask.sum(0).float()
    order = torch.argsort(pop, descending=True, stable=True)
    n_head = int(math.ceil(cfg.head_ratio * n_items))
    head = torch.zeros(n_items, dtype=torch.bool)
    head[order[:n_head]] = True

    n_train = int(lens.sum())
    stats = {
        "raw_ratings": n_raw,
        "n_users": n_users, "n_items": n_items, "n_train": n_train,
        "density": n_train / (n_users * n_items),
        "head_items": n_head,
        "head_interaction_share": float(pop[head].sum() / pop.sum()),
        "test_target_head_share": float(head[torch.from_numpy(test)].float().mean()),
    }
    log.info("Dataset %s: %s", cfg.name, {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})
    return RecData(
        n_users, n_items,
        torch.from_numpy(padded), torch.from_numpy(lens), mask,
        torch.from_numpy(val), torch.from_numpy(test), pop, head, stats,
    )
