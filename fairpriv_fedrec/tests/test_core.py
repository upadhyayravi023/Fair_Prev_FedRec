"""Run with:  pytest -q"""
import numpy as np
import pandas as pd
import torch

from fairpriv.config import DataConfig, ExperimentConfig, FairnessConfig, PrivacyConfig
from fairpriv.data import process_interactions
from fairpriv.fairness import BetaController, compute_item_weights, exposure_gap, gap_penalty
from fairpriv.federated import FederatedTrainer
from fairpriv.metrics import evaluate_scores, gini
from fairpriv.privacy import (GradientPrivatizer, calibrate_sigma, clip_l2, compute_epsilon, gaussian_rdp_epsilon)


def toy_data(n_users=30, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    probs = 1.0 / np.arange(1, n_items + 1)
    probs /= probs.sum()
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=15, replace=False, p=probs)
        for t, i in enumerate(items):
            rows.append((u, int(i), 5 if t % 3 else 4, t))
        rows.append((u, int(rng.integers(n_items)), 2, 99))     # low rating: must be dropped
    return process_interactions(pd.DataFrame(rows, columns=["user", "item", "rating", "ts"]), DataConfig())


# ---------------------------------------------------------------- data
def test_ratings_below_threshold_are_dropped_and_split_has_no_leakage():
    d = toy_data()
    assert d.stats["n_train"] <= 30 * 13
    u = torch.arange(d.n_users)
    assert not d.train_mask[u, d.val_items].any()
    assert not d.train_mask[u, d.test_items].any()
    assert (d.val_items != d.test_items).all()


# ---------------------------------------------------------------- fairness
def test_popaware_weight_matches_formula():
    pop = torch.tensor([1000.0, 100.0, 10.0])
    w = compute_item_weights(pop, FairnessConfig(weighting="popaware", beta=2.0))
    assert torch.allclose(w, torch.tensor([1.0, 1 + 2 * 0.9, 1 + 2 * 0.99]))
    assert torch.equal(compute_item_weights(pop, FairnessConfig(weighting="popaware", beta=0.0)), torch.ones(3))


def test_ips_is_mean_normalised():
    pop = torch.tensor([500.0, 50.0, 5.0])
    w = compute_item_weights(pop, FairnessConfig(weighting="ips"))
    assert (w[2] > w[1] > w[0]).item() and abs(float((pop * w).sum() / pop.sum()) - 1.0) < 1e-5


def test_exposure_gap_parity_and_extreme():
    assert abs(exposure_gap(0.2, 0.2)) < 1e-9            # exposure share == catalogue share -> parity
    assert abs(exposure_gap(1.0, 0.2) - 5.0) < 1e-9      # all slots to head -> 1 / r


def test_gap_penalty_zero_at_parity_and_positive_when_head_favoured():
    head = torch.tensor([True, False, False, False, False])
    assert gap_penalty(torch.zeros(5), head, 1.0, 1.0) < 1e-12            # uniform scores: parity
    assert gap_penalty(torch.tensor([5.0, 0, 0, 0, 0]), head, 1.0, 1.0) > 0.1
    g = torch.tensor([1.0, 0, 0, 0, 0], requires_grad=True)
    gap_penalty(g, head, 1.0, 1.0).backward()
    assert g.grad[0] > 0                                                    # gradient descent lowers head score


def test_beta_controller_moves_toward_target_and_is_bounded():
    cfg = FairnessConfig(gap_target=2.0, beta_lr=0.5, beta_max=3.0)
    c = BetaController(cfg, 1.0)
    assert c.update(1, 4.0) == 2.0                        # gap above target -> beta up
    assert c.update(2, 10.0) == 3.0                       # capped at beta_max
    assert c.update(3, -10.0) == 0.0                      # floored at 0


# ---------------------------------------------------------------- privacy
def test_clipping_bounds_norm_and_keeps_small_vectors():
    g = torch.randn(50, 200) * torch.linspace(0.01, 5, 50).unsqueeze(1)
    clipped, norms = clip_l2(g, 1.0)
    assert (clipped.norm(dim=1) <= 1.0 + 1e-5).all()
    assert torch.allclose(clipped[norms <= 1.0], g[norms <= 1.0])


def test_aggregate_noise_scale_local_and_distributed_with_weights():
    D, n = 40000, 16
    a = torch.rand(n) + 0.5
    a = a / a.sum()
    for placement, expected in (("local", 0.5 * 2.0 * float(a.norm())), ("distributed", 0.5 * 2.0 * float(a.max()))):
        p = GradientPrivatizer(PrivacyConfig(mechanism="gaussian", clip_norm=2.0, noise_multiplier=0.5, placement=placement))
        total = p.finalize(p.process_chunk(torch.zeros(n, D), a), float(a.max()), float(a.norm()))
        assert abs(float(total.std()) - expected) / expected < 0.03, placement


def test_no_clipping_ablation_does_not_clip_but_still_measures_norms():
    g = torch.ones(4, 100) * 5.0                            # norm 50
    p = GradientPrivatizer(PrivacyConfig(mechanism="gaussian", clip=False, noise_multiplier=0.0))
    out = p.process_chunk(g, torch.full((4,), 0.25))
    assert torch.allclose(out, torch.full((100,), 5.0))
    p2 = GradientPrivatizer(PrivacyConfig(mechanism="gaussian", clip=True, clip_norm=1.0, noise_multiplier=0.0))
    assert float(p2.process_chunk(g, torch.full((4,), 0.25)).norm()) <= 1.0 + 1e-5
    assert compute_epsilon(PrivacyConfig(mechanism="gaussian", clip=False, noise_multiplier=1.0), 10) == float("inf")


def test_accountant_monotone_and_calibration_roundtrip():
    assert gaussian_rdp_epsilon(2.0, 100, 1e-5) < gaussian_rdp_epsilon(1.0, 100, 1e-5)
    loc = compute_epsilon(PrivacyConfig(mechanism="gaussian", noise_multiplier=1.0, placement="local"), 50)
    dis = compute_epsilon(PrivacyConfig(mechanism="gaussian", noise_multiplier=1.0, placement="distributed"), 50)
    assert loc > dis
    for plc in ("local", "distributed"):
        s = calibrate_sigma(100.0, 100, 1e-5, plc)
        eps = compute_epsilon(PrivacyConfig(mechanism="gaussian", noise_multiplier=s, placement=plc), 100)
        assert 99.0 < eps <= 100.0 + 1e-6


# ---------------------------------------------------------------- metrics
def test_gini_extremes():
    assert abs(gini(np.ones(100))) < 1e-9
    x = np.zeros(100)
    x[0] = 1
    assert gini(x) > 0.98


def test_perfect_ranker_and_new_metrics():
    d = toy_data()
    scores = torch.zeros(d.n_users, d.n_items)
    scores[torch.arange(d.n_users), d.test_items] = 10.0
    m, _ = evaluate_scores(scores, d, "test", (5, 10))
    assert m["HR@5"] == 1.0 and m["Recall@5"] == 1.0 and abs(m["NDCG@5"] - 1.0) < 1e-6 and m["MRR@5"] == 1.0
    assert abs(m["recall_gap@5"]) < 1e-9


# ---------------------------------------------------------------- federated internals
def _trainer(**over):
    d = toy_data()
    cfg = ExperimentConfig().but(**over)
    return FederatedTrainer(cfg, d, torch.device("cpu")), d


def test_vmap_client_gradient_matches_autograd_with_gap_regulariser():
    tr, d = _trainer(**{"fair.weighting": "popaware", "fair.beta": 3.0, "fair.gap_lambda": 5.0})
    u = torch.tensor([0, 1, 2])
    args = tr._client_batch(u)
    params = {n: p.detach() for n, p in tr.model.named_parameters()}
    (g_params, g_ug, _), _ = tr._per_client(params, *args)

    ug, um, pos, neg, w, cand, cand_head = args
    ug1 = ug[1].clone().requires_grad_(True)
    items = torch.cat([pos[1].unsqueeze(-1), neg[1]], -1).reshape(-1)
    s = tr.model(ug1, um[1], items).reshape(-1, 1 + neg.shape[-1])
    loss = (w[1].unsqueeze(-1) * -torch.nn.functional.logsigmoid(s[:, :1] - s[:, 1:])).mean()
    loss = loss + gap_penalty(tr.model(ug1, um[1], cand[1]), cand_head[1], 5.0, 1.0)
    ref = torch.autograd.grad(loss, list(tr.model.parameters()) + [ug1])
    for (name, _), r in zip(tr.model.named_parameters(), ref[:-1]):
        assert torch.allclose(g_params[name][1], r, atol=1e-6), name
    assert torch.allclose(g_ug[1], ref[-1], atol=1e-6)


def test_round_data_size_weighting_and_sampling():
    tr, d = _trainer(**{"train.client_fraction": 0.5})
    tr.run_round(1)
    assert int(tr.participation.sum()) == 15


def test_adaptive_beta_changes_item_weights():
    tr, d = _trainer(**{"fair.weighting": "popaware", "fair.beta": 1.0, "fair.adaptive_beta": True, "fair.monitor_every": 1,
                        "fair.gap_target": 50.0})                 # target above any reachable gap -> beta must drop
    w0 = tr.item_w.clone()
    tr.run_round(1)
    assert tr.beta < 1.0 and not torch.equal(w0, tr.item_w)


def test_attack_auc_is_high_without_privacy_and_drops_with_large_noise():
    tr, d = _trainer(**{"train.attack_clients": 30})
    tr.run_round(1)
    clean = tr.evaluate_attack(30)["attack_auc"]
    tr2, _ = _trainer(**{"train.attack_clients": 30, "privacy.mechanism": "gaussian", "privacy.noise_multiplier": 5.0})
    tr2.run_round(1)
    noisy = tr2.evaluate_attack(30)["attack_auc"]
    assert clean > 0.7 and noisy < clean - 0.15
