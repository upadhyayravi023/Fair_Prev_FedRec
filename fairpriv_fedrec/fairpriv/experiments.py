"""Paper experiments (project-plan tables / figures).

    main       Table 1 (utility) + Table 2 (fairness) for all baselines and the proposed method
    ablation   remove one component at a time (clipping, noise, fairness weight, gap regulariser) + adaptive beta
    beta       different beta   -> Figure 1: accuracy vs fairness trade-off
    lambda     different lambda (exposure-gap regulariser strength)
    privacy    different noise levels sigma -> Table 3: privacy-utility trade-off (incl. attack accuracy)
    epsilon    different target privacy budgets epsilon (sigma calibrated with the accountant)

Every function saves raw + summary CSVs, LaTeX tables and figures under ``out/<experiment>/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import reporting as rp
from .config import ExperimentConfig
from .data import RecData
from .privacy import calibrate_sigma
from .runner import run_method
from .utils import get_logger

log = get_logger()

MAIN_METHODS = ["Random", "MostPop", "Centralized-MF", "Centralized-NCF", "Vanilla-FedRec", "Fair-FedRec (IPS)",
                "DP-FedRec", "Fair-FedRec", "FairPriv-FedRec (ours)"]
ABLATION_METHODS = ["Vanilla-FedRec", "DP-FedRec", "Fair-FedRec", "FairPriv-FedRec (ours)", "Ours w/o clipping",
                    "Ours w/o noise", "Ours w/o fairness weight", "Ours w/o gap regulariser", "Ours + adaptive beta"]

UTILITY = ["test_HR@10", "test_Recall@10", "test_NDCG@10", "test_MRR@10"]
FAIRNESS = ["test_tail_coverage@10", "test_exposure_gap@10", "test_gini@10", "test_recall_gap@10",
            "test_tail_share@10", "test_ARP@10"]
PRIVACY = ["priv_attack_auc", "priv_attack_prec"]
ALL_METRICS = (UTILITY + FAIRNESS + PRIVACY + ["test_HR@20", "test_NDCG@20", "test_MRR", "test_HR@10_head",
               "test_HR@10_tail", "test_coverage@10", "test_exposure_ratio@10", "test_pop_corr@10",
               "epsilon", "seconds", "priv_clip_fraction", "priv_grad_norm_mean", "beta"])

Job = Tuple[str, Dict[str, Any], Dict[str, Any]]      # (method, config overrides, extra columns)


def _run_jobs(jobs: Sequence[Job], cfg: ExperimentConfig, data: RecData, seeds: Sequence[int]):
    rows, curves, exposures = [], {}, {}
    for method, over, tags in jobs:
        run_cfg = cfg.but(**over)
        for seed in seeds:
            out = run_method(method, run_cfg, data, seed)
            rows.append({**out["row"], **tags})
            if seed == seeds[0]:
                curves[method] = out["history"]
                exposures[method] = out["extras"][f"exposure@{cfg.train.topk[0]}"]
    return pd.DataFrame(rows), curves, exposures


def _save(df: pd.DataFrame, summ: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_raw.csv", index=False)
    summ.to_csv(out / f"{name}_summary.csv", index=False)


# --------------------------------------------------------------------------- #
def exp_main(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path, **_) -> pd.DataFrame:
    out = out / "main"
    df, curves, expo = _run_jobs([(m, {}, {}) for m in MAIN_METHODS], cfg, data, seeds)
    summ = rp.summarize(df, ["method"], ALL_METRICS)
    _save(df, summ, out, "main")
    cap = f"({cfg.data.name}, mean $\\pm$ std over seeds; bold = best excluding Random)"
    rp.to_latex(summ, ["method"], UTILITY, out / "table1_utility.tex", f"Table 1: utility {cap}.", "tab:utility", exclude_rows=("Random",))
    rp.to_latex(summ, ["method"], FAIRNESS, out / "table2_fairness.tex", f"Table 2: fairness {cap}.", "tab:fairness", exclude_rows=("Random",))
    rp.to_latex(summ, ["method"], UTILITY[:3] + FAIRNESS[:2] + PRIVACY + ["epsilon"], out / "table_main_with_privacy.tex",
                f"Utility, fairness and empirical leakage {cap}.", "tab:main", exclude_rows=("Random",))
    rp.plot_training_curves({k: v for k, v in curves.items() if v}, out / "fig_training_curves", cfg.train.topk[0])
    rp.plot_exposure({k: v for k, v in expo.items() if k != "Random"}, data.head_mask.cpu().numpy(), out / "fig_exposure", cfg.train.topk[0])
    return summ


def exp_ablation(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path, **_) -> pd.DataFrame:
    out = out / "ablation"
    df, _, _ = _run_jobs([(m, {}, {}) for m in ABLATION_METHODS], cfg, data, seeds)
    summ = rp.summarize(df, ["method"], ALL_METRICS)
    _save(df, summ, out, "ablation")
    rp.to_latex(summ, ["method"], UTILITY[:3] + FAIRNESS[:3] + PRIVACY + ["epsilon"], out / "table_ablation.tex",
                "Ablation: one component removed at a time.", "tab:ablation")
    return summ


def exp_beta(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path,
             betas: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), **_) -> pd.DataFrame:
    out = out / "beta"
    jobs = [(m, {"fair.beta": b}, {"beta_set": b}) for m in ("Fair-FedRec", "FairPriv-FedRec (ours)") for b in betas]
    df, _, _ = _run_jobs(jobs, cfg, data, seeds)
    summ = rp.summarize(df, ["method", "beta_set"], ALL_METRICS)
    _save(df, summ, out, "beta")
    rp.to_latex(summ, ["method", "beta_set"], UTILITY[:3] + FAIRNESS[:4], out / "table_beta.tex",
                "Effect of the fairness strength $\\beta$.", "tab:beta")
    rp.plot_tradeoff(summ, out / "fig1_accuracy_vs_fairness", "test_exposure_gap@10", "test_NDCG@10", "method", "beta_set",
                     "Exposure gap@10 (lower = fairer)", "NDCG@10", "Accuracy vs fairness (labels = $\\beta$)")
    rp.plot_lines(summ, out / "fig_beta_lines", "beta_set",
                  [("test_NDCG@10", "NDCG@10"), ("test_tail_coverage@10", "Tail coverage@10"), ("test_gini@10", "Gini@10")],
                  "method", "$\\beta$")
    return summ


def exp_lambda(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path,
               lambdas: Sequence[float] = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0), **_) -> pd.DataFrame:
    out = out / "lambda"
    jobs = [(m, {"fair.gap_lambda": lam}, {"lambda_set": lam}) for m in ("Fair-FedRec", "FairPriv-FedRec (ours)") for lam in lambdas]
    df, _, _ = _run_jobs(jobs, cfg, data, seeds)
    summ = rp.summarize(df, ["method", "lambda_set"], ALL_METRICS)
    _save(df, summ, out, "lambda")
    rp.to_latex(summ, ["method", "lambda_set"], UTILITY[:3] + FAIRNESS[:4], out / "table_lambda.tex",
                "Effect of the exposure-gap regularisation strength $\\lambda$.", "tab:lambda")
    rp.plot_tradeoff(summ, out / "fig_lambda_tradeoff", "test_exposure_gap@10", "test_NDCG@10", "method", "lambda_set",
                     "Exposure gap@10 (lower = fairer)", "NDCG@10", "Effect of $\\lambda$ (labels = $\\lambda$)")
    return summ


def exp_privacy(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path,
                sigmas: Sequence[float] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3), **_) -> pd.DataFrame:
    out = out / "privacy"
    jobs: List[Job] = [(m, {}, {"sigma_set": 0.0}) for m in ("Vanilla-FedRec", "Fair-FedRec")]      # no-privacy references
    jobs += [(m, {"privacy.noise_multiplier": s}, {"sigma_set": s}) for m in ("DP-FedRec", "FairPriv-FedRec (ours)") for s in sigmas]
    df, _, _ = _run_jobs(jobs, cfg, data, seeds)
    summ = rp.summarize(df, ["method", "sigma_set"], ALL_METRICS)
    _save(df, summ, out, "privacy")
    rp.to_latex(summ, ["method", "sigma_set"], ["epsilon"] + PRIVACY + UTILITY[:3] + FAIRNESS[:2],
                out / "table3_privacy_utility.tex", "Table 3: privacy--utility trade-off for different noise levels $\\sigma$.", "tab:privacy")
    dp = summ[summ["method"].isin(["DP-FedRec", "FairPriv-FedRec (ours)"])]
    rp.plot_lines(dp, out / "fig_privacy_vs_sigma", "sigma_set",
                  [("test_NDCG@10", "NDCG@10"), ("priv_attack_auc", "Attack AUC (0.5 = no leakage)"),
                   ("test_exposure_gap@10", "Exposure gap@10")], "method", "Noise multiplier $\\sigma$", xlog=True)
    rp.plot_tradeoff(dp, out / "fig_utility_privacy_curve", "priv_attack_auc", "test_NDCG@10", "method", "sigma_set",
                     "Attack AUC (lower = more private)", "NDCG@10", "Utility--privacy curve (labels = $\\sigma$)")
    return summ


def exp_epsilon(cfg: ExperimentConfig, data: RecData, seeds: Sequence[int], out: Path,
                eps_grid: Sequence[float] = (1e4, 1e5, 1e6, 1e7), **_) -> pd.DataFrame:
    """Target epsilon -> sigma (accountant, delta from config, T = rounds participations) -> run."""
    out = out / "epsilon"
    T, delta, plc = cfg.train.rounds, cfg.privacy.delta, cfg.privacy.placement
    jobs: List[Job] = []
    for m in ("DP-FedRec", "FairPriv-FedRec (ours)"):
        for eps in eps_grid:
            sig = calibrate_sigma(eps, T, delta, plc)
            log.info("target epsilon %.3g (%s, T=%d) -> sigma = %.4g", eps, plc, T, sig)
            jobs.append((m, {"privacy.noise_multiplier": sig}, {"target_epsilon": eps}))
    df, _, _ = _run_jobs(jobs, cfg, data, seeds)
    summ = rp.summarize(df, ["method", "target_epsilon"], ALL_METRICS + ["sigma"])
    _save(df, summ, out, "epsilon")
    rp.to_latex(summ, ["method", "target_epsilon"], ["sigma"] + PRIVACY + UTILITY[:3] + FAIRNESS[:2],
                out / "table_epsilon.tex", "Utility, fairness and empirical leakage for different privacy budgets $\\epsilon$.", "tab:eps")
    rp.plot_lines(summ, out / "fig_epsilon", "target_epsilon",
                  [("test_NDCG@10", "NDCG@10"), ("priv_attack_auc", "Attack AUC"), ("test_exposure_gap@10", "Exposure gap@10")],
                  "method", "Target $\\epsilon$ (smaller = more private)", xlog=True)
    return summ


EXPERIMENTS = {"main": exp_main, "ablation": exp_ablation, "beta": exp_beta, "lambda": exp_lambda,
               "privacy": exp_privacy, "epsilon": exp_epsilon}
