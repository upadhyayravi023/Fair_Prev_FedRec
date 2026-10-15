"""Aggregation over seeds, LaTeX tables and publication-quality figures."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# metric -> (pretty name, higher_is_better)
PRETTY: Dict[str, Tuple[str, bool]] = {
    "test_HR@10": ("HR@10", True), "test_Recall@10": ("Recall@10", True),
    "test_NDCG@10": ("NDCG@10", True), "test_MRR@10": ("MRR@10", True),
    "test_HR@10_head": ("Recall@10 head", True), "test_HR@10_tail": ("Recall@10 tail", True),
    "test_recall_gap@10": ("Head--tail recall gap", False),
    "test_exposure_gap@10": ("Exposure gap", False), "test_tail_coverage@10": ("Tail coverage@10", True),
    "test_tail_share@10": ("Tail share@10", True), "test_coverage@10": ("Coverage@10", True),
    "test_gini@10": ("Gini@10", False), "test_ARP@10": ("ARP@10", False),
    "test_exposure_ratio@10": ("Exp. ratio@10", False), "test_pop_corr@10": ("Pop. corr@10", False),
    "priv_attack_auc": ("Attack AUC", False), "priv_attack_prec": ("Attack prec.", False),
    "epsilon": ("$\\epsilon$", False), "sigma": ("$\\sigma$", False), "beta": ("$\\beta$", True),
    "gap_lambda": ("$\\lambda$", True), "seconds": ("Time (s)", False),
}


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summarize(df: pd.DataFrame, group_cols: Sequence[str], metrics: Iterable[str]) -> pd.DataFrame:
    """mean / std over seeds for every metric (group order preserved)."""
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(list(group_cols), sort=False, dropna=False)[metrics]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=0).add_suffix("_std")
    n = g.size().rename("n_seeds")
    return pd.concat([mean, std, n], axis=1).reset_index()


def _fmt(mean: float, std: float, nd: int = 4, sci: bool = False) -> str:
    if not np.isfinite(mean):
        return "--" if np.isnan(mean) else "$\\infty$"
    if sci:
        return f"{mean:.1e}"
    return f"{mean:.{nd}f}" if not np.isfinite(std) or std == 0 else f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def to_latex(summary: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str], path: Path,
             caption: str, label: str, bold_best: bool = True, exclude_rows: Sequence[str] = ()) -> None:
    """Booktabs table 'mean ± std'; best value per column in bold (rows in ``exclude_rows`` never win)."""
    metrics = [m for m in metrics if f"{m}_mean" in summary.columns]
    eligible = ~summary[group_cols[0]].astype(str).isin(list(exclude_rows))
    best = {}
    for m in metrics:
        if m in {"epsilon", "sigma", "beta", "gap_lambda", "seconds"}:
            continue
        col = summary[f"{m}_mean"].replace([np.inf, -np.inf], np.nan).where(eligible)
        if col.notna().any():
            best[m] = col.idxmax() if PRETTY.get(m, (m, True))[1] else col.idxmin()
    header = " & ".join([c.replace("_", " ").title() for c in group_cols] + [PRETTY.get(m, (m, True))[0] for m in metrics])
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + "l" * len(group_cols) + "c" * len(metrics) + "}",
             r"\toprule", header + r" \\", r"\midrule"]
    for idx, row in summary.iterrows():
        cells = [str(row[c]) for c in group_cols]
        for m in metrics:
            sci = m == "epsilon"
            nd = 2 if m in {"test_ARP@10", "test_exposure_ratio@10", "test_exposure_gap@10", "seconds"} else 4
            cell = _fmt(row[f"{m}_mean"], row.get(f"{m}_std", 0.0), nd, sci)
            if bold_best and best.get(m) == idx:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells).replace("_", r"\_") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_training_curves(histories: Dict[str, List[dict]], path: Path, k: int = 10) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for label, hist in histories.items():
        if not hist:
            continue
        x = [h["round"] for h in hist]
        axes[0].plot(x, [h[f"val_NDCG@{k}"] for h in hist], marker="o", ms=3, label=label)
        axes[1].plot(x, [h[f"val_exposure_gap@{k}"] for h in hist], marker="o", ms=3, label=label)
    axes[0].set(xlabel="Round / epoch", ylabel=f"Validation NDCG@{k}", title="Accuracy during training")
    axes[1].set(xlabel="Round / epoch", ylabel=f"Validation exposure gap@{k}", title="Exposure gap during training (lower = fairer)")
    axes[0].legend(fontsize=6)
    save_fig(fig, path)


def plot_exposure(exposures: Dict[str, np.ndarray], head_mask: np.ndarray, path: Path, k: int = 10) -> None:
    """Left: Lorenz curves of item exposure. Right: head/tail share of recommendation slots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, counts in exposures.items():
        x = np.sort(counts)
        cum = np.cumsum(x) / x.sum()
        axes[0].plot(np.linspace(0, 1, len(x) + 1), np.r_[0, cum], label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect equality")
    axes[0].set(xlabel="Cumulative share of items (least $\\rightarrow$ most exposed)", ylabel="Cumulative share of exposure",
                title=f"Lorenz curve of top-{k} exposure")
    axes[0].legend(fontsize=6)
    labels = list(exposures)
    head = np.array([exposures[l][head_mask].sum() / exposures[l].sum() for l in labels])
    xpos = np.arange(len(labels))
    axes[1].bar(xpos, head, label="head items")
    axes[1].bar(xpos, 1 - head, bottom=head, label="tail items")
    axes[1].axhline(head_mask.mean(), color="k", ls="--", lw=0.8, label="catalogue share of head")
    axes[1].set_xticks(xpos, labels, rotation=35, ha="right", fontsize=7)
    axes[1].set(ylabel="Share of recommendation slots", title=f"Head vs tail exposure (top-{k})")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    save_fig(fig, path)


def plot_tradeoff(summary: pd.DataFrame, path: Path, x: str, y: str, series_col: str, label_col: str,
                  xlabel: str, ylabel: str, title: str) -> None:
    """Scatter/line trade-off with error bars; each point annotated with ``label_col``."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, g in summary.groupby(series_col, sort=False):
        g = g.sort_values(label_col)
        ax.errorbar(g[f"{x}_mean"], g[f"{y}_mean"], xerr=g[f"{x}_std"], yerr=g[f"{y}_std"], marker="o", ms=4,
                    capsize=2, label=name)
        for xi, yi, lab in zip(g[f"{x}_mean"], g[f"{y}_mean"], g[label_col]):
            ax.annotate(f"{lab:g}", (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=7)
    save_fig(fig, path)


def plot_lines(summary: pd.DataFrame, path: Path, x_col: str, ys: Sequence[Tuple[str, str]], series_col: str,
               xlabel: str, xlog: bool = False, title: Optional[str] = None) -> None:
    """One panel per (metric, ylabel) in ``ys`` against a sweep variable."""
    fig, axes = plt.subplots(1, len(ys), figsize=(4.4 * len(ys), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (y, ylabel) in zip(axes, ys):
        for name, g in summary.groupby(series_col, sort=False):
            g = g.sort_values(x_col)
            xs = g[x_col].replace([0, np.inf], np.nan) if xlog else g[x_col]
            ax.errorbar(xs, g[f"{y}_mean"], yerr=g[f"{y}_std"], marker="o", ms=4, capsize=2, label=name)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if xlog:
            ax.set_xscale("log")
    axes[0].legend(fontsize=7)
    if title:
        fig.suptitle(title, y=1.02)
    save_fig(fig, path)
