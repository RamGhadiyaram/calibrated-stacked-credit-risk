"""Analysis of the credit-risk benchmark: tables, paired tests, fairness audit, stress sensitivity,
reliability diagrams. Every number in the paper is produced (and re-checked) by functions in this file.

Usage: python analyze_benchmark.py            (reads <BASE>/out, writes <BASE>/out/results.md and figures)
"""
import os, re, json
import numpy as np, pandas as pd
from scipy import stats

BASE = os.environ.get("CR_BASE", ".")
OUT = os.path.join(BASE, "out")
DATASETS = {"german": "South German Credit", "taiwan": "Taiwan credit-card default"}
MAIN = ["Logistic", "XGBoost", "LightGBM", "CatBoost", "TabNet", "FT-Transformer", "Stack-GBM (raw)", "Stack-All (raw)"]
COMPARATORS = ["Logistic", "XGBoost", "LightGBM", "CatBoost", "TabNet", "FT-Transformer", "Stack-GBM (raw)"]
TEST_RATIO = 0.25          # n_test / n_train for 5-fold cross-validation (Nadeau-Bengio)
APPROVAL_FAIR = 0.80       # approval share used in the fairness audit
KEEP = {"german": ["age_under_25", "foreign_worker"], "taiwan": ["age_under_25", "female"]}
GROUP_LABEL = {"age_under_25": "Age under 25", "foreign_worker": "Foreign worker", "female": "Female"}


def key(name):
    return re.sub(r"[ ()-]+", "_", name).strip("_")


def load(d, out=OUT):
    f = f"{out}/{d}_fold_metrics.csv"
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def mean_sd(df, metric):
    g = df.groupby("model")[metric].agg(["mean", "std"])
    return g


def fmt_ms(m, s, nd=3):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


# ------------------------------------------------------------------ paired tests
def nb_test(diff):
    J = len(diff); se = np.sqrt((1 / J + TEST_RATIO) * diff.var(ddof=1))
    t = diff.mean() / se; p = 2 * (1 - stats.t.cdf(abs(t), J - 1)); h = stats.t.ppf(0.975, J - 1) * se
    return float(diff.mean()), float(diff.mean() - h), float(diff.mean() + h), float(p)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def paired_table(df, metric="auc", ref="Stack-All (raw)", comps=COMPARATORS):
    piv = df.pivot(index="fold", columns="model", values=metric)
    rows = []
    for c in comps:
        if c in piv.columns and ref in piv.columns:
            dm, lo, hi, p = nb_test(piv[ref] - piv[c])
            rows.append(dict(comparator=c, diff=dm, lo=lo, hi=hi, p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].values) if len(t) else []
    return t


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ fairness
def fairness(d, models=("Logistic", "CatBoost", "Stack-GBM (raw)", "Stack-All (raw)"), out=OUT):
    pooled = np.load(f"{out}/{d}_pooled.npz"); groups = pd.read_csv(f"{out}/{d}_groups.csv")
    rows = []
    for m in models:
        if key(m) not in pooled.files:
            continue
        A = pooled[key(m)]
        for g in KEEP[d]:
            gap_dp, gap_eo, gap_er = [], [], []
            for k in np.unique(A[:, 2]):
                s = A[A[:, 2] == k]; y, p, idx = s[:, 0], s[:, 1], s[:, 3].astype(int)
                gv = groups[g].values[idx]
                appr = np.zeros(len(y), bool)
                appr[np.argsort(p, kind="stable")[: int(round(APPROVAL_FAIR * len(y)))]] = True
                u, r = gv == 1, gv == 0
                if u.sum() < 5 or r.sum() < 5:
                    continue
                gap_dp.append(appr[u].mean() - appr[r].mean())
                good = y == 0; bad = y == 1
                if (u & good).sum() >= 3 and (r & good).sum() >= 3:
                    gap_eo.append(appr[u & good].mean() - appr[r & good].mean())
                if (u & bad).sum() >= 3 and (r & bad).sum() >= 3:
                    gap_er.append(appr[u & bad].mean() - appr[r & bad].mean())
            # group-wise calibration on pooled out-of-fold predictions
            y, p, idx = A[:, 0], A[:, 1], A[:, 3].astype(int); gv = groups[g].values[idx]
            from credit_benchmark import ece as _ece
            from sklearn.metrics import roc_auc_score
            rows.append(dict(model=m, group=g, n_unpriv=int((groups[g] == 1).sum()),
                             dp_gap=np.mean(gap_dp), dp_sd=np.std(gap_dp, ddof=1),
                             eo_gap=np.mean(gap_eo), eo_sd=np.std(gap_eo, ddof=1),
                             bad_gap=np.mean(gap_er), bad_sd=np.std(gap_er, ddof=1),
                             ece_unpriv=_ece(y[gv == 1], p[gv == 1]), ece_ref=_ece(y[gv == 0], p[gv == 0]),
                             auc_unpriv=roc_auc_score(y[gv == 1], p[gv == 1]),
                             auc_ref=roc_auc_score(y[gv == 0], p[gv == 0]),
                             obs_unpriv=float(y[gv == 1].mean()), obs_ref=float(y[gv == 0].mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stress
def stress_table(d, out=OUT):
    f = f"{out}/{d}_stress.csv"
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f)
    g = s.groupby(["model", "severity"]).agg(base=("mean_pd_base", "mean"), stress=("mean_pd_stress", "mean"),
                                             spearman=("spearman", "mean")).reset_index()
    g["uplift"] = g["stress"] - g["base"]
    g["uplift_rel"] = g["uplift"] / g["base"]
    return g


# ------------------------------------------------------------------ report and figures
def reliability_figure(out=OUT, path=None):
    """Reliability diagrams sized for a two-column IEEE page (text stays >= 8 pt when printed)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [d for d in DATASETS if os.path.exists(f"{out}/{d}_pooled.npz")]
    if not ds:
        return
    show = [("CatBoost", "#1f77b4", "-", "o"), ("TabNet", "#8c564b", "-", "s"), ("FT-Transformer", "#9467bd", "-", "^"),
            ("Stack-All (raw)", "#d62728", "-", "D"), ("Stack-All (isotonic)", "#2ca02c", "--", "v")]
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    fig, axs = plt.subplots(1, len(ds), figsize=(7.2, 3.3), squeeze=False)
    for ax, d in zip(axs[0], ds):
        P = np.load(f"{out}/{d}_pooled.npz")
        ax.plot([0, 1], [0, 1], color="#555", lw=1, ls=":", label="Perfect")
        top = 0.05
        for name, c, ls, mk in show:
            if key(name) not in P.files:
                continue
            A = P[key(name)]; y, p = A[:, 0], A[:, 1]
            q = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
            xs = [p[b == i].mean() for i in range(len(q) - 1) if (b == i).any()]
            ys = [y[b == i].mean() for i in range(len(q) - 1) if (b == i).any()]
            top = max(top, max(xs), max(ys))
            ax.plot(xs, ys, marker=mk, ms=4, lw=1.6, color=c, ls=ls, label=name)
        ax.set_xlabel("Mean predicted probability"); ax.set_title(DATASETS[d]); ax.grid(alpha=0.25)
        lim = min(1.0, top * 1.06)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    axs[0][0].set_ylabel("Observed default rate")
    axs[0][0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path or f"{out}/fig_reliability.png", dpi=300); plt.close(fig)


def paired_figure(out=OUT, path=None):
    """Forest plot: AUC and Brier difference of Stack-All (raw) against every comparator, 95% CI (Nadeau-Bengio)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    ds = [d for d in DATASETS if load(d) is not None]
    fig, axs = plt.subplots(2, len(ds), figsize=(7.2, 4.6), squeeze=False)
    for j, d in enumerate(ds):
        df = load(d)
        for i, (metric, lab) in enumerate((("auc", "AUC difference (higher is better)"),
                                           ("brier", "Brier difference (lower is better)"))):
            t = paired_table(df, metric)[::-1].reset_index(drop=True)
            ax = axs[i][j]
            ax.axvline(0, color="#555", lw=1)
            ax.errorbar(t["diff"], range(len(t)), xerr=[t["diff"] - t["lo"], t["hi"] - t["diff"]], fmt="o", ms=4.5,
                        color="#d62728", ecolor="#d62728", capsize=3, lw=1.4)
            ax.set_yticks(range(len(t))); ax.set_yticklabels(t["comparator"])
            ax.set_xlabel(lab); ax.grid(axis="x", alpha=0.25)
            if i == 0:
                ax.set_title(DATASETS[d])
            if j > 0:
                ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(path or f"{out}/fig_paired.png", dpi=300); plt.close(fig)


def main():
    lines = []
    for d, title in DATASETS.items():
        df = load(d)
        if df is None:
            continue
        n_folds = df.fold.nunique()
        lines.append(f"## {title} ({n_folds} outer folds)\n")
        lines.append("| Model | ROC-AUC | PR-AUC | Brier | ECE | Cal. slope | Loss reduction |\n|:--|:--|:--|:--|:--|:--|:--|")
        models = [m for m in df.model.unique()]
        for m in models:
            s = df[df.model == m]
            lines.append(f"| {m} | " + " | ".join(
                f"{s[c].mean():.3f} ± {s[c].std():.3f}" for c in ["auc", "pr_auc", "brier", "ece", "cal_slope", "loss_reduction"]) + " |")
        for metric in ("auc", "brier"):
            lines.append(f"\nPaired differences in {metric}, Stack-All (raw) minus comparator (Nadeau-Bengio, Holm-adjusted):\n")
            lines.append("| Comparator | Diff | 95% CI | p | p (Holm) |\n|:--|--:|:--|--:|--:|")
            for _, r in paired_table(df, metric).iterrows():
                lines.append(f"| {r.comparator} | {r['diff']:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | {r.p:.3f} | {r.p_holm:.3f} |")
        f = fairness(d)
        lines.append("\nFairness audit (approve lowest-risk 80%; gap = unprivileged minus reference group):\n")
        lines.append(f.round(4).to_markdown(index=False))
        st = stress_table(d)
        if st is not None:
            lines.append("\nStress sensitivity (mean predicted PD uplift, Spearman rank stability):\n")
            lines.append(st.round(4).to_markdown(index=False))
        lines.append("")
    open(f"{OUT}/results.md", "w").write("\n".join(lines))
    reliability_figure()
    paired_figure()
    print("\n".join(lines)[:6000])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
