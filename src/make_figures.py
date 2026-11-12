"""Paper-size figures (text stays 8 pt or larger when printed in a two-column IEEE page).
Writes out/fig_reliability.png and out/fig_paired.png. Run after the benchmark: python make_figures.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_benchmark as A
from analyze_benchmark import OUT, DATASETS, key, load, paired_table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})


def reliability_figure(out=OUT, path=None):
    ds = [d for d in DATASETS if os.path.exists(f"{out}/{d}_pooled.npz")]
    show = [("CatBoost", "#1f77b4", "-", "o"), ("TabNet", "#8c564b", "-", "s"), ("FT-Transformer", "#9467bd", "-", "^"),
            ("Stack-All (raw)", "#d62728", "-", "D"), ("Stack-All (isotonic)", "#2ca02c", "--", "v")]
    fig, axs = plt.subplots(1, len(ds), figsize=(7.2, 3.3), squeeze=False)
    for ax, d in zip(axs[0], ds):
        P = np.load(f"{out}/{d}_pooled.npz")
        ax.plot([0, 1], [0, 1], color="#555", lw=1, ls=":", label="Perfect")
        top = 0.05
        for name, c, ls, mk in show:
            if key(name) not in P.files:
                continue
            Arr = P[key(name)]; y, p = Arr[:, 0], Arr[:, 1]
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
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if metric == "brier" and d == "taiwan" else "%.2f"))
            if i == 0:
                ax.set_title(DATASETS[d])
            if j > 0:
                ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(path or f"{out}/fig_paired.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    reliability_figure(); paired_figure()
    print("wrote", [f for f in sorted(os.listdir(OUT)) if f.startswith("fig_")])
