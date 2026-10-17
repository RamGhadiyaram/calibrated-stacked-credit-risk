import os, sys, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("CR_BASE", ".")
OUT = os.path.join(BASE, "out")
ORDER = ["Logistic", "XGBoost", "LightGBM", "CatBoost", "AttnBiLSTM",
         "Stacked (raw)", "Stacked (isotonic)", "Stacked (Platt)"]
MET = ["auc", "pr_auc", "brier", "ks", "ece", "cal_slope", "cal_citl"]
NAMES = {"auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier", "ks": "KS",
         "ece": "ECE", "cal_slope": "Cal. slope", "cal_citl": "CITL"}


def table(df):
    g = df.groupby("model")[MET].agg(["mean", "std"])
    rows = []
    for m in ORDER:
        if m not in g.index: continue
        r = {"Model": m}
        for k in MET:
            r[NAMES[k]] = f"{g.loc[m,(k,'mean')]:.3f} ± {g.loc[m,(k,'std')]:.3f}"
        rows.append(r)
    return pd.DataFrame(rows)


def nb_test(d, n_folds_per_repeat=5):
    """Nadeau-Bengio corrected resampled t-test on paired per-fold differences."""
    d = np.asarray(d); J = len(d)
    ratio = 1.0 / (n_folds_per_repeat - 1)  # n_test / n_train = 1/4 for 5-fold
    se = np.sqrt((1.0 / J + ratio) * d.var(ddof=1))
    t = d.mean() / se
    p = 2 * (1 - stats.t.cdf(abs(t), J - 1))
    h = stats.t.ppf(0.975, J - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, p


def paired(df, ref):
    piv = df.pivot(index="fold", columns="model", values="auc")
    rows = []
    for m in ["Logistic", "XGBoost", "LightGBM", "CatBoost", "AttnBiLSTM"]:
        dm, lo, hi, p = nb_test(piv[ref] - piv[m])
        rows.append({"Comparison": f"{ref} minus {m}", "ΔAUC": f"{dm:+.4f}",
                     "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "p (NB-corrected)": f"{p:.3f}"})
    return pd.DataFrame(rows)


def reliability(ax, pooled, key, label, bins=10):
    a = pooled[key]; y, p = a[:, 0], a[:, 1]
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    xs, ys = [], []
    for b in range(len(q) - 1):
        mk = idx == b
        if mk.sum() > 0:
            xs.append(p[mk].mean()); ys.append(y[mk].mean())
    ax.plot(xs, ys, marker="o", ms=4, lw=1.3, label=label)


def main():
    md = []
    for ds, title in [("synth", "Simulated tariff-stress portfolio (n=6,000)"),
                      ("german", "South German Credit (n=1,000)"),
                      ("taiwan", "Taiwan credit-card default (stratified n=10,000)")]:
        try:
            df = pd.read_csv(f"{OUT}/{ds}_fold_metrics.csv")
        except FileNotFoundError:
            continue
        md.append(f"## {title}\n")
        md.append(table(df).to_markdown(index=False) + "\n")
        md.append("Paired AUC differences, stacked (isotonic) vs each model:\n")
        md.append(paired(df, "Stacked (isotonic)").to_markdown(index=False) + "\n")
        md.append("Paired AUC differences, stacked (raw) vs each model:\n")
        md.append(paired(df, "Stacked (raw)").to_markdown(index=False) + "\n")
        # PR-AUC change from calibration
        piv = df.pivot(index="fold", columns="model", values="pr_auc")
        d = piv["Stacked (isotonic)"] - piv["Stacked (raw)"]
        md.append(f"PR-AUC change from isotonic calibration: mean {d.mean():+.4f}, "
                  f"SD {d.std(ddof=1):.4f}, share of folds lower {(d<0).mean():.2f}\n")
        try:
            coefs = json.load(open(f"{OUT}/{ds}_coefs.json"))
            c = pd.DataFrame(coefs).mean()
            md.append("Mean meta-learner coefficients (logit inputs): " +
                      ", ".join(f"{k} {v:.2f}" for k, v in c.items()) + "\n")
        except FileNotFoundError:
            pass
        pooled = np.load(f"{OUT}/{ds}_pooled.npz")
        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        for key, lab in [("AttnBiLSTM", "Attn-BiLSTM"), ("Stacked_(raw)", "Stacked raw"),
                         ("Stacked_(isotonic)", "Stacked isotonic")]:
            reliability(ax, pooled, key, lab)
        ax.set_xlabel("Mean predicted default probability")
        ax.set_ylabel("Observed default rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.tick_params(labelsize=7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{OUT}/reliability_{ds}.png")
        plt.close(fig)
    open(f"{OUT}/results.md", "w").write("\n".join(md))
    print("\n".join(md))

main()
