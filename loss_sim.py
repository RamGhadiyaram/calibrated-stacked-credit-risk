"""Portfolio loss simulation on the simulated portfolio (paper Section VI-F).
For each outer fold, approve the lowest-risk 90% of applicants by the raw stack's out-of-fold
probability and compare the default rate among approved applicants with the unscreened rate.
Reads out/synth_pooled.npz, writes out/loss_sim.json."""
import os, json
import numpy as np
BASE = os.environ.get("CR_BASE", ".")
OUT = os.path.join(BASE, "out")
a = np.load(os.path.join(OUT, "synth_pooled.npz"))["Stacked_(raw)"]   # columns: y, p, fold
base, appr = [], []
for k in np.unique(a[:, 2]):
    m = a[:, 2] == k
    y, p = a[m, 0], a[m, 1]
    keep = np.argsort(p)[: int(round(0.9 * len(y)))]
    base.append(y.mean()); appr.append(y[keep].mean())
base, appr = np.array(base), np.array(appr)
red = (base - appr) / base
res = {"approve_share": 0.9, "default_rate_all": float(base.mean()),
       "default_rate_approved": float(appr.mean()), "default_rate_approved_sd": float(appr.std(ddof=1)),
       "daily_loss_before_per_1m": float(base.mean() * 1e6), "daily_loss_after_per_1m": float(appr.mean() * 1e6),
       "reduction_mean": float(red.mean()), "reduction_sd": float(red.std(ddof=1)), "folds": int(len(base))}
json.dump(res, open(os.path.join(OUT, "loss_sim.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
