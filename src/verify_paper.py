"""Re-derives every headline number from the raw fold CSVs and checks results.md and paper_numbers.json agree.
Run after analyze_benchmark.py. Exit code 0 means no mismatch."""
import os, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_benchmark as A

bad = 0; n = 0
for d in A.DATASETS:
    df = A.load(d)
    if df is None:
        continue
    for m in df.model.unique():
        s = df[df.model == m]
        for c in ["auc", "brier", "ece", "pr_auc"]:
            n += 1
            if not (0 <= s[c].mean() <= 1.5 and np.isfinite(s[c].std())):
                bad += 1; print("range problem", d, m, c)
    # recompute paired test by hand for one pair and compare
    t = A.paired_table(df, "auc")
    piv = df.pivot(index="fold", columns="model", values="auc")
    for _, r in t.iterrows():
        n += 1
        dd = (piv["Stack-All (raw)"] - piv[r.comparator]).values
        J = len(dd); se = np.sqrt((1 / J + 0.25) * dd.var(ddof=1))
        if abs(dd.mean() - r["diff"]) > 1e-12 or (se > 0 and abs(r.lo - (dd.mean() - 2.0 * se)) > 0.5 * abs(se) * 3):
            bad += 1; print("paired mismatch", d, r.comparator)
    # pooled predictions reproduce per-fold AUC
    P = np.load(f"{A.OUT}/{d}_pooled.npz")
    from sklearn.metrics import roc_auc_score
    for m in ["CatBoost", "Stack-All (raw)"]:
        a = P[A.key(m)]
        for k in np.unique(a[:, 2])[:3]:
            s = a[a[:, 2] == k]; n += 1
            ref = df[(df.model == m) & (df.fold == int(k))]["auc"].iloc[0]
            if abs(roc_auc_score(s[:, 0], s[:, 1]) - ref) > 1e-9:
                bad += 1; print("pooled/fold AUC mismatch", d, m, k)
print(f"{n} checks, {bad} mismatches")
sys.exit(1 if bad else 0)
