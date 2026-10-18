"""Check every number in the paper's results tables against the notebook outputs.
Usage: python verify_paper.py paper.tex out
Reads out/{synth,german,taiwan}_fold_metrics.csv (and out_shap/shap_rankings.json if present)."""
import sys, re, json, os
import numpy as np, pandas as pd
from scipy import stats

tex = open(sys.argv[1]).read(); OUT = sys.argv[2]
MET = ["auc","pr_auc","brier","ks","ece","cal_slope"]
ROWS = {"Logistic":"Logistic","XGBoost":"XGBoost","LightGBM":"LightGBM","CatBoost":"CatBoost",
        "Attn-BiLSTM":"AttnBiLSTM","Stacked (raw)":"Stacked (raw)","Stacked (isotonic)":"Stacked (isotonic)"}
DS = ["synth","german","taiwan"]
dfs = {d: pd.read_csv(f"{OUT}/{d}_fold_metrics.csv") for d in DS if os.path.exists(f"{OUT}/{d}_fold_metrics.csv")}
for d in DS:
    if d not in dfs: print(f"note: {d}_fold_metrics.csv not found, {d} not checked")
ok = bad = 0
def check(label, paper, comp):
    global ok, bad
    good = paper == comp
    ok += good; bad += (not good)
    if not good: print(f"MISMATCH {label}: paper={paper!r} recomputed={comp!r}")

# split tex into table blocks I-V in order
tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, flags=re.S)
def rows_of(t):
    body = t.split("\\midrule")[1].split("\\bottomrule")[0]
    return [[c.strip() for c in r.split("&")] for r in body.split("\\\\") if r.strip()]

# Tables I-III
for ti, d in enumerate(DS):
    if d not in dfs: continue
    g = dfs[d].groupby("model")[MET].agg(["mean","std"])
    n_folds = dfs[d].fold.nunique()
    for r in rows_of(tables[ti]):
        m = ROWS[r[0]]
        for k, cell in zip(MET, r[1:]):
            comp = f"{g.loc[m,(k,'mean')]:.3f} $\\pm$ {g.loc[m,(k,'std')]:.3f}"
            check(f"Table {ti+1} {r[0]} {k}", cell, comp)
    cap = re.search(r"\\caption\{(.*?)\}", tables[ti]).group(1)
    check(f"Table {ti+1} fold count in caption", f"{n_folds} outer folds" in cap.lower(), True)

# Table IV (Nadeau-Bengio, ratio n_test/n_train = 1/4)
def nb(dd):
    J = len(dd); se = np.sqrt((1/J + 0.25)*dd.var(ddof=1)); t = dd.mean()/se
    p = 2*(1 - stats.t.cdf(abs(t), J-1)); h = stats.t.ppf(.975, J-1)*se
    return dd.mean(), dd.mean()-h, dd.mean()+h, p
def fmt_p(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
for r in rows_of(tables[3]):
    m = ROWS[r[0]]
    for d, cell in zip(DS, r[1:]):
        if d not in dfs: continue
        piv = dfs[d].pivot(index="fold", columns="model", values="auc")
        dm, lo, hi, p = nb(piv["Stacked (raw)"] - piv[m])
        comp = f"{dm:+.4f} [{lo:+.4f}, {hi:+.4f}], {fmt_p(p)}"
        check(f"Table IV {r[0]} {d}", cell, comp)

# Prose claims
for d, ab in zip(DS, ["0.824","0.793","0.767"]):
    if d not in dfs: continue
    s = dfs[d].query("model=='Stacked (raw)'")
    check(f"stack AUC {d} in text", ab in tex, True)
    check(f"stack AUC {d} value", f"{s.auc.mean():.3f}", ab)
exp = dict(zip(DS, ["1.02","1.09","1.03"]))
for d in dfs: check(f"stack slope {d}", f"{dfs[d].query('model==\"Stacked (raw)\"').cal_slope.mean():.2f}", exp[d])
for d, v in zip(DS, ["0.017","0.032","0.017"]):
    if d not in dfs: continue
    piv = dfs[d].pivot(index="fold", columns="model", values="pr_auc")
    diff = piv["Stacked (isotonic)"] - piv["Stacked (raw)"]
    check(f"PR-AUC drop {d}", f"{-diff.mean():.3f}", v)
    check(f"PR-AUC drop all folds negative {d}", bool((diff < 0).all()), True)
for d in dfs:   # within 0.005 of best single model
    g = dfs[d].groupby("model").auc.mean()
    best = g[["Logistic","XGBoost","LightGBM","CatBoost","AttnBiLSTM"]].max()
    check(f"within 0.005 of best {d}", bool(best - g["Stacked (raw)"] <= 0.005 + 1e-9), True)
# Bonferroni count: comparisons with p < 0.05/15
surv = 0
for d in dfs:
    piv = dfs[d].pivot(index="fold", columns="model", values="auc")
    for m in ["Logistic","XGBoost","LightGBM","CatBoost","AttnBiLSTM"]:
        surv += nb(piv["Stacked (raw)"] - piv[m])[3] < 0.05/15
if len(dfs)==3: check("Bonferroni survivors = 4", int(surv), 4)
else: print("note: Bonferroni count needs all three datasets")
# SHAP ranking and loss numbers if available
sp = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "out_shap", "shap_rankings.json")
if os.path.exists(sp):
    j = json.load(open(sp)); t = j["treeshap_mean_abs"]
    top = sorted(t, key=t.get, reverse=True)[:3]
    check("TreeSHAP top3", top, ["credit_score","tariff_exposure_index","debt_to_income"])
    check("TreeSHAP values", [f"{t[k]:.2f}" for k in top], ["0.82","0.69","0.52"])
    check("single split AUC 0.849", f"{j['heldout_auc_this_split']:.3f}", "0.849")
else:
    print("note: out_shap/shap_rankings.json not found, SHAP numbers not checked")
print(f"\n{ok} checks passed, {bad} mismatches")
