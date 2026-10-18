"""Regenerate Figures 4 and 5 (TreeExplainer on XGBoost, KernelSHAP on the full calibrated stack)
and the portfolio-loss simulation for the synthetic portfolio, using the revision pipeline.

Run from the folder that contains pipeline_revision.py:  python shap_figs.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import pipeline_revision as pr

BASE = os.environ.get("CR_BASE", ".")
OUT = os.path.join(BASE, "out_shap")
os.makedirs(OUT, exist_ok=True)
SEED = pr.SEED

X, y = pr.make_synth()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

xgbm = pr.fit_xgb(Xtr, ytr)
bil = pr.AttnBiLSTM().fit(Xtr, ytr)

# out-of-fold predictions on the training part for the meta-learner and isotonic calibrator
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_x, oof_b = np.zeros(len(ytr)), np.zeros(len(ytr))
for a, b in inner.split(Xtr, ytr):
    oof_x[b] = pr.fit_xgb(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
    oof_b[b] = pr.AttnBiLSTM().fit(Xtr.iloc[a], ytr[a]).predict_proba1(Xtr.iloc[b])

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

Z = np.column_stack([logit(oof_x), logit(oof_b)])
meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, ytr)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(meta.predict_proba(Z)[:, 1], ytr)
cols = list(X.columns)

def stack_predict(arr):
    d = pd.DataFrame(arr, columns=cols)
    z = np.column_stack([logit(xgbm.predict_proba(d)[:, 1]), logit(bil.predict_proba1(d))])
    return iso.predict(meta.predict_proba(z)[:, 1])

p_te = stack_predict(Xte.values)
from sklearn.metrics import roc_auc_score
print("held-out AUC (this split):", round(roc_auc_score(yte, p_te), 4), flush=True)

# Figure 4: TreeExplainer on XGBoost
sv = shap.TreeExplainer(xgbm).shap_values(Xte)
plt.figure()
shap.summary_plot(sv, Xte, show=False, max_display=7)
fig = plt.gcf(); fig.set_size_inches(5.2, 3.6)
plt.tight_layout(); fig.savefig(f"{OUT}/fig4_treeshap_xgboost.png", dpi=300); plt.close("all")
tree_rank = dict(zip(cols, np.abs(sv).mean(0).round(4).tolist()))

# Figure 5: KernelSHAP on the whole calibrated stack (30 background, 40 explained, as in the paper)
bg = shap.sample(Xtr, 30, random_state=SEED)
ex = Xte.sample(40, random_state=SEED)
kex = shap.KernelExplainer(stack_predict, bg.values)
ksv = kex.shap_values(ex.values, nsamples=500, silent=True)
plt.figure()
shap.summary_plot(ksv, ex, show=False, max_display=7)
fig = plt.gcf(); fig.set_size_inches(5.2, 3.6)
plt.tight_layout(); fig.savefig(f"{OUT}/fig5_kernelshap_ensemble.png", dpi=300); plt.close("all")
kern_rank = dict(zip(cols, np.abs(ksv).mean(0).round(4).tolist()))

json.dump({"treeshap_mean_abs": tree_rank, "kernelshap_mean_abs": kern_rank,
           "heldout_auc_this_split": float(roc_auc_score(yte, p_te))},
          open(f"{OUT}/shap_rankings.json", "w"), indent=1)
print("tree:", sorted(tree_rank.items(), key=lambda t: -t[1]))
print("kernel:", sorted(kern_rank.items(), key=lambda t: -t[1]))
