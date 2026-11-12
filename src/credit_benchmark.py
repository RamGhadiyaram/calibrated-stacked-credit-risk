"""Credit-risk benchmark: gradient boosting, tabular deep learning and stacked ensembles.

For each dataset this script runs repeated, stratified, nested cross-validation and saves
  <out>/<data>_fold_metrics.csv     per-fold metrics for every model
  <out>/<data>_pooled.npz           out-of-fold predictions (columns: y, p, fold, row)
  <out>/<data>_groups.csv           protected-attribute groups per row (for the fairness audit)
  <out>/<data>_stress.csv           per-fold stress-sensitivity results
  <out>/<data>_coefs.json           meta-learner coefficients per fold

Models: Logistic regression, XGBoost, LightGBM, CatBoost, TabNet, FT-Transformer, and
logistic-meta-learner stacks (raw, Platt and isotonic calibration).

Usage:
  python credit_benchmark.py --data german --repeats 3
  python credit_benchmark.py --data taiwan --repeats 2 --n_max 30000
"""
import os, re, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn

BASE = os.environ.get("CR_BASE", ".")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))


def seed_all(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- data
def clean_cols(X):
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)) for c in X.columns]
    return X.loc[:, ~X.columns.duplicated()]


def load_german(path):
    """South German Credit (corrected). 1000 loans, 20 attributes. credit_risk: 1 = good, 0 = bad."""
    d = pd.read_csv(path)
    y = (d["credit_risk"] == 0).astype(int).values            # 1 = default
    cat = [c for c in d.columns if c not in ("laufzeit", "hoehe", "alter", "credit_risk")]
    # laufzeit, hoehe, alter are the genuinely numeric attributes. The ordinal items
    # (rate, wohnzeit, bishkred, pers) are treated as categorical codes and one-hot encoded.
    raw = d.drop(columns="credit_risk").copy()
    for c in cat:
        raw[c] = raw[c].astype(int).astype(str)
    X = clean_cols(pd.get_dummies(raw, drop_first=False).astype(float))
    foreign_val = d["gastarb"].value_counts().idxmax()          # majority code = foreign worker
    groups = pd.DataFrame({
        "age_under_25": (d["alter"] < 25).astype(int),
        "foreign_worker": (d["gastarb"] == foreign_val).astype(int),
    })
    return X, y, groups, d.copy()


def load_taiwan(path, n_max=30000):
    """Taiwan credit-card default (UCI id 350). X2 sex, X3 education, X4 marriage, X5 age."""
    d = pd.read_csv(path)
    if "ID" in d.columns:
        d = d.drop(columns="ID")
    if n_max and len(d) > n_max:
        d, _ = train_test_split(d, train_size=n_max, stratify=d["target"], random_state=SEED)
    d = d.reset_index(drop=True)
    y = d["target"].astype(int).values
    raw = d.drop(columns="target").copy()
    for c in ("X2", "X3", "X4"):
        raw[c] = raw[c].astype(int).astype(str)
    X = clean_cols(pd.get_dummies(raw, drop_first=False).astype(float))
    groups = pd.DataFrame({
        "age_under_25": (d["X5"] < 25).astype(int),
        "female": (d["X2"] == 2).astype(int),
    })
    return X, y, groups, d.copy()


def stress_german(Xdf, s):
    """Stylised cost-of-living shock of severity s on the encoded German features:
    loan amount (hoehe) rises by s. The shock leaves the labels untouched."""
    Z = Xdf.copy()
    Z["hoehe"] = Z["hoehe"] * (1 + s)
    return Z


def stress_taiwan(Xdf, s):
    """Stylised shock: monthly repayments (X18-X23) fall by s, outstanding bills (X12-X17) rise by s."""
    Z = Xdf.copy()
    for c in [f"X{i}" for i in range(18, 24)]:
        Z[c] = Z[c] * (1 - s)
    for c in [f"X{i}" for i in range(12, 18)]:
        Z[c] = Z[c] * (1 + s)
    return Z


# ----------------------------------------------------------------------------- models
def fit_lr(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)

def fit_xgb(X, y):
    return xgb.XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                             n_jobs=os.cpu_count(), eval_metric="logloss", verbosity=0).fit(X, y)

def fit_lgb(X, y):
    return lgb.LGBMClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=os.cpu_count(), verbose=-1).fit(X, y)

def fit_cat(X, y):
    return CatBoostClassifier(iterations=350, depth=4, learning_rate=0.05, random_seed=SEED,
                              verbose=0, thread_count=os.cpu_count(), allow_writing_files=False).fit(X, y)


class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al., 2021): every feature becomes a token, a [CLS] token is
    prepended, and a small pre-norm Transformer encoder feeds a linear head."""
    def __init__(self, n_feat, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], dim=1)
        return self.head(self.enc(t)[:, 0]).squeeze(-1)


class FTT:
    def __init__(self, epochs=40, patience=6, batch=512, lr=1e-3):
        self.epochs, self.patience, self.batch, self.lr = epochs, patience, batch, lr

    def fit(self, X, y):
        seed_all()
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        self.sc = StandardScaler().fit(X)
        Xs = np.clip(self.sc.transform(X), -5, 5).astype(np.float32)
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=SEED)
        self.net = FTTransformer(Xs.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        xt = torch.tensor(Xs[tr], device=DEVICE); yt = torch.tensor(y[tr], dtype=torch.float32, device=DEVICE)
        xv = torch.tensor(Xs[va], device=DEVICE); yv = torch.tensor(y[va], dtype=torch.float32, device=DEVICE)
        best, best_state, bad = 1e9, None, 0
        g = torch.Generator().manual_seed(SEED)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(tr), generator=g)
            for i in range(0, len(tr), self.batch):
                idx = perm[i:i + self.batch].to(DEVICE)
                opt.zero_grad(); lossf(self.net(xt[idx]), yt[idx]).backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = lossf(self.net(xv), yv).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        Xs = np.clip(self.sc.transform(np.asarray(X, dtype=np.float32)), -5, 5).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor(Xs, device=DEVICE))).cpu().numpy()


class TabNetModel:
    def __init__(self, max_epochs=60, patience=8):
        self.max_epochs, self.patience = max_epochs, patience

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier
        seed_all()
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=SEED)
        bs = int(min(1024, max(64, len(tr) // 4)))
        self.m = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.5, seed=SEED, verbose=0,
                                  optimizer_params=dict(lr=2e-2), device_name=DEVICE)
        self.m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric=["logloss"],
                   max_epochs=self.max_epochs, patience=self.patience, batch_size=bs,
                   virtual_batch_size=min(128, bs), drop_last=False)
        return self

    def predict(self, X):
        return self.m.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]


BASES = {
    "Logistic": (fit_lr, lambda m, X: m.predict_proba(X)[:, 1]),
    "XGBoost": (fit_xgb, lambda m, X: m.predict_proba(X)[:, 1]),
    "LightGBM": (fit_lgb, lambda m, X: m.predict_proba(X)[:, 1]),
    "CatBoost": (fit_cat, lambda m, X: m.predict_proba(X)[:, 1]),
    "TabNet": (lambda X, y: TabNetModel().fit(X, y), lambda m, X: m.predict(X)),
    "FT-Transformer": (lambda X, y: FTT().fit(X, y), lambda m, X: m.predict(X)),
}
GBM_STACK = ["Logistic", "XGBoost", "LightGBM", "CatBoost"]
ALL_STACK = GBM_STACK + ["TabNet", "FT-Transformer"]


# ----------------------------------------------------------------------------- metrics
def ks_stat(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))

def slope_citl(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
    sl = LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y).coef_[0, 0]
    citl = brentq(lambda a: (1 / (1 + np.exp(-(z + a)))).mean() - y.mean(), -10, 10)
    return float(sl), float(citl)

def approve_default_rate(y, p, share=0.9):
    """Approve the lowest-risk `share` of applicants; return the default rate among the approved."""
    keep = np.argsort(p)[: int(round(share * len(y)))]
    return float(y[keep].mean())

def metrics(y, p):
    sl, ci = slope_citl(y, p)
    base = y.mean(); adr = approve_default_rate(y, p)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), ks=ks_stat(y, p), ece=ece(y, p),
                cal_slope=sl, cal_citl=ci, base_rate=base, approved_default_rate=adr,
                loss_reduction=(base - adr) / base)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- one outer fold
def run_outer_fold(Xtr, ytr, Xte, stress_fn, sev, use_deep):
    names = list(BASES) if use_deep else [n for n in BASES if n not in ("TabNet", "FT-Transformer")]
    stack_all = [n for n in ALL_STACK if n in names]
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models, oof = {}, {n: np.zeros(len(ytr)) for n in names}
    for n in names:
        fit, pred = BASES[n]
        models[n] = fit(Xtr, ytr)
        for a, b in inner.split(Xtr, ytr):
            oof[n][b] = pred(fit(Xtr.iloc[a], ytr[a]), Xtr.iloc[b])

    def predict_all(X):
        P = {n: BASES[n][1](models[n], X) for n in names}
        return P

    stacks = {}
    for label, members in (("Stack-GBM", GBM_STACK), ("Stack-All", stack_all)):
        if label == "Stack-All" and not use_deep:
            continue
        Z = np.column_stack([logit(oof[n]) for n in members])
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, ytr)
        raw_oof = meta.predict_proba(Z)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_oof, ytr)
        platt = LogisticRegression(C=1e6, max_iter=2000).fit(logit(raw_oof)[:, None], ytr)
        stacks[label] = (members, meta, iso, platt)

    def predict_stacks(P):
        out = {}
        for label, (members, meta, iso, platt) in stacks.items():
            raw = meta.predict_proba(np.column_stack([logit(P[n]) for n in members]))[:, 1]
            out[f"{label} (raw)"] = raw
            out[f"{label} (Platt)"] = platt.predict_proba(logit(raw)[:, None])[:, 1]
            out[f"{label} (isotonic)"] = iso.predict(raw)
        return out

    P = predict_all(Xte); P.update(predict_stacks(P))
    coefs = {label: dict(zip(members, meta.coef_[0].round(4).tolist()))
             for label, (members, meta, _, _) in stacks.items()}
    stress_rows = []
    for s in sev:
        Ps = predict_all(stress_fn(Xte, s)); Ps.update(predict_stacks(Ps))
        for n in P:
            stress_rows.append(dict(model=n, severity=s, mean_pd_base=float(P[n].mean()),
                                    mean_pd_stress=float(Ps[n].mean()),
                                    spearman=float(spearmanr(P[n], Ps[n])[0])))
    return P, coefs, stress_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["german", "taiwan"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--german_path", default=f"{BASE}/data/GermanCredit.csv")
    ap.add_argument("--taiwan_path", default=f"{BASE}/data/taiwan_credit.csv")
    ap.add_argument("--n_max", type=int, default=30000)
    ap.add_argument("--no_deep", action="store_true", help="skip TabNet and FT-Transformer")
    ap.add_argument("--max_folds", type=int, default=0, help="debug: stop after this many folds")
    ap.add_argument("--out", default=f"{BASE}/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); seed_all()
    if a.data == "german":
        X, y, groups, _ = load_german(a.german_path); stress_fn = stress_german
    else:
        X, y, groups, _ = load_taiwan(a.taiwan_path, a.n_max); stress_fn = stress_taiwan
    groups.to_csv(f"{a.out}/{a.data}_groups.csv", index=False)
    print(a.data, X.shape, "default rate", round(y.mean(), 4), "device", DEVICE, flush=True)

    sev = [0.1, 0.2, 0.3]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=SEED)
    fold_rows, pooled, coefs, stress_all = [], {}, [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        if a.max_folds and k >= a.max_folds:
            break
        P, c, st = run_outer_fold(X.iloc[tr], y[tr], X.iloc[te], stress_fn, sev, not a.no_deep)
        coefs.append(c)
        for r in st:
            r["fold"] = k; stress_all.append(r)
        for n, p in P.items():
            m = metrics(y[te], p); m.update(model=n, fold=k); fold_rows.append(m)
            pooled.setdefault(n, []).append(np.column_stack([y[te], p, np.full(len(te), k), te]))
        print(f"fold {k+1}/{5*a.repeats} done, {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(fold_rows)
    df.to_csv(f"{a.out}/{a.data}_fold_metrics.csv", index=False)
    pd.DataFrame(stress_all).to_csv(f"{a.out}/{a.data}_stress.csv", index=False)
    np.savez_compressed(f"{a.out}/{a.data}_pooled.npz",
                        **{re.sub(r"[ ()-]+", "_", k).strip("_"): np.vstack(v) for k, v in pooled.items()})
    json.dump(coefs, open(f"{a.out}/{a.data}_coefs.json", "w"))
    cols = ["auc", "pr_auc", "brier", "ece", "cal_slope", "loss_reduction"]
    print(df.groupby("model")[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
