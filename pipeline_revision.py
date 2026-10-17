"""Revision experiments for the credit-risk stacking paper.

Addresses reviewer points:
  R2-1  stronger tabular baselines (LightGBM, CatBoost)
  R2-3  logistic-regression benchmark
  R2-4  repeated stratified CV instead of one 80/20 split (mean, SD, CIs)
  R2-5  explanation of calibration vs PR-AUC (reported numerically)
  R2-6  reliability diagrams, ECE, calibration slope and intercept
  R1    larger real dataset (optional, see --extra)

Usage: python3 pipeline_revision.py --data german|synth [--repeats 3]
Outputs JSON + pooled out-of-fold predictions into ./out
"""
import os, sys, json, argparse, time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
BASE = os.environ.get('CR_BASE', '.')   # project root (Google Drive folder in Colab)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.config.experimental.enable_op_determinism()


def make_synth(n=6000, seed=SEED, target_rate=0.303):
    r = np.random.default_rng(seed)
    credit = np.clip(r.normal(670, 75, n), 300, 850)
    income = r.lognormal(10.8, 0.45, n)
    txn = r.gamma(4.0, 600, n)
    tariff = r.uniform(0, 100, n)
    dti = np.clip(r.normal(0.36, 0.12, n), 0.02, 0.95)
    delinq = r.poisson(1.4, n).astype(float)
    vol = np.abs(r.normal(0.3, 0.15, n))
    X = pd.DataFrame(dict(credit_score=credit, income=income, transaction_volume=txn,
                          tariff_exposure_index=tariff, debt_to_income=dti,
                          delinquency_history=delinq, alt_spend_volatility=vol))
    z = (X - X.mean()) / X.std()
    w = dict(credit_score=-1.25, income=-0.45, transaction_volume=-0.20,
             tariff_exposure_index=0.95, debt_to_income=0.75,
             delinquency_history=0.55, alt_spend_volatility=0.35)
    s = sum(w[c] * z[c] for c in w) + r.normal(0, 1.0, n)
    lo, hi = -6, 6
    for _ in range(60):
        b = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(s + b)))).mean()
        lo, hi = (b, hi) if rate < target_rate else (lo, b)
    p = 1 / (1 + np.exp(-(s + b)))
    y = (r.random(n) < np.asarray(p)).astype(int)
    return X, y


def load_taiwan(path, n_max=10000):
    """Taiwan credit-card default data (UCI id 350, 30,000 clients).
    CSV must contain a binary column named 'target' (1 = default); an optional 'ID' column is dropped.
    A stratified subsample of n_max rows keeps the nested BiLSTM stacking tractable on CPU."""
    import re
    d = pd.read_csv(path)
    if "ID" in d.columns:
        d = d.drop(columns="ID")
    if n_max and len(d) > n_max:
        d, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
            d, train_size=n_max, stratify=d["target"], random_state=SEED)
        d = d.reset_index(drop=True)
    y = d["target"].astype(int).values
    X = d.drop(columns="target")
    for c in X.columns:
        if c.lower() in ("x2", "x3", "x4", "sex", "education", "marriage"):
            X[c] = X[c].astype(int).astype(str)
    X = pd.get_dummies(X, drop_first=False).astype(float)
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


def load_german(path):
    d = pd.read_csv(path)
    y = (d["credit_risk"] == 0).astype(int).values  # 1 = default (bad)
    X = d.drop(columns="credit_risk")
    X = pd.get_dummies(X, drop_first=False).astype(float)
    import re
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


# ---------------- models ----------------
def fit_xgb(X, y):
    m = xgb.XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                          n_jobs=4, eval_metric="logloss", verbosity=0)
    return m.fit(X, y)

def fit_lgb(X, y):
    m = lgb.LGBMClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                           n_jobs=4, verbose=-1)
    return m.fit(X, y)

def fit_cat(X, y):
    m = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.05, random_seed=SEED,
                           verbose=0, thread_count=4, allow_writing_files=False)
    return m.fit(X, y)

def fit_lr(X, y):
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)


class AttnBiLSTM:
    def __init__(self, epochs=25):
        self.epochs = epochs

    def fit(self, X, y):
        keras.utils.set_random_seed(SEED)
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)[:, None, :].astype("float32")
        inp = keras.Input(shape=(1, Xs.shape[2]))
        h = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inp)
        a = layers.MultiHeadAttention(num_heads=2, key_dim=16)(h, h)
        h = layers.LayerNormalization()(layers.Add()([h, a]))
        h = layers.GlobalAveragePooling1D()(h)
        h = layers.Dropout(0.3)(layers.Dense(32, activation="relu")(h))
        out = layers.Dense(1, activation="sigmoid")(h)
        self.m = keras.Model(inp, out)
        self.m.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
        self.m.fit(Xs, np.asarray(y), epochs=self.epochs, batch_size=64, verbose=0, shuffle=False)
        return self

    def predict_proba1(self, X):
        Xs = self.sc.transform(X)[:, None, :].astype("float32")
        return self.m.predict(Xs, verbose=0).ravel()


BASES = {
    "XGBoost": (fit_xgb, lambda m, X: m.predict_proba(X)[:, 1]),
    "LightGBM": (fit_lgb, lambda m, X: m.predict_proba(X)[:, 1]),
    "CatBoost": (fit_cat, lambda m, X: m.predict_proba(X)[:, 1]),
    "Logistic": (fit_lr, lambda m, X: m.predict_proba(X)[:, 1]),
    "AttnBiLSTM": (lambda X, y: AttnBiLSTM().fit(X, y), lambda m, X: m.predict_proba1(X)),
}
STACK = ["XGBoost", "AttnBiLSTM"]  # the paper's ensemble


# ---------------- metrics ----------------
def ks(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        mk = idx == b
        if mk.any():
            e += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    return float(e)

def cal_slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    s = LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y)
    # intercept at slope fixed to 1 (calibration-in-the-large) via offset solve
    from scipy.optimize import brentq
    f = lambda a: (1 / (1 + np.exp(-(z + a)))).mean() - y.mean()
    citl = brentq(f, -10, 10)
    return float(s.coef_[0, 0]), float(citl)

def metrics(y, p):
    sl, it = cal_slope_intercept(y, p)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), ks=ks(y, p), ece=ece(y, p),
                cal_slope=sl, cal_citl=it)


def run_outer_fold(Xtr, ytr, Xte, yte):
    """Nested: base-learner OOF on training part -> meta LR + isotonic; refit bases on full train."""
    res = {}
    # base learners on test
    test_pred, oof = {}, {}
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, (fit, pred) in BASES.items():
        m = fit(Xtr, ytr)
        test_pred[name] = pred(m, Xte)
        if name in STACK:
            o = np.zeros(len(ytr))
            for a, b in inner.split(Xtr, ytr):
                mm = fit(Xtr.iloc[a], ytr[a])
                o[b] = pred(mm, Xtr.iloc[b])
            oof[name] = o
    # stacking
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    Z = np.column_stack([logit(oof[n]) for n in STACK])
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, ytr)
    raw_oof = meta.predict_proba(Z)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_oof, ytr)
    Zt = np.column_stack([logit(test_pred[n]) for n in STACK])
    raw = meta.predict_proba(Zt)[:, 1]
    cal = iso.predict(raw)
    test_pred["Stacked (raw)"] = raw
    test_pred["Stacked (isotonic)"] = cal
    # parametric alternative calibrator (Platt) for the discussion of isotonic vs sigmoid
    platt = LogisticRegression(C=1e6, max_iter=2000).fit(logit(raw_oof)[:, None], ytr)
    test_pred["Stacked (Platt)"] = platt.predict_proba(logit(raw)[:, None])[:, 1]
    res["coef"] = dict(zip(STACK, meta.coef_[0].tolist()))
    return test_pred, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["german", "synth", "taiwan"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--german_path", default=f"{BASE}/data/GermanCredit.csv")
    ap.add_argument("--taiwan_path", default=f"{BASE}/data/taiwan_credit.csv")
    ap.add_argument("--n_max", type=int, default=10000)
    ap.add_argument("--out", default=f"{BASE}/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.data == "synth":
        X, y = make_synth()
    elif a.data == "taiwan":
        X, y = load_taiwan(a.taiwan_path, a.n_max)
    else:
        X, y = load_german(a.german_path)
    print(a.data, X.shape, "default rate", y.mean(), flush=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=SEED)
    fold_metrics, pooled, coefs = [], {}, []
    t0 = time.time()
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        preds, r = run_outer_fold(X.iloc[tr], y[tr], X.iloc[te], y[te])
        coefs.append(r["coef"])
        for name, p in preds.items():
            m = metrics(y[te], p); m.update(model=name, fold=k)
            fold_metrics.append(m)
            pooled.setdefault(name, []).append(np.column_stack([y[te], p, np.full(len(te), k)]))
        print(f"fold {k+1}/{5*a.repeats} done, {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(fold_metrics)
    df.to_csv(f"{a.out}/{a.data}_fold_metrics.csv", index=False)
    np.savez_compressed(f"{a.out}/{a.data}_pooled.npz",
                        **{k.replace(" ", "_"): np.vstack(v) for k, v in pooled.items()})
    json.dump(coefs, open(f"{a.out}/{a.data}_coefs.json", "w"))
    print(df.groupby("model")[["auc", "pr_auc", "brier", "ks", "ece", "cal_slope", "cal_citl"]]
          .agg(["mean", "std"]).round(4).to_string())

if __name__ == "__main__":
    main()
