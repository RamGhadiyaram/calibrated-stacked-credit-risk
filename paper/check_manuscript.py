"""Checks that the numbers in the manuscript agree with results/results.md (the output of analyze_benchmark.py).
1) every table in tables/*.tex must equal what build_tables.py produces from results.md;
2) every number quoted in the abstract and prose is re-derived from results.md and compared.
Exit code 0 means no mismatch. Usage: python check_manuscript.py [path/to/results.md]"""
import re, sys, subprocess, os, filecmp, tempfile, shutil
here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "results", "results.md")
txt = open(res).read()
bad = 0; n = 0
def chk(label, got, want, tol=5e-4):
    global bad, n
    n += 1
    if abs(got - want) > tol:
        bad += 1; print("MISMATCH", label, "results.md:", got, "paper:", want)

secs = {("german" if s.startswith("South") else "taiwan"): s for s in re.split(r"^## ", txt, flags=re.M)[1:]}
def main_row(d, model):
    for line in secs[d].split("\n"):
        if line.startswith(f"| {model} |") and "±" in line:
            return [float(x.split(" ± ")[0]) for x in line.strip("|").split("|")[1:]]
def paired(d, metric, comp):
    s = secs[d]; i = s.index(f"Paired differences in {metric}")
    for line in s[i:].split("\n")[4:]:
        if line.startswith(f"| {comp} |"):
            c = [x.strip() for x in line.strip("|").split("|")]
            return float(c[1]), float(c[3]), float(c[4])
def fair(d, model, group, col):
    s = secs[d]; i = s.index("Fairness audit"); L = s[i:].split("\n")
    hdr = [x.strip() for x in L[2].strip("|").split("|")]
    for line in L[4:]:
        c = [x.strip() for x in line.strip("|").split("|")]
        if c[0] == model and c[1] == group: return float(c[hdr.index(col)])
def stress(d, model, sev, col):
    s = secs[d]; i = s.index("Stress sensitivity"); L = s[i:].split("\n")
    hdr = [x.strip() for x in L[2].strip("|").split("|")]
    for line in L[4:]:
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) > 2 and c[0] == model and float(c[1]) == sev: return float(c[hdr.index(col)])
A, PR, BR, EC, SL, LR = 0, 1, 2, 3, 4, 5
# abstract and results prose
chk("de stack-all auc", main_row("german", "Stack-All (raw)")[A], 0.795)
chk("tw stack-all auc", main_row("taiwan", "Stack-All (raw)")[A], 0.785)
chk("de catboost auc", main_row("german", "CatBoost")[A], 0.799)
chk("tw catboost auc", main_row("taiwan", "CatBoost")[A], 0.783)
chk("de auc diff vs catboost", paired("german", "auc", "CatBoost")[0], -0.003)
chk("tw auc diff vs catboost", paired("taiwan", "auc", "CatBoost")[0], 0.002)
chk("tw auc diff vs catboost exact", paired("taiwan", "auc", "CatBoost")[0], 0.0016, 5e-5)
chk("tw auc holm vs catboost", paired("taiwan", "auc", "CatBoost")[2], 0.110, 5e-4)
chk("de auc holm vs catboost", paired("german", "auc", "CatBoost")[2], 1.0, 5e-4)
chk("tw brier diff vs catboost", paired("taiwan", "brier", "CatBoost")[0], -0.0004, 5e-5)
chk("tw brier holm vs catboost", paired("taiwan", "brier", "CatBoost")[2], 0.032, 5e-4)
chk("tw brier diff vs lightgbm", paired("taiwan", "brier", "LightGBM")[0], -0.0006, 5e-5)
chk("tw brier holm vs lightgbm", paired("taiwan", "brier", "LightGBM")[2], 0.013, 5e-4)
chk("tw brier diff vs stackgbm", paired("taiwan", "brier", "Stack-GBM (raw)")[0], -0.0002, 5e-5)
chk("tw brier holm vs stackgbm", paired("taiwan", "brier", "Stack-GBM (raw)")[2], 0.024, 5e-4)
chk("de slope stack-all", main_row("german", "Stack-All (raw)")[SL], 1.05, 5e-3)
chk("de slope stack-gbm", main_row("german", "Stack-GBM (raw)")[SL], 1.10, 5e-3)
chk("de slope catboost", main_row("german", "CatBoost")[SL], 0.83, 5e-3)
for m in ("XGBoost", "LightGBM", "Logistic"):   # prose: "0.63 to 0.74"
    v = main_row("german", m)[SL]; n += 1
    if not (0.625 <= v <= 0.745): bad += 1; print("MISMATCH de slope range", m, v)
chk("de tabnet auc diff", paired("german", "auc", "TabNet")[0], 0.155, 6e-4)
chk("de tabnet auc holm", paired("german", "auc", "TabNet")[2], 0.012, 5e-4)
chk("de tabnet brier diff", paired("german", "brier", "TabNet")[0], -0.042, 5e-4)
chk("de tabnet brier holm", paired("german", "brier", "TabNet")[2], 0.001, 5e-4)
chk("tw logistic auc", main_row("taiwan", "Logistic")[A], 0.724)
chk("tw tabnet auc", main_row("taiwan", "TabNet")[A], 0.765)
chk("de tabnet auc", main_row("german", "TabNet")[A], 0.641)
chk("de ft auc", main_row("german", "FT-Transformer")[A], 0.740)
chk("tw ft auc", main_row("taiwan", "FT-Transformer")[A], 0.778)
chk("tw stackgbm auc", main_row("taiwan", "Stack-GBM (raw)")[A], 0.784)
chk("tw xgboost auc", main_row("taiwan", "XGBoost")[A], 0.783)
chk("de ece stackgbm", main_row("german", "Stack-GBM (raw)")[EC], 0.065)
chk("de ece stackall", main_row("german", "Stack-All (raw)")[EC], 0.067)
chk("de ece catboost", main_row("german", "CatBoost")[EC], 0.071)
chk("tw logistic ece", main_row("taiwan", "Logistic")[EC], 0.050)
chk("de iso prauc", main_row("german", "Stack-All (isotonic)")[PR], 0.600)
chk("de raw prauc", main_row("german", "Stack-All (raw)")[PR], 0.636)
chk("tw iso prauc", main_row("taiwan", "Stack-All (isotonic)")[PR], 0.550)
chk("tw raw prauc", main_row("taiwan", "Stack-All (raw)")[PR], 0.561)
for m in ("XGBoost", "LightGBM", "CatBoost", "TabNet", "FT-Transformer", "Stack-GBM (raw)", "Stack-All (raw)"):
    chk(f"tw loss reduction {m}", main_row("taiwan", m)[LR], 0.24, 0.0045)
# fairness prose
for m in ("Logistic", "CatBoost", "Stack-GBM (raw)", "Stack-All (raw)"):
    g = fair("german", m, "age_under_25", "dp_gap"); n += 1
    if not (-0.1650 <= g <= -0.1400): bad += 1; print("MISMATCH de age dp range", m, g)
    g = fair("taiwan", m, "age_under_25", "dp_gap"); n += 1
    if not (-0.0760 <= g <= -0.0540): bad += 1; print("MISMATCH tw age dp range", m, g)
    g = fair("taiwan", m, "female", "dp_gap"); n += 1
    if not (0.0520 <= g <= 0.0620): bad += 1; print("MISMATCH tw female dp range", m, g)
chk("de obs young", fair("german", "Stack-All (raw)", "age_under_25", "obs_unpriv"), 0.41, 5e-3)
chk("de obs old", fair("german", "Stack-All (raw)", "age_under_25", "obs_ref"), 0.28, 5e-3)
chk("tw obs young", fair("taiwan", "Stack-All (raw)", "age_under_25", "obs_unpriv"), 0.27, 5e-3)
chk("tw obs old", fair("taiwan", "Stack-All (raw)", "age_under_25", "obs_ref"), 0.22, 5e-3)
chk("tw obs female", fair("taiwan", "Stack-All (raw)", "female", "obs_unpriv"), 0.21, 5e-3)
chk("tw obs male", fair("taiwan", "Stack-All (raw)", "female", "obs_ref"), 0.24, 5e-3)
chk("de n young", fair("german", "Stack-All (raw)", "age_under_25", "n_unpriv"), 150, 0.5)
chk("de n foreign", fair("german", "Stack-All (raw)", "foreign_worker", "n_unpriv"), 963, 0.5)
chk("tw n young", fair("taiwan", "Stack-All (raw)", "age_under_25", "n_unpriv"), 2685, 0.5)
chk("tw n female", fair("taiwan", "Stack-All (raw)", "female", "n_unpriv"), 18112, 0.5)
# stress prose (percentage points at severity 0.3)
for m, v in (("Logistic", 0.3), ("XGBoost", 2.0), ("LightGBM", 2.3), ("CatBoost", 1.3), ("Stack-All (raw)", 1.6)):
    chk(f"tw stress uplift {m}", 100 * stress("taiwan", m, 0.3, "uplift"), v, 0.06)
for d in ("german", "taiwan"):
    for m in ("Logistic", "XGBoost", "LightGBM", "CatBoost", "TabNet", "FT-Transformer", "Stack-GBM (raw)", "Stack-All (raw)"):
        r = stress(d, m, 0.3, "spearman"); n += 1
        if r < 0.93: bad += 1; print("MISMATCH rank stability", d, m, r)
chk("de stack-all vs stack-gbm auc", paired("german", "auc", "Stack-GBM (raw)")[0], 0.0, 1.1e-3)
chk("tw stack-all vs stack-gbm auc", paired("taiwan", "auc", "Stack-GBM (raw)")[0], 0.0, 1.1e-3)
# tables must be exactly what build_tables.py produces from this results file
tmp = tempfile.mkdtemp(); shutil.copytree(os.path.join(here, "tables"), os.path.join(tmp, "old"))
subprocess.run([sys.executable, os.path.join(here, "build_tables.py"), res], cwd=here, check=True, capture_output=True)
for f in os.listdir(os.path.join(tmp, "old")):
    n += 1
    if not filecmp.cmp(os.path.join(tmp, "old", f), os.path.join(here, "tables", f), shallow=False):
        bad += 1; print("TABLE STALE", f)
print(f"{n} checks, {bad} mismatches"); sys.exit(1 if bad else 0)
