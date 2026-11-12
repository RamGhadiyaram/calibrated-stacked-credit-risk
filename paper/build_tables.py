"""Parse results.md (Colab run) and emit LaTeX tables, so no number in the paper is typed by hand."""
import re, sys
src = sys.argv[1] if len(sys.argv) > 1 else "../results/results.md"
txt = open(src).read()
secs = re.split(r"^## ", txt, flags=re.M)[1:]
D = {}
for s in secs:
    title = s.split("\n")[0]
    d = "german" if title.startswith("South") else "taiwan"
    blocks = s.split("\n\n")
    D[d] = {}
    # main table
    main = {}
    for line in s.split("\n"):
        if line.startswith("|") and "±" in line:
            c = [x.strip() for x in line.strip("|").split("|")]
            main[c[0]] = [x.split(" ± ")[0] for x in c[1:]]
    D[d]["main"] = main
    paired = {}
    for m in ("auc", "brier"):
        i = s.index(f"Paired differences in {m}")
        rows = []
        for line in s[i:].split("\n")[4:]:
            if not line.startswith("|"): break
            c = [x.strip() for x in line.strip("|").split("|")]
            rows.append(c)
        paired[m] = rows
    D[d]["paired"] = paired
    def tab(marker):
        i = s.index(marker); L = s[i:].split("\n")[2:]
        hdr = [x.strip() for x in s[i:].split("\n")[2].strip("|").split("|")]
        out = []
        for line in s[i:].split("\n")[4:]:
            if not line.startswith("|"): break
            out.append(dict(zip(hdr, [x.strip() for x in line.strip("|").split("|")])))
        return out
    D[d]["fair"] = tab("Fairness audit"); D[d]["stress"] = tab("Stress sensitivity")

NAMES = {"Stack-GBM (raw)": "Stack-GBM", "Stack-All (raw)": "Stack-All"}
nm = lambda m: NAMES.get(m, m)
ORDER = ["Logistic", "XGBoost", "LightGBM", "CatBoost", "TabNet", "FT-Transformer", "Stack-GBM (raw)", "Stack-All (raw)"]
fmtp = lambda p: "$<$0.001" if float(p) < 0.001 else p
sg = lambda x: x.replace("-", "$-$")

# Table: main results (cols: model, auc, pr, brier, ece, slope, loss)
t = [r"\begin{table*}[t]", r"\centering", r"\caption{Out-of-fold performance (mean over outer folds). German: 15 folds (3 repeats of 5-fold); Taiwan: 10 folds (2 repeats of 5-fold, 30,000 rows). Best value per column in bold; ties within rounding are all bold.}",
     r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
     r"\begin{tabular}{l|cccc|cccc}", r"\toprule",
     r" & \multicolumn{4}{c|}{South German Credit} & \multicolumn{4}{c}{Taiwan credit-card default}\\",
     r"Model & AUC & Brier & ECE & Slope & AUC & Brier & ECE & Slope\\", r"\midrule"]
idx = {"auc": 0, "brier": 2, "ece": 3, "slope": 4}
best = {}
for d in D:
    for k, j in idx.items():
        vals = {m: float(D[d]["main"][m][j]) for m in ORDER}
        if k == "auc": b = max(vals.values()); w = {m for m, v in vals.items() if abs(v - b) < 0.0005 + 1e-9}
        elif k in ("brier", "ece"): b = min(vals.values()); w = {m for m, v in vals.items() if abs(v - b) < 0.0005 + 1e-9}
        else: b = min(vals.values(), key=lambda v: abs(v - 1)); w = {m for m, v in vals.items() if abs(abs(v - 1) - abs(b - 1)) < 0.0005 + 1e-9}
        best[(d, k)] = w
for m in ORDER:
    cells = []
    for d in ("german", "taiwan"):
        for k, j in idx.items():
            v = D[d]["main"][m][j]
            cells.append(r"\textbf{%s}" % v if m in best[(d, k)] else v)
    t.append(nm(m) + " & " + " & ".join(cells) + r"\\")
    if m == "FT-Transformer": t.append(r"\midrule")
t += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open("tables/main.tex", "w").write("\n".join(t))

# Table: paired tests
t = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Paired differences, Stack-All minus each comparator, over outer folds. Mean difference with 95\% interval from the Nadeau-Bengio corrected resampled $t$-test; $p_H$ is the Holm-adjusted $p$-value over the seven comparators. Negative Brier differences favor Stack-All.}",
     r"\label{tab:paired}", r"\small", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{l|rcr|rcr}", r"\toprule",
     r"Comparator & \multicolumn{3}{c|}{AUC difference} & \multicolumn{3}{c}{Brier difference}\\",
     r" & Diff. & 95\% interval & $p_H$ & Diff. & 95\% interval & $p_H$\\", r"\midrule"]
for d, lab in (("german", "South German Credit (15 folds)"), ("taiwan", "Taiwan credit-card default (10 folds)")):
    t.append(r"\multicolumn{7}{l}{\textit{%s}}\\" % lab)
    for a, b in zip(D[d]["paired"]["auc"], D[d]["paired"]["brier"]):
        t.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (
            nm(a[0]), sg(a[1]), sg(a[2]).replace(",", ", "), fmtp(a[4]),
            sg(b[1]), sg(b[2]).replace(",", ", "), fmtp(b[4])))
    if d == "german": t.append(r"\midrule")
t += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open("tables/paired.tex", "w").write("\n".join(t))

# Table: fairness
F = ["Logistic", "CatBoost", "Stack-All (raw)"]
GL = {"age_under_25": "Age under 25", "foreign_worker": "Foreign worker", "female": "Female"}
t = [r"\begin{table}[t]", r"\centering",
     r"\caption{Fairness audit at 80\% approval. Gaps are the unprivileged group minus the reference group (mean over folds). Approval gap: share approved. Good-payer gap: share of non-defaulters approved. Observed default rate of each group shown for context. DE is South German Credit and TW is Taiwan; the number in brackets is the size of the unprivileged group. The German foreign-worker reference group has only 37 applicants, so that gap is noisy.}",
     r"\label{tab:fair}", r"\small", r"\setlength{\tabcolsep}{1.5pt}",
     r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
     r"Group & Model & Approval & Good-payer & \multicolumn{2}{c}{Default rate}\\",
     r" & & gap & gap & Unpriv. & Ref.\\", r"\midrule"]
for d, lab in (("german", "South German Credit"), ("taiwan", "Taiwan")):
    for g in (["age_under_25", "foreign_worker"] if d == "german" else ["age_under_25", "female"]):
        first = True
        for m in F:
            r = [x for x in D[d]["fair"] if x["model"] == m and x["group"] == g][0]
            gl = (("DE " if d == "german" else "TW ") + {"age_under_25":"age $<$25","foreign_worker":"foreign","female":"female"}[g] + " (%d)" % int(r["n_unpriv"])) if first else ""
            t.append("%s & %s & %s & %s & %.1f\\%% & %.1f\\%%\\\\" % (
                gl, nm(m), sg("%+.3f" % float(r["dp_gap"])), sg("%+.3f" % float(r["eo_gap"])),
                100 * float(r["obs_unpriv"]), 100 * float(r["obs_ref"])))
            first = False
        t.append(r"\midrule")
t[-1] = r"\bottomrule"
t += [r"\end{tabular}", r"\end{table}"]
open("tables/fair.tex", "w").write("\n".join(t))

# Table: stress (severity 0.3)
t = [r"\begin{table}[t]", r"\centering",
     r"\caption{Stylized stress sensitivity at severity 0.3: mean increase in predicted default probability (percentage points) and Spearman rank correlation with the unstressed ranking. Labels are unchanged; this is not a tariff forecast.}",
     r"\label{tab:stress}", r"\small", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{l|cc|cc}", r"\toprule",
     r" & \multicolumn{2}{c|}{German} & \multicolumn{2}{c}{Taiwan}\\", r"Model & Uplift & Rank $\rho$ & Uplift & Rank $\rho$\\", r"\midrule"]
for m in ORDER:
    cells = []
    for d in ("german", "taiwan"):
        r = [x for x in D[d]["stress"] if x["model"] == m and float(x["severity"]) == 0.3][0]
        cells += ["%.2f" % (100 * float(r["uplift"])), "%.3f" % float(r["spearman"])]
    t.append(nm(m) + " & " + " & ".join(cells) + r"\\")
t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("tables/stress.tex", "w").write("\n".join(t))
print("ok")
