# Calibration, Not Accuracy: What Stacking Adds to Gradient Boosting in Credit Risk Scoring

Code, results and manuscript source for the paper of the same name (R. Ghadiyaram, V. Peddapaidi, S. K. Kaata).

## What this repository shows
A calibrated stack of six learners (logistic regression, XGBoost, LightGBM, CatBoost, TabNet, FT-Transformer) is compared with each of them on two public credit datasets. The stack ties CatBoost on ranking, and its measurable advantage is calibration stability. Fairness gaps are nearly the same for every model. Details and numbers are in `results/results.md` and in the paper.

## Layout
- `src/credit_benchmark.py`: data loaders, the models, nested repeated cross-validation with out-of-fold stacking, calibration, fairness and stress outputs.
- `src/analyze_benchmark.py`: tables, Nadeau-Bengio paired tests with Holm adjustment, `results.md`.
- `src/make_figures.py`: print-size figures.
- `src/verify_paper.py`: re-derives headline numbers from the fold-level outputs.
- `notebooks/credit_risk_benchmark.ipynb`: one-shot Google Colab notebook (GPU recommended for Taiwan).
- `data/`: the two public datasets (see `data/README.md`).
- `results/`: `results.md` from the reported Colab run and the paper figures.
- `paper/`: LaTeX source, `build_tables.py` (tables are generated from `results/results.md`), and `check_manuscript.py`, which checks every table and the numbers quoted in the text against `results/results.md`.

## Reproduce locally
```bash
pip install -r requirements.txt
export CR_BASE=$PWD
python src/credit_benchmark.py --data german --repeats 3
python src/credit_benchmark.py --data taiwan --repeats 2 --n_max 30000   # GPU recommended
python src/analyze_benchmark.py
python src/make_figures.py
python src/verify_paper.py
```
Add `--no_deep` for a CPU-only run without TabNet and the FT-Transformer. Seeds are fixed. The deep models were trained on a GPU in the reported run, so small numerical differences on CPU are expected.

## Check the paper against the results
```bash
cd paper
python check_manuscript.py     # expects 0 mismatches
pdflatex paper.tex && pdflatex paper.tex
```

## Limitations
Both datasets are old (German 1973 to 1975, Taiwan 2005). The stress analysis is a stylized input shift, not real tariff or macroeconomic data. The German foreign-worker reference group has only 37 applicants. Hyperparameters are fixed at standard values.

## Cite
See `CITATION.cff`. Code is MIT licensed. The data keep their original CC BY 4.0 licenses.
