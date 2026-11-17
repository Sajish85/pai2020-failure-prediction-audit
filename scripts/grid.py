#!/usr/bin/env python3
"""R1 step 5: evaluation grid, request features only.

Splits (S3 base, label Failed=1 / Terminated=0):
  random       75/25 shuffled split, seeds 0..4                  (5 cells per model)
  user_folds   paigpu.user_folds(k=5, seed) for seeds 0..2,
               every fold held out once                          (15 cells per model)
  time         paigpu.time_split(0.75): earliest 75% -> latest 25%; model seed 0
  leave_top_user_out  train on all other users, test on the largest user (plan rev 3)

Models (fixed seeds, fixed hyperparameters; no tuning, so no test leakage through tuning):
  logreg  sklearn LogisticRegression on log1p(numeric) standardised + one-hot categoricals
  gbm     sklearn HistGradientBoostingClassifier with native categorical handling

Features: paigpu.REQUEST_FEATURES only = plan_cpu, plan_mem, plan_gpu, inst_num (Paper 1's
"Request" family). No machine-side column (gpu_type, cap_gpu) and no categorical. If categoricals
are ever added to paigpu.CATEGORICAL_FEATURES, task_name keeps its TOP_TASK_NAMES most frequent
TRAINING values. Missing numerics: median of the training part (logreg); native NaN handling (gbm).

Every cell is written to results/eval_grid.csv and results/eval_grid.json with split, model,
seed, fold, n_train, n_test, test users, test failure rate, and AUC. A cell that can't be computed
(e.g. a single-class test set) gets status "not run" and a reason, never a number.

Usage: python scripts/grid.py [--frame PATH] [--out-dir DIR]
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

R1 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
import paigpu  # noqa: E402

RANDOM_SEEDS = [0, 1, 2, 3, 4]
FOLD_SEEDS = [0, 1, 2]
K = 5
TIME_FRAC = 0.75
TEST_FRAC = 0.25
TOP_TASK_NAMES = 30
NUM = paigpu.NUMERIC_FEATURES
CAT = paigpu.CATEGORICAL_FEATURES


def prep_categoricals(train, test):
    train, test = train.copy(), test.copy()
    keep = set(train["task_name"].value_counts().index[:TOP_TASK_NAMES]) if "task_name" in CAT else set()
    for d in (train, test):
        for c in CAT:
            if c == "task_name":
                d[c] = d[c].where(d[c].isin(keep), "other")
            d[c] = d[c].fillna("<NULL>").astype(str)
    return train, test


def make_model(name, seed, train):
    if name == "logreg":
        parts = [("num", make_pipeline(SimpleImputer(strategy="median"),
                                       FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                       StandardScaler()), NUM)]
        if CAT:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), CAT))
        pre = ColumnTransformer(parts)
        return make_pipeline(pre, LogisticRegression(max_iter=2000, random_state=seed))
    if name == "gbm":
        if not CAT:
            return HistGradientBoostingClassifier(random_state=seed)
        cats = {c: pd.CategoricalDtype(sorted(train[c].unique())) for c in CAT}
        to_cat = FunctionTransformer(
            lambda X: X.astype({c: cats[c] for c in CAT}), feature_names_out="one-to-one")
        clf = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed)
        return make_pipeline(to_cat, clf)
    raise ValueError(name)


def _auc_or_none(y, p):
    return round(float(roc_auc_score(y, p)), 6) if len(y) and len(np.unique(y)) == 2 else None


def run_cell(df, train_idx, test_idx, model, seed, meta, top_user=None):
    tr, te = df.loc[train_idx], df.loc[test_idx]
    te_top = (te["user"] == top_user).to_numpy()
    row = dict(meta, model=model, seed=seed, n_train=len(tr), n_test=len(te),
               train_users=int(tr["user"].nunique()), test_users=int(te["user"].nunique()),
               train_failure_rate_pct=round(100 * tr["label"].mean(), 4) if len(tr) else None,
               test_failure_rate_pct=round(100 * te["label"].mean(), 4) if len(te) else None,
               test_top_user_share_pct=round(100 * te_top.mean(), 4) if len(te) else None,
               train_top_user_failure_rate_pct=round(100 * tr.loc[tr["user"] == top_user, "label"].mean(), 4)
               if (tr["user"] == top_user).any() else None,
               test_top_user_failure_rate_pct=round(100 * te.loc[te_top, "label"].mean(), 4) if te_top.any() else None,
               auc_test_top_user_only=None, auc_test_without_top_user=None)
    if len(te) == 0 or te["label"].nunique() < 2:
        return dict(row, status="not run", reason="test set has one class (AUC undefined)", auc=None)
    if tr["label"].nunique() < 2:
        return dict(row, status="not run", reason="train set has one class", auc=None)
    Xtr, Xte = prep_categoricals(tr[NUM + CAT], te[NUM + CAT])
    t0 = time.time()
    try:
        m = make_model(model, seed, Xtr).fit(Xtr, tr["label"].to_numpy())
        p = m.predict_proba(Xte)[:, 1]
        y = te["label"].to_numpy()
        auc = float(roc_auc_score(y, p))
        row["auc_test_top_user_only"] = _auc_or_none(y[te_top], p[te_top])
        row["auc_test_without_top_user"] = _auc_or_none(y[~te_top], p[~te_top])
    except Exception as e:  # recorded, never silently replaced
        return dict(row, status="not run", reason=f"{type(e).__name__}: {e}", auc=None)
    print(f"  fit+predict {time.time() - t0:.1f}s", flush=True)  # log only; kept out of results
    return dict(row, status="ok", reason="", auc=round(auc, 6))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, default=paigpu.DEFAULT_FRAME)
    ap.add_argument("--out-dir", type=pathlib.Path, default=R1 / "results")
    a = ap.parse_args(argv)
    t0 = time.time()
    df = paigpu.load_trace(a.frame)
    rows = []
    models = ["logreg", "gbm"]
    top = df["user"].value_counts().idxmax()  # largest user by S3 tasks (identifier never written)

    for s in RANDOM_SEEDS:
        perm = np.random.default_rng(s).permutation(len(df))
        n_te = int(round(TEST_FRAC * len(df)))
        te_idx, tr_idx = df.index[perm[:n_te]], df.index[perm[n_te:]]
        for m in models:
            rows.append(run_cell(df, tr_idx, te_idx, m, s, {"split": "random", "fold": None}, top_user=top))
            print(rows[-1]["split"], m, s, rows[-1]["auc"], flush=True)

    for s in FOLD_SEEDS:
        folds = paigpu.user_folds(K, s, frame=df)["fold"].to_numpy()
        for f in range(K):
            te_idx, tr_idx = df.index[folds == f], df.index[folds != f]
            for m in models:
                rows.append(run_cell(df, tr_idx, te_idx, m, s, {"split": "user_folds", "fold": f}, top_user=top))
                print("user_folds", m, s, f, rows[-1]["auc"], flush=True)

    try:
        ts = paigpu.time_split(TIME_FRAC, frame=df)["split"].to_numpy()
    except ValueError as e:
        ts = None
        for m in models:
            rows.append({"split": "time", "fold": None, "model": m, "seed": 0, "n_train": None,
                         "n_test": None, "status": "not run", "reason": f"time_split: {e}", "auc": None})
    for m in (models if ts is not None else []):
        rows.append(run_cell(df, df.index[ts == "train"], df.index[ts == "test"], m, 0,
                             {"split": "time", "fold": None}, top_user=top))
        print("time", m, rows[-1]["auc"], flush=True)

    is_top = (df["user"] == top).to_numpy()
    for m in models:
        rows.append(run_cell(df, df.index[~is_top], df.index[is_top], m, 0,
                             {"split": "leave_top_user_out", "fold": None}, top_user=top))
        print("leave_top_user_out", m, rows[-1]["auc"], flush=True)

    grid = pd.DataFrame(rows)
    ok = grid[grid["status"] == "ok"]
    summary = (ok.groupby(["split", "model"])["auc"]
               .agg(n_cells="count", mean="mean", sd="std", min="min", max="max")
               .round(4).reset_index())
    summary_wo = (ok.groupby(["split", "model"])["auc_test_without_top_user"]
                  .agg(n_cells="count", mean="mean", sd="std", min="min", max="max")
                  .round(4).reset_index())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(a.out_dir / "eval_grid.csv", index=False)
    meta = {"script": "r1/scripts/grid.py", "frame": a.frame.name, "n_S3": len(df),
            "features": paigpu.REQUEST_FEATURES, "top_task_names": TOP_TASK_NAMES,
            "random_seeds": RANDOM_SEEDS, "fold_seeds": FOLD_SEEDS, "k": K,
            "time_frac": TIME_FRAC, "test_frac_random": TEST_FRAC,
            "top_user_columns": "auc_test_top_user_only / auc_test_without_top_user: the same cell's predictions scored on the largest S3 user's test rows vs everyone else's",
            "sd_note": "sample SD (ddof=1) across cells; for user_folds the cells are 5 folds x 3 seeds",
            "python": platform.python_version(), "sklearn": sklearn.__version__}
    (a.out_dir / "eval_grid.json").write_text(json.dumps(
        {"meta": meta, "summary": summary.to_dict(orient="records"),
         "summary_auc_test_without_top_user": summary_wo.to_dict(orient="records"),
         "cells": grid.to_dict(orient="records")}, indent=2, default=str))
    print(summary.to_string(index=False))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
