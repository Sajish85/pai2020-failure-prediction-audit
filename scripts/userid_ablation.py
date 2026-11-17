#!/usr/bin/env python3
"""R1 revision step 4: how much of the signal is simply which user submitted the task?

The grid of scripts/grid.py (same splits, seeds and models) run with three feature sets:

  request         plan_cpu, plan_mem, plan_gpu, inst_num   (the grid's own cells; must reproduce them)
  request+user    the four request columns plus user identity
  user            user identity alone

User identity is encoded with scikit-learn's TargetEncoder, fitted on the training part only:
each user becomes that user's (smoothed, cross-fitted) training failure rate. A user who is not in
the training part gets the training base rate, so under the user-grouped and leave-top-user-out
splits the user column is constant on the test part by construction; those cells are run as a
control. This is user identity on OUR frame. It is not the encoding or the pipeline of any published
study.

Writes results/userid_ablation.json (meta, summary, baseline check, cells) and
results/userid_ablation.csv (one row per cell). No user identifier is written.

Usage: python scripts/userid_ablation.py [--frame PATH] [--out-dir DIR]
"""
import argparse
import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

R1 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
sys.path.insert(0, str(R1 / "scripts"))
import paigpu  # noqa: E402
import grid as G  # noqa: E402

NUM = paigpu.NUMERIC_FEATURES
MODELS = ["logreg", "gbm"]
FEATURE_SETS = {"request": NUM, "request+user": NUM + ["user_code"], "user": ["user_code"]}


def auc(y, p):
    return round(float(roc_auc_score(y, p)), 6) if len(y) and len(np.unique(y)) == 2 else None


def make_model(model, fs, seed, Xtr):
    if fs == "request":
        return G.make_model(model, seed, Xtr)  # identical to the grid
    enc = TargetEncoder(target_type="binary", random_state=seed)
    num = [c for c in FEATURE_SETS[fs] if c != "user_code"]
    if model == "logreg":
        parts = [("user", make_pipeline(enc, StandardScaler()), ["user_code"])]
        if num:
            parts.insert(0, ("num", G.make_model("logreg", seed, Xtr).steps[0][1].transformers[0][1], num))
        return make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=2000, random_state=seed))
    parts = [("user", enc, ["user_code"])]
    if num:
        parts.insert(0, ("num", "passthrough", num))
    return make_pipeline(ColumnTransformer(parts), HistGradientBoostingClassifier(random_state=seed))


def run_cell(df, tr_idx, te_idx, model, fs, seed, meta):
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]  # row order as in grid.py (it matters to gbm's early-stopping hold-out)
    cols = FEATURE_SETS[fs]
    y, te_top = te["label"].to_numpy(), te["is_top"].to_numpy()
    seen = te["user_code"].isin(np.unique(tr["user_code"].to_numpy())).to_numpy()
    m = make_model(model, fs, seed, tr[cols]).fit(tr[cols], tr["label"].to_numpy())
    p = m.predict_proba(te[cols])[:, 1]
    return dict(meta, model=model, feature_set=fs, seed=seed, n_train=len(tr), n_test=len(te),
                test_users=int(te["user_code"].nunique()),
                test_tasks_from_users_seen_in_training_pct=round(100 * float(seen.mean()), 4),
                test_top_user_share_pct=round(100 * float(te_top.mean()), 4),
                auc=auc(y, p), auc_test_top_user_only=auc(y[te_top], p[te_top]),
                auc_test_without_top_user=auc(y[~te_top], p[~te_top]),
                auc_test_users_seen_in_training=auc(y[seen], p[seen]),
                auc_test_users_not_seen_in_training=auc(y[~seen], p[~seen]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, default=paigpu.DEFAULT_FRAME)
    ap.add_argument("--out-dir", type=pathlib.Path, default=R1 / "results")
    a = ap.parse_args(argv)
    df = paigpu.load_trace(a.frame)
    top = df["user"].value_counts().idxmax()
    df["is_top"] = df["user"] == top
    df["user_code"] = pd.factorize(df["user"], sort=True)[0].astype(str)  # opaque codes, in memory only
    n = len(df)
    splits = []
    for s in G.RANDOM_SEEDS:  # same construction as grid.py
        perm = np.random.default_rng(s).permutation(n)
        n_te = int(round(G.TEST_FRAC * n))
        splits.append(({"split": "random", "fold": None}, s, perm[n_te:], perm[:n_te]))
    for s in G.FOLD_SEEDS:
        folds = paigpu.user_folds(G.K, s, frame=df)["fold"].to_numpy()
        for f in range(G.K):
            splits.append(({"split": "user_folds", "fold": f}, s, np.flatnonzero(folds != f), np.flatnonzero(folds == f)))
    ts = paigpu.time_split(G.TIME_FRAC, frame=df)["split"].to_numpy()
    splits.append(({"split": "time", "fold": None}, 0, np.flatnonzero(ts == "train"), np.flatnonzero(ts == "test")))
    it = df["is_top"].to_numpy()
    splits.append(({"split": "leave_top_user_out", "fold": None}, 0, np.flatnonzero(~it), np.flatnonzero(it)))

    rows = []
    for meta, seed, trm, tem in splits:  # trm / tem are positional row indices
        for fs in FEATURE_SETS:
            for m in MODELS:
                rows.append(run_cell(df, trm, tem, m, fs, seed, meta))
                print(meta["split"], meta["fold"], seed, fs, m, rows[-1]["auc"], flush=True)
    grid = pd.DataFrame(rows)

    # the request cells must be the grid's cells
    ref = pd.read_csv(a.out_dir / "eval_grid.csv")
    mine = grid[grid["feature_set"] == "request"].merge(
        ref, on=["split", "fold", "seed", "model"], how="outer", suffixes=("", "_grid"), indicator=True)
    check = {"request_cells": int((grid["feature_set"] == "request").sum()), "eval_grid_cells": int(len(ref)),
             "cells_matched": int((mine["_merge"] == "both").sum()),
             "max_abs_auc_difference_from_eval_grid": float((mine["auc"] - mine["auc_grid"]).abs().max()),
             "all_equal": bool((mine["_merge"] == "both").all() and (mine["auc"] == mine["auc_grid"]).all())}

    def summ(col):
        return (grid.groupby(["split", "feature_set", "model"])[col]
                .agg(n_cells="count", mean="mean", sd="std", min="min", max="max").round(4).reset_index())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(a.out_dir / "userid_ablation.csv", index=False)
    meta = {"script": "r1/scripts/userid_ablation.py", "frame": a.frame.name, "n_S3": n, "n_users": int(df["user"].nunique()),
            "feature_sets": {k: [c.replace("user_code", "user (target-encoded)") for c in v] for k, v in FEATURE_SETS.items()},
            "user_encoding": "sklearn.preprocessing.TargetEncoder(target_type='binary', random_state=seed), defaults otherwise "
                             "(smooth='auto', cv=5 cross-fitting on the training part); users absent from training get the training base rate",
            "splits": "identical to scripts/grid.py", "random_seeds": G.RANDOM_SEEDS, "fold_seeds": G.FOLD_SEEDS, "k": G.K,
            "time_frac": G.TIME_FRAC, "python": platform.python_version(), "sklearn": sklearn.__version__}
    (a.out_dir / "userid_ablation.json").write_text(json.dumps(
        {"meta": meta, "baseline_check": check, "summary": json.loads(summ("auc").to_json(orient="records")),
         "summary_auc_test_without_top_user": json.loads(summ("auc_test_without_top_user").to_json(orient="records")),
         "cells": json.loads(grid.to_json(orient="records"))}, indent=2))
    print(check)
    print(summ("auc").to_string(index=False))
    print(summ("auc_test_without_top_user").to_string(index=False))
    print(grid[grid["split"].isin(["time", "leave_top_user_out"])].drop(columns=["fold", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
