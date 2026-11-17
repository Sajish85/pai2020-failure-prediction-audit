#!/usr/bin/env python3
"""R1 revision step 2: why does gradient boosting score AUC 0.28 on the time-ordered split?

Everything here is on the S3 analysis base with the four request features, the same models and
seeds as scripts/grid.py (imported from it, so the baseline cell is the same code path).

Writes
  results/time_split_diagnostics.json
      baseline            the grid's time cell recomputed here + equality check against eval_grid.json
      split_check         paigpu.time_split vs an independent numpy sort of (start_time, job, task)
      pair_decomposition  exact AUC over within-top-user, within-others and cross-user test pairs
      train_fit           in-sample AUC on the training period (all / largest user / everyone else)
      without_top_user    largest user removed from both sides: same cut, and a fresh 75% cut
      cut_dates           the same split at other fractions, with and without the largest user
      shape_lookup        a model-free score: the training-period failure rate of a task's exact
                          request shape (plan_cpu, plan_mem, plan_gpu, inst_num); no sklearn involved
      shape_summary       how the test period divides by what the training period said about each shape
      top_user_by_plan_gpu / top_user_by_plan_cpu   the largest user's rate by request value, per period
  results/time_split_by_request_shape.csv
      one row per request shape with >= MIN_N tasks in either period, plus one pooled row for the rest:
      counts and failure rates per period, split into the largest user and everyone else, and the
      mean test-period score each model gives the shape.

No user identifier is written. Shapes are request values, never rows.

Usage: python scripts/time_split_diag.py [--frame PATH] [--out-dir DIR]
"""
import argparse
import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

R1 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
sys.path.insert(0, str(R1 / "scripts"))
import paigpu  # noqa: E402
import grid as G  # noqa: E402

NUM = paigpu.NUMERIC_FEATURES
MODELS = ["logreg", "gbm"]
FRAC = G.TIME_FRAC
CUTS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
MIN_N = 100
N_BINS = 20
SEED = 0


def auc(y, p):
    y = np.asarray(y)
    return round(float(roc_auc_score(y, p)), 6) if len(y) and len(np.unique(y)) == 2 else None


def pct(x):
    return None if x is None or x != x else round(100 * float(x), 4)


def rate(s):
    return pct(s.mean()) if len(s) else None


def fit_predict(tr, te, model):
    m = G.make_model(model, SEED, tr[NUM]).fit(tr[NUM], tr["label"].to_numpy())
    return m, m.predict_proba(te[NUM])[:, 1]


def three_way(y, p, is_top):
    return {"auc_all": auc(y, p), "auc_top_user_only": auc(y[is_top], p[is_top]),
            "auc_without_top_user": auc(y[~is_top], p[~is_top])}


def cross_auc(y, p, a, b):
    """AUC over pairs (failed task from group a, successful task from group b)."""
    pos, neg = p[a & (y == 1)], p[b & (y == 0)]
    if not len(pos) or not len(neg):
        return None, 0
    yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return round(float(roc_auc_score(yy, np.r_[pos, neg])), 6), int(len(pos)) * int(len(neg))


def describe(tr, te, top):
    te_top, tr_top = te["user"] == top, tr["user"] == top
    return {"n_train": len(tr), "n_test": len(te),
            "train_users": int(tr["user"].nunique()), "test_users": int(te["user"].nunique()),
            "train_failure_rate_pct": rate(tr["label"]), "test_failure_rate_pct": rate(te["label"]),
            "train_top_user_share_pct": pct(tr_top.mean()), "test_top_user_share_pct": pct(te_top.mean()),
            "train_top_user_failure_rate_pct": rate(tr.loc[tr_top, "label"]),
            "test_top_user_failure_rate_pct": rate(te.loc[te_top, "label"]),
            "train_others_failure_rate_pct": rate(tr.loc[~tr_top, "label"]),
            "test_others_failure_rate_pct": rate(te.loc[~te_top, "label"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, default=paigpu.DEFAULT_FRAME)
    ap.add_argument("--out-dir", type=pathlib.Path, default=R1 / "results")
    a = ap.parse_args(argv)
    df = paigpu.load_trace(a.frame)
    top = df["user"].value_counts().idxmax()
    df["is_top"] = df["user"] == top
    out = {"meta": {"script": "r1/scripts/time_split_diag.py", "frame": a.frame.name, "n_S3": len(df),
                    "features": NUM, "time_frac": FRAC, "model_seed": SEED, "cut_fractions": CUTS,
                    "shape": "exact value of (plan_cpu, plan_mem, plan_gpu, inst_num)",
                    "csv_min_n": MIN_N, "python": platform.python_version(), "sklearn": sklearn.__version__}}

    # ---- the split, and an independent re-derivation of it
    split = paigpu.time_split(FRAC, frame=df)["split"].to_numpy()
    order = np.lexsort((df["task_name"].to_numpy().astype(str), df["job_name"].to_numpy().astype(str),
                        df["start_time"].to_numpy()))
    indep = np.full(len(df), "test", dtype=object)
    indep[order[: int(np.floor(FRAC * len(df)))]] = "train"
    out["split_check"] = {"method": "numpy lexsort on (start_time, job_name, task_name), earliest floor(frac*n) = train",
                          "rows_assigned_differently_from_paigpu_time_split": int((indep != split).sum()),
                          "start_time_nulls": int(df["start_time"].isna().sum()),
                          "max_train_start_le_min_test_start": bool(
                              df.loc[split == "train", "start_time"].max() <= df.loc[split == "test", "start_time"].min())}
    tr, te = df[split == "train"], df[split == "test"]
    y, te_top = te["label"].to_numpy(), te["is_top"].to_numpy()
    ytr, tr_top = tr["label"].to_numpy(), tr["is_top"].to_numpy()

    # ---- baseline (must equal the grid's time cells), in-sample fit, exact pair decomposition
    grid_cells = {c["model"]: c for c in json.loads(
        (a.out_dir / "eval_grid.json").read_text().replace("NaN", "null"))["cells"] if c["split"] == "time"}
    preds, base, fit, pairs = {}, {"population": describe(tr, te, top)}, {}, {}
    for m in MODELS:
        mdl, p = fit_predict(tr, te, m)
        preds[m] = p
        base[m] = three_way(y, p, te_top)
        g = grid_cells[m]
        base[m]["equals_eval_grid_cell"] = bool(
            base[m]["auc_all"] == g["auc"] and base[m]["auc_top_user_only"] == g["auc_test_top_user_only"]
            and base[m]["auc_without_top_user"] == g["auc_test_without_top_user"])
        ptr = mdl.predict_proba(tr[NUM])[:, 1]
        fit[m] = {"train_" + k: v for k, v in three_way(ytr, ptr, tr_top).items()}
        tot = int(y.sum()) * int((1 - y).sum())
        d = {}
        for name, ga, gb in (("top_failed_vs_top_ok", te_top, te_top), ("other_failed_vs_other_ok", ~te_top, ~te_top),
                             ("top_failed_vs_other_ok", te_top, ~te_top), ("other_failed_vs_top_ok", ~te_top, te_top)):
            v, n = cross_auc(y, p, ga, gb)
            d[name] = {"auc": v, "pair_share_pct": pct(n / tot)}
        cross_n = (d["top_failed_vs_other_ok"]["pair_share_pct"] + d["other_failed_vs_top_ok"]["pair_share_pct"])
        d["cross_user_pairs"] = {
            "auc": round((d["top_failed_vs_other_ok"]["auc"] * d["top_failed_vs_other_ok"]["pair_share_pct"]
                          + d["other_failed_vs_top_ok"]["auc"] * d["other_failed_vs_top_ok"]["pair_share_pct"]) / cross_n, 6),
            "pair_share_pct": round(cross_n, 4)}
        pairs[m] = d
    out["baseline"], out["train_fit"], out["pair_decomposition"] = base, fit, pairs

    # ---- largest user removed from both sides
    wo = {}
    tr_o, te_o = tr[~tr["is_top"]], te[~te["is_top"]]
    wo["same_cut"] = {"note": "the 75% cut above; the largest user's rows dropped from train and from test",
                      "population": describe(tr_o, te_o, top)}
    df_o = df[~df["is_top"]].reset_index(drop=True)
    s_o = paigpu.time_split(FRAC, frame=df_o)["split"].to_numpy()
    tr_r, te_r = df_o[s_o == "train"], df_o[s_o == "test"]
    wo["recut"] = {"note": "the largest user dropped first, then a fresh earliest-75% / latest-25% cut",
                   "population": describe(tr_r, te_r, top)}
    for m in MODELS:
        wo["same_cut"][m] = {"auc": auc(te_o["label"].to_numpy(), fit_predict(tr_o, te_o, m)[1])}
        wo["recut"][m] = {"auc": auc(te_r["label"].to_numpy(), fit_predict(tr_r, te_r, m)[1])}
    out["without_top_user"] = wo

    # ---- other cut dates
    cuts = []
    for f in CUTS:
        s = paigpu.time_split(f, frame=df)["split"].to_numpy()
        a_tr, a_te = df[s == "train"], df[s == "test"]
        row = {"frac": f, **describe(a_tr, a_te, top)}
        b_tr, b_te = a_tr[~a_tr["is_top"]], a_te[~a_te["is_top"]]
        for m in MODELS:
            p = fit_predict(a_tr, a_te, m)[1]
            row[m] = three_way(a_te["label"].to_numpy(), p, a_te["is_top"].to_numpy())
            row[m]["auc_top_user_removed_from_both_sides"] = auc(b_te["label"].to_numpy(), fit_predict(b_tr, b_te, m)[1])
        cuts.append(row)
        print("cut", f, row["gbm"], row["logreg"], flush=True)
    out["cut_dates"] = cuts

    # ---- request shapes
    key = NUM
    g_tr = tr.groupby(key)["label"].agg(n_train="size", train_failed="sum")
    g_te = te.groupby(key)["label"].agg(n_test="size", test_failed="sum")
    sh = g_tr.join(g_te, how="outer").fillna(0)
    for part, d, tag in ((tr[tr["is_top"]], "train", "top"), (te[te["is_top"]], "test", "top"),
                         (tr[~tr["is_top"]], "train", "others"), (te[~te["is_top"]], "test", "others")):
        gg = part.groupby(key)["label"].agg(**{f"n_{d}_{tag}": "size", f"{d}_failed_{tag}": "sum"})
        sh = sh.join(gg, how="left")
    sh = sh.fillna(0).astype(int)
    te_k = te[key].copy()
    for m in MODELS:
        te_k[f"score_{m}"] = preds[m]
    sh = sh.join(te_k.groupby(key)[[f"score_{m}" for m in MODELS]].mean().rename(
        columns={f"score_{m}": f"mean_test_score_{m}" for m in MODELS}), how="left")

    # model-free lookup: training-period failure rate of the exact shape; unseen shapes get the training base rate
    train_rate = (g_tr["train_failed"] / g_tr["n_train"]).rename("r")
    base_rate = float(ytr.mean())
    look = te[key].merge(train_rate.reset_index(), on=key, how="left")["r"].to_numpy()
    seen = ~np.isnan(look)
    look = np.where(seen, look, base_rate)
    out["shape_lookup"] = {
        "definition": "score = training-period failure rate of the task's exact request shape; shapes not seen in training get the training base rate",
        "shapes_in_train": int(len(g_tr)), "shapes_in_test": int(len(g_te)),
        "test_tasks_with_shape_seen_in_train_pct": pct(seen.mean()),
        **three_way(y, look, te_top),
        "spearman_with_model_test_scores": {m: round(float(pd.Series(look).corr(pd.Series(preds[m]), method="spearman")), 4)
                                            for m in MODELS}}

    # test period divided by what training said about the shape
    tr_rate_te = look
    te_shape_rate = te[key].merge((g_te["test_failed"] / g_te["n_test"]).rename("t").reset_index(), on=key, how="left")["t"].to_numpy()
    summ = []
    for name, mask in (("shape below training base rate in training, above test base rate in test",
                        seen & (tr_rate_te < base_rate) & (te_shape_rate > y.mean())),
                       ("shape above training base rate in training, below test base rate in test",
                        seen & (tr_rate_te >= base_rate) & (te_shape_rate <= y.mean())),
                       ("same side of the base rate in both periods",
                        seen & ((tr_rate_te < base_rate) == (te_shape_rate <= y.mean()))),
                       ("shape not seen in training", ~seen)):
        summ.append({"group": name, "test_tasks": int(mask.sum()), "test_tasks_pct": pct(mask.mean()),
                     "test_failures_pct_of_all_test_failures": pct(y[mask].sum() / y.sum()),
                     "top_user_share_of_group_pct": pct(te_top[mask].mean()) if mask.any() else None,
                     "test_failure_rate_pct": pct(y[mask].mean()) if mask.any() else None})
    out["shape_summary"] = {"training_base_rate_pct": pct(base_rate), "test_base_rate_pct": pct(y.mean()), "groups": summ}

    def by_value(col):
        rows = []
        for v in sorted(set(tr.loc[tr_top, col]) | set(te.loc[te_top, col])):
            a_, b_ = tr.loc[tr_top & (tr[col] == v).to_numpy(), "label"], te.loc[te_top & (te[col] == v).to_numpy(), "label"]
            if len(a_) + len(b_) < MIN_N:
                continue
            rows.append({col: float(v), "n_train": len(a_), "train_failure_rate_pct": rate(a_),
                         "n_test": len(b_), "test_failure_rate_pct": rate(b_)})
        return rows
    out["top_user_by_plan_gpu"] = by_value("plan_gpu")
    out["top_user_by_plan_cpu"] = by_value("plan_cpu")

    # the request shape with the most test tasks, followed through the trace in 20 equal-count time bins
    big_shape = sh["n_test"].idxmax()
    in_shape = np.ones(len(df), bool)
    for c, v in zip(key, big_shape):
        in_shape &= (df[c] == v).to_numpy()
    tbin = np.empty(len(df), dtype=int)
    tbin[order] = np.arange(len(df)) * N_BINS // len(df)
    bins = []
    t_first = df["start_time"].min()
    top_np, cap8 = df["is_top"].to_numpy(), (df["cap_gpu"] == 8).to_numpy()
    for b in range(N_BINS):
        mk = in_shape & (tbin == b)
        lab = df.loc[mk, "label"]
        bins.append({"bin": b + 1, "tasks_from_pct": round(100 * b / N_BINS, 1), "tasks_to_pct": round(100 * (b + 1) / N_BINS, 1),
                     "period": "train" if (b + 1) / N_BINS <= FRAC else "test",
                     "days_from_first_start": [round(float((df.loc[tbin == b, "start_time"].min() - t_first) / 86400), 2),
                                               round(float((df.loc[tbin == b, "start_time"].max() - t_first) / 86400), 2)],
                     "shape_tasks": int(mk.sum()), "shape_failed": int(lab.sum()), "shape_share_of_bin_pct": pct(mk.sum() / (tbin == b).sum()),
                     "shape_failure_rate_pct": rate(lab),
                     "top_user_share_of_shape_pct": pct(df.loc[mk, "is_top"].mean()) if mk.any() else None})
    out["largest_test_shape_over_time"] = {
        "shape": dict(zip(key, (float(v) for v in big_shape))),
        "bins": f"{N_BINS} equal-count bins of all S3 tasks in time order; bin edges fall on the 75% cut",
        "tasks_all_periods": int(in_shape.sum()), "top_user_tasks_all_periods": int((in_shape & df["is_top"].to_numpy()).sum()),
        "failed_all_periods": int(df.loc[in_shape, "label"].sum()),
        "top_user_failed_all_periods": int(df.loc[in_shape & top_np, "label"].sum()),
        "top_user_quarter_gpu_tasks_all_shapes": int((df["is_top"] & (df["plan_gpu"] == 25)).sum()),
        "top_user_tasks_of_this_shape_on_8gpu_machines": int((in_shape & top_np & cap8).sum()),
        "top_user_failed_of_this_shape_on_8gpu_machines": int(df.loc[in_shape & top_np & cap8, "label"].sum()),
        "top_user_quarter_gpu_tasks_on_8gpu_machines": int((top_np & cap8 & (df["plan_gpu"] == 25).to_numpy()).sum()),
        "top_user_quarter_gpu_failed_on_8gpu_machines": int(df.loc[top_np & cap8 & (df["plan_gpu"] == 25).to_numpy(), "label"].sum()),
        "trace_span_days": round(float((df["start_time"].max() - t_first) / 86400), 2),
        "rows": bins}

    # per-shape CSV
    sh = sh.reset_index()
    big = (sh["n_train"] >= MIN_N) | (sh["n_test"] >= MIN_N)
    cnt = [c for c in sh.columns if c.startswith("n_") or "failed" in c]
    rest = sh.loc[~big, cnt].sum()
    keep = sh[big].copy()
    keep["shapes_pooled"] = 1
    pooled = {**{k: None for k in key}, **rest.to_dict(), "shapes_pooled": int((~big).sum())}
    for m in MODELS:  # pooled mean score, weighted by test count
        w = sh.loc[~big, "n_test"]
        pooled[f"mean_test_score_{m}"] = float((sh.loc[~big, f"mean_test_score_{m}"].fillna(0) * w).sum() / w.sum()) if w.sum() else None
    keep = pd.concat([keep, pd.DataFrame([pooled])], ignore_index=True)

    def r_(n, f):
        return (100 * keep[f] / keep[n].where(keep[n] > 0)).round(4)
    keep["train_failure_rate_pct"] = r_("n_train", "train_failed")
    keep["test_failure_rate_pct"] = r_("n_test", "test_failed")
    keep["test_share_pct"] = (100 * keep["n_test"] / len(te)).round(4)
    keep["test_top_user_share_of_shape_pct"] = (100 * keep["n_test_top"] / keep["n_test"].where(keep["n_test"] > 0)).round(4)
    keep["train_top_user_share_of_shape_pct"] = (100 * keep["n_train_top"] / keep["n_train"].where(keep["n_train"] > 0)).round(4)
    for tag in ("top", "others"):
        keep[f"train_failure_rate_{tag}_pct"] = r_(f"n_train_{tag}", f"train_failed_{tag}")
        keep[f"test_failure_rate_{tag}_pct"] = r_(f"n_test_{tag}", f"test_failed_{tag}")
    for m in MODELS:
        keep[f"mean_test_score_{m}"] = keep[f"mean_test_score_{m}"].round(4)
    keep = keep.sort_values(["shapes_pooled", "n_test", "n_train"], ascending=[True, False, False]).reset_index(drop=True)
    keep.insert(0, "rank_by_test_tasks", keep.index + 1)
    cols = (["rank_by_test_tasks"] + key + ["shapes_pooled", "n_train", "train_failure_rate_pct", "n_test", "test_failure_rate_pct",
            "test_share_pct", "train_top_user_share_of_shape_pct", "test_top_user_share_of_shape_pct",
            "n_train_top", "train_failure_rate_top_pct", "n_test_top", "test_failure_rate_top_pct",
            "n_train_others", "train_failure_rate_others_pct", "n_test_others", "test_failure_rate_others_pct"]
            + [f"mean_test_score_{m}" for m in MODELS])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    keep[cols].to_csv(a.out_dir / "time_split_by_request_shape.csv", index=False)
    out["shape_table"] = {"file": "time_split_by_request_shape.csv", "rows": int(len(keep)),
                          "shapes_listed": int(big.sum()), "shapes_pooled_in_last_row": int((~big).sum()),
                          "test_tasks_in_listed_shapes_pct": pct(sh.loc[big, "n_test"].sum() / len(te)),
                          "top": json.loads(keep[cols].head(8).to_json(orient="records"))}
    (a.out_dir / "time_split_diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("split_check", "baseline", "train_fit", "pair_decomposition",
                                           "without_top_user", "shape_lookup", "shape_summary", "largest_test_shape_over_time",
                                           "top_user_by_plan_gpu", "top_user_by_plan_cpu")}, indent=1))
    print(keep[cols].head(12).to_string())


if __name__ == "__main__":
    main()
