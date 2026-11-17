#!/usr/bin/env python3
"""Counts quoted in the report that no other result file holds. S3 analysis base, counts only.

Writes results/supplementary_counts.json
  concentration_top_1pct   the largest 1% of users BY TASK COUNT: their share of tasks and of failures;
                           and, for contrast, the share of failures held by the 1% of users with the
                           MOST FAILURES (a different set; that is the by_failed curve in
                           paper1_recheck.json and the failure curve of Figure 2)
  burst                    the largest test-period request shape inside time bins 16-18 of 20
                           (the bins of time_split_diagnostics.json), split into the largest user and
                           everyone else, with shares of the test tasks and test failures
  test_period_span         how much of the trace's time span the latest 25% of tasks cover

No user identifier is written.

Usage: python scripts/supplementary_counts.py [--frame PATH] [--out-dir DIR]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

R1 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
import paigpu  # noqa: E402

FRAC = 0.75
N_BINS = 20
BURST_BINS = (16, 17, 18)
SHAPE = {"plan_cpu": 100.0, "plan_mem": 29.296875, "plan_gpu": 25.0, "inst_num": 1.0}


def pct(a, b):
    return round(100 * a / b, 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, default=paigpu.DEFAULT_FRAME)
    ap.add_argument("--out-dir", type=pathlib.Path, default=R1 / "results")
    a = ap.parse_args(argv)
    df = paigpu.load_trace(a.frame)
    n, n_failed = len(df), int(df["label"].sum())
    out = {"meta": {"script": "scripts/supplementary_counts.py", "frame": a.frame.name, "n_S3": n,
                    "time_frac": FRAC, "n_bins": N_BINS, "burst_bins": list(BURST_BINS), "shape": SHAPE}}

    # ---- concentration: one ordering of users (by task count), both shares read off it
    pu = df.groupby("user").agg(tasks=("label", "size"), failed=("label", "sum"))
    k = math.ceil(0.01 * len(pu))
    by_tasks = pu.sort_values(["tasks", "failed"], ascending=False).head(k)
    by_failed = pu.sort_values(["failed", "tasks"], ascending=False).head(k)
    out["concentration_top_1pct"] = {
        "users": len(pu), "users_in_top_1pct": k,
        "largest_by_task_count": {"share_of_tasks_pct": pct(by_tasks["tasks"].sum(), n),
                                  "share_of_failures_pct": pct(by_tasks["failed"].sum(), n_failed)},
        "most_failures": {"share_of_failures_pct": pct(by_failed["failed"].sum(), n_failed),
                          "users_also_in_largest_by_task_count": int(by_failed.index.isin(by_tasks.index).sum())},
    }

    # ---- time order, the 75% cut and 20 equal-count bins (same sort as paigpu.time_split)
    order = np.lexsort((df["task_name"].to_numpy().astype(str), df["job_name"].to_numpy().astype(str),
                        df["start_time"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    cut = int(np.floor(FRAC * n))
    df["bin"] = np.minimum(np.arange(n) * N_BINS // n, N_BINS - 1) + 1
    te = df.iloc[cut:]
    top = pu["tasks"].idxmax()
    is_shape = np.ones(n, dtype=bool)
    for col, v in SHAPE.items():
        is_shape &= (df[col] == v).to_numpy()
    b = df[is_shape & df["bin"].isin(BURST_BINS).to_numpy()]
    bt, bo = b[b["user"] == top], b[b["user"] != top]
    n_te, f_te = len(te), int(te["label"].sum())
    out["burst"] = {
        "test_tasks": n_te, "test_failed": f_te,
        "tasks": len(b), "failed": int(b["label"].sum()),
        "share_of_test_tasks_pct": pct(len(b), n_te),
        "share_of_test_failures_pct": pct(int(b["label"].sum()), f_te),
        "largest_user": {"tasks": len(bt), "failed": int(bt["label"].sum()),
                         "share_of_burst_tasks_pct": pct(len(bt), len(b))},
        "other_users": {"users": int(bo["user"].nunique()), "tasks": len(bo), "failed": int(bo["label"].sum()),
                        "failure_rate_pct": pct(int(bo["label"].sum()), len(bo)) if len(bo) else None,
                        "bins_with_tasks": sorted(int(x) for x in bo["bin"].unique()),
                        "largest_single_other_user": {
                            "tasks": int(bo["user"].value_counts().iloc[0]) if len(bo) else 0,
                            "failed": int(bo.groupby("user")["label"].sum()[bo["user"].value_counts().index[0]]) if len(bo) else 0}},
    }

    # ---- how much of the time span the test period covers
    t0, t1, tc = float(df["start_time"].iloc[0]), float(df["start_time"].iloc[-1]), float(te["start_time"].iloc[0])
    out["test_period_span"] = {
        "trace_span_days": round((t1 - t0) / 86400, 2),
        "test_starts_at_day": round((tc - t0) / 86400, 2),
        "test_share_of_tasks_pct": pct(n_te, n),
        "test_share_of_time_span_pct": pct(t1 - tc, t1 - t0),
    }
    path = a.out_dir / "supplementary_counts.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
