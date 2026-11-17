#!/usr/bin/env python3
"""R1 step 3: Paper 1 re-checks on the raw trace (aggregates only).

Definitions are taken from Paper 1's public repo (github.com/Sajish85/machine-class-is-the-message,
cloned to ../paper1-ref), which wins over the earlier generic version of this script:
  machine class   = cap_gpu, GPUs installed on the task's assigned machine (pai_machine_spec.cap_gpu
                    of the first instance's machine). Paper 1: data/data_dictionary.md; stratified.py
                    groups every within-class analysis by cap_gpu.
  request shape   = exact plan_gpu value (100 = one GPU). Paper 1 chart 2 uses 10, 25, 50, 100;
                    stratified.py reports every plan_gpu value with n >= 100 within each class.
  quarter-GPU cell = plan_gpu == 25 on cap_gpu == 8 machines, contrasted with plan_gpu == 25 on
                    cap_gpu == 2 machines (Paper 1 chart 2 title).

Two bases, each with and without the top user (the user with the most S3 tasks):
  S3      resolved GPU tasks (funnel.json S3), the whole raw-trace population
  kanvas  the KANVAS-comparable subset: KANVAS's own filter (KANVAS repo kanvas/data.py) applied to
          every labeled task, without KANVAS's stratified random sample (paigpu/convert.py)

Secondary (kept from the first version, not Paper 1's definition): failure by task gpu_type and by
plan_gpu bucket.

If --paper1-ref is given, Paper 1's committed result values for the same cells are copied into
"paper1_published_reference" with the source file and commit, for side-by-side comparison. They are
Paper 1's numbers, not re-derived here.

Usage: python scripts/recheck.py [--frame PATH] [--out PATH] [--paper1-ref DIR]
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

R1 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
import paigpu  # noqa: E402

MC = paigpu.MACHINE_CLASS
PAPER1_PLAN_GPU = [10, 25, 50, 100]   # Paper 1 chart 2 x-axis
MIN_N_LIST = 100                       # Paper 1 stratified.py: plan_gpu values with n >= 100
PLAN_GPU_BUCKETS = [
    ("(0,25)", lambda g: (g > 0) & (g < 25)),
    ("=25 (quarter GPU)", lambda g: g == 25),
    ("(25,50)", lambda g: (g > 25) & (g < 50)),
    ("=50", lambda g: g == 50),
    ("(50,100)", lambda g: (g > 50) & (g < 100)),
    ("=100 (one GPU)", lambda g: g == 100),
    ("(100,200)", lambda g: (g > 100) & (g < 200)),
    (">=200", lambda g: g >= 200),
]


def _key(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "<NULL>"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def cell_table(df, by, total_failed):
    by = by if isinstance(by, list) else [by]
    g = df.groupby(by, dropna=False, observed=True)
    t = pd.DataFrame({"tasks": g.size(), "failed": g["label"].sum(), "users": g["user"].nunique()})
    t["failure_rate_pct"] = (100 * t["failed"] / t["tasks"]).round(4)
    t["share_of_all_failures_pct"] = (100 * t["failed"] / total_failed).round(4)
    t["share_of_tasks_pct"] = (100 * t["tasks"] / len(df)).round(4)
    t = t.reset_index()
    for c in by:
        t[c] = [_key(v) for v in t[c].astype(object)]
    t = t.sort_values(["tasks"] + by, ascending=[False] + [True] * len(by))
    return [{k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in r.items()}
            for r in t.to_dict(orient="records")]


def cell(df, mask):
    s = df[mask]
    n, f = len(s), int(s["label"].sum())
    return {"tasks": n, "failed": f, "users": int(s["user"].nunique()),
            "failure_rate_pct": round(100 * f / n, 4) if n else None}


def bucket_plan_gpu(df):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, f in PLAN_GPU_BUCKETS:
        out[f(df["plan_gpu"])] = name
    return out


def concentration(df, points=101):
    per_user = df.groupby("user").agg(tasks=("label", "size"), failed=("label", "sum"))
    out = {}
    for key in ["tasks", "failed"]:
        s = np.sort(per_user[key].to_numpy())[::-1]
        cum = np.cumsum(s) / s.sum()
        xs = np.linspace(0, 1, points)
        idx = np.clip(np.ceil(xs * len(s)).astype(int) - 1, -1, len(s) - 1)
        ys = np.where(idx < 0, 0.0, cum[np.maximum(idx, 0)])
        out[f"by_{key}"] = {"user_fraction": xs.round(4).tolist(),
                            "cumulative_share": ys.round(6).tolist()}
    return out


def recheck(df):
    df = df.assign(label=df["label"].astype("int64"))
    total_failed = int(df["label"].sum())
    df = df.assign(plan_gpu_bucket=bucket_plan_gpu(df))
    g, c = df["plan_gpu"], df[MC]
    q8, q2 = cell(df, (g == 25) & (c == 8)), cell(df, (g == 25) & (c == 2))
    ratio = (round(q8["failure_rate_pct"] / q2["failure_rate_pct"], 4)
             if q8["failure_rate_pct"] is not None and q2["failure_rate_pct"] else None)
    rest8 = cell(df, (g != 25) & (c == 8))
    within = {}
    for cap in sorted(c.dropna().unique()):
        sub = df[c == cap]
        vc = sub["plan_gpu"].value_counts()
        within[_key(cap)] = [dict(plan_gpu=_key(v), **cell(sub, sub["plan_gpu"] == v))
                             for v in sorted(vc[vc >= MIN_N_LIST].index)]
    return {
        "n_tasks": len(df), "n_failed": total_failed, "n_users": int(df["user"].nunique()),
        "failure_rate_pct": round(100 * total_failed / len(df), 4) if len(df) else None,
        "machine_class_cap_gpu": cell_table(df, MC, total_failed),
        "quarter_gpu_cell": {
            "definition": "plan_gpu == 25 and cap_gpu == 8 (Paper 1); contrast plan_gpu == 25 and cap_gpu == 2",
            "plan25_cap8": q8, "plan25_cap2": q2,
            "rate_ratio_cap8_over_cap2": ratio,
            "cap8_all_other_plan_gpu": rest8,
            "plan25_cap8_share_of_all_failures_pct":
                round(100 * q8["failed"] / total_failed, 4) if total_failed else None,
        },
        "plan_gpu_x_cap_gpu_paper1_values": [
            dict(plan_gpu=v, cap_gpu=cap, **cell(df, (g == v) & (c == cap)))
            for cap in (2, 8) for v in PAPER1_PLAN_GPU],
        "plan_gpu_within_cap_gpu_n_ge_100": within,
        "secondary_task_gpu_type": cell_table(df, "gpu_type", total_failed),
        "secondary_gpu_type_x_cap_gpu": cell_table(df, ["gpu_type", MC], total_failed),
        "secondary_plan_gpu_bucket": cell_table(df, "plan_gpu_bucket", total_failed),
    }


def paper1_reference(ref):
    ref = pathlib.Path(ref)
    try:
        commit = subprocess.run(["git", "-C", str(ref), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        commit = f"unknown ({type(e).__name__})"
    eda = json.loads((ref / "scripts" / "eda_results.json").read_text())
    st = json.loads((ref / "scripts" / "stratified_results.json").read_text())

    def pct(d):
        return {k: {"failure_rate_pct": round(100 * v["rate"], 4), "tasks": v["n"]} for k, v in d.items()}
    return {
        "note": "Paper 1's own committed outputs (KANVAS 18,890-task sample), copied for comparison only",
        "repo": "github.com/Sajish85/machine-class-is-the-message", "commit": commit,
        "n_tasks": eda["shape"][0], "failure_rate_pct": round(100 * eda["failure_rate"], 4),
        "rate_by_cap_gpu": pct(eda["rate_by_cap_gpu"]),
        "rate_by_plan_gpu_cap2": pct(st["rate_by_plan_gpu_cap2"]),
        "rate_by_plan_gpu_cap8": pct(st["rate_by_plan_gpu_cap8"]),
        "source_files": ["scripts/eda_results.json", "scripts/stratified_results.json"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, default=paigpu.DEFAULT_FRAME)
    ap.add_argument("--out", type=pathlib.Path, default=R1 / "results" / "paper1_recheck.json")
    ap.add_argument("--paper1-ref", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    s3 = paigpu.load_trace(a.frame, base="S3")
    kv = paigpu.load_trace(a.frame, base="kanvas")
    top_user = s3["user"].value_counts().idxmax()
    res = {
        "script": "r1/scripts/recheck.py",
        "frame": a.frame.name,
        "paper1_definitions": "machine class = cap_gpu; quarter-GPU cell = plan_gpu 25 on cap_gpu 8 "
                              "(from paper1-ref; see script docstring)",
        "units": "plan_gpu is a percentage of one GPU (trace README l.182)",
        "top_user": "the user with the most S3 tasks (identifier not written)",
        "top_user_task_share_pct_S3": round(100 * (s3["user"] == top_user).mean(), 4),
        "top_user_failure_share_pct_S3": round(
            100 * s3.loc[s3["user"] == top_user, "label"].astype("int64").sum() / s3["label"].astype("int64").sum(), 4),
        "top_user_task_share_pct_kanvas": round(100 * (kv["user"] == top_user).mean(), 4) if len(kv) else None,
        "top_user_failure_share_pct_kanvas": round(
            100 * kv.loc[kv["user"] == top_user, "label"].astype("int64").sum() / kv["label"].astype("int64").sum(), 4)
            if len(kv) and kv["label"].sum() else None,
        "kanvas_comparable_definition": "KANVAS kanvas/data.py filter on all labeled tasks, no random "
                                        "sample, no clipping; plan_gpu == 0 tasks are kept, as in KANVAS",
        "full_S3": recheck(s3),
        "S3_without_top_user": recheck(s3[s3["user"] != top_user]),
        "kanvas_comparable": recheck(kv) if len(kv) else "empty",
        "kanvas_comparable_without_top_user": recheck(kv[kv["user"] != top_user]) if len(kv) else "empty",
        "user_concentration_curve_S3": concentration(s3.assign(label=s3["label"].astype("int64"))),
    }
    if a.paper1_ref:
        res["paper1_published_reference"] = paper1_reference(a.paper1_ref)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
