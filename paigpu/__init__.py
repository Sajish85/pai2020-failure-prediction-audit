"""paigpu: load the Alibaba PAI GPU 2020 task frame and build leak-safe splits.

Public API (see r1/SCHEMA.md for every column and type):

    load_trace(path=None, base="S3")  -> pandas.DataFrame, one row per task
    user_folds(k, seed, frame=None)    -> DataFrame [job_name, task_name, user, fold]
    time_split(frac, frame=None)       -> DataFrame [job_name, task_name, start_time, split]

The frame is produced by `python -m paigpu.convert` (or `make frame`). It holds
trace rows and must stay local: the trace has no licence.

Leakage rule (r1/AUDIT.md findings 3-4): no predictor may use end_time,
duration, the task's own sensor rows or its own machine-metric rows. The frame
does not carry end_time or any telemetry values; REQUEST_FEATURES lists the only
columns R1's grid uses.
"""
import pathlib

import numpy as np
import pandas as pd

__all__ = ["load_trace", "user_folds", "time_split", "MACHINE_CLASS",
           "REQUEST_FEATURES", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES", "KEY", "DEFAULT_FRAME"]

R1 = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_FRAME = R1 / "data" / "parquet" / "tasks_frame.parquet"
KEY = ["job_name", "task_name"]
# Request features = Paper 1's "Request" family (paper1-ref/scripts/charts.py, `fam`):
# what the user asked for. gpu_type is NOT a request: the trace README (l.183) defines it as
# "type of GPUs assigned to this task", i.e. machine-side, and it nearly fixes cap_gpu.
NUMERIC_FEATURES = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"]
CATEGORICAL_FEATURES = []
REQUEST_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Paper 1's machine class: GPUs installed on the task's assigned (first-instance) machine,
# pai_machine_spec.cap_gpu (paper1-ref/data/data_dictionary.md; stratified.py groups by it).
MACHINE_CLASS = "cap_gpu"


def load_trace(path=None, base="S3"):
    """Load the task frame.

    base="S3" (default): resolved GPU tasks (status Failed/Terminated, plan_gpu > 0,
    user known), the analysis base in r1/funnel.json. base="kanvas": tasks passing KANVAS's
    filter without its random sample (column kanvas_cmp; see paigpu/convert.py).
    base="all": every task row (S0).
    `label` is 1 for Failed, 0 for Terminated (NA outside S2).
    """
    path = pathlib.Path(path) if path else DEFAULT_FRAME
    df = pd.read_parquet(path)
    if base in ("S3", "kanvas"):
        df = df[df["in_s3" if base == "S3" else "kanvas_cmp"].fillna(False).astype(bool)].copy()
        df["label"] = df["label"].astype("int8")
    elif base != "all":
        raise ValueError(f"base must be 'S3', 'kanvas' or 'all', got {base!r}")
    return df.reset_index(drop=True)


def _frame(frame):
    return load_trace() if frame is None else frame


def user_folds(k, seed, frame=None):
    """Assign every task to one of k folds so that no user spans two folds.

    Users are shuffled with numpy's default_rng(seed). In that order, each user goes
    to the fold that currently holds the fewest tasks, which keeps the fold sizes close
    while staying seed-dependent. The largest user (about 29% of S3 tasks; funnel.json)
    always fills most of one fold on its own, so fold sizes cannot be equal.

    Returns a DataFrame with columns job_name, task_name, user, fold (int8, 0..k-1),
    in the same row order as `frame`.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    df = _frame(frame)
    sizes = df.groupby("user", sort=True).size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    load = np.zeros(k, dtype=np.int64)
    assign = {}
    for u in order:
        f = int(np.argmin(load))  # ties -> lowest fold index (deterministic)
        assign[u] = f
        load[f] += sizes[u]
    out = df[KEY + ["user"]].copy()
    out["fold"] = out["user"].map(assign).astype("int8")
    return out


def time_split(frac, frame=None):
    """Time-ordered split: the earliest `frac` of tasks by start_time are train, the rest test.

    Ties on start_time are broken by (job_name, task_name), so the cut is
    deterministic. Users can appear on both sides; this split tests drift over
    time, not generalisation to new users.

    Returns a DataFrame with columns job_name, task_name, start_time, split ('train'/'test'),
    in the same row order as `frame`.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    df = _frame(frame)
    if df["start_time"].isna().any():
        raise ValueError("start_time has nulls; time_split needs every task timed")
    order = df.sort_values(["start_time", "job_name", "task_name"], kind="mergesort").index
    n_train = int(np.floor(frac * len(df)))
    split = pd.Series("test", index=df.index)
    split.loc[order[:n_train]] = "train"
    out = df[KEY + ["start_time"]].copy()
    out["split"] = split.values
    return out
