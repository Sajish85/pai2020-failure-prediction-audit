#!/usr/bin/env python3
"""R1 revision figure, from results/time_split_diagnostics.json (aggregates only).

figures/time_split_shape.png
  left:  the request shape with the most test-period tasks, counted in 20 equal-count time bins
         (successful and failed tasks stacked), with the 75% cut marked
  right: test AUC at six cut positions: gradient boosting and logistic regression on all tasks,
         and gradient boosting with the largest user removed from both sides

Style as scripts/figures.py (copied from Paper 1's charts.py).
Usage: python scripts/figures_revision.py [--results DIR] [--out DIR]
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R1 = pathlib.Path(__file__).resolve().parents[1]
C_ALL, C2, C8, C_REF = "#444444", "#2c7fb8", "#d95f02", "#999999"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})
SAVE = {"dpi": 200, "metadata": {"Software": None}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=R1 / "results")
    ap.add_argument("--out", type=pathlib.Path, default=R1 / "figures")
    a = ap.parse_args(argv)
    d = json.loads((a.results / "time_split_diagnostics.json").read_text())
    rows = d["largest_test_shape_over_time"]["rows"]
    frac = d["meta"]["time_frac"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.array([(r["tasks_from_pct"] + r["tasks_to_pct"]) / 2 for r in rows])
    failed = np.array([r["shape_failed"] for r in rows])
    ok = np.array([r["shape_tasks"] - r["shape_failed"] for r in rows])
    ax.bar(x, ok / 1000, width=4.2, color=C2, label="Terminated")
    ax.bar(x, failed / 1000, width=4.2, bottom=ok / 1000, color=C8, label="Failed")
    ax.axvline(100 * frac, color="#333333", lw=0.9, ls="--")
    ax.text(100 * frac - 1.5, ax.get_ylim()[1] * 0.93, "75% cut", ha="right", fontsize=8.5, color="#333333")
    ax.set_xlabel("Position in the trace (% of tasks, in start-time order)")
    ax.set_ylabel("Tasks of this request shape (thousands)")
    ax.set_title("One request shape over time\n(1 core, 29.3 GB, 0.25 GPU, 1 instance)", fontsize=10)
    ax.legend(frameon=False, loc="upper left")

    cuts = d["cut_dates"]
    cx = [100 * c["frac"] for c in cuts]
    bx.plot(cx, [c["gbm"]["auc_all"] for c in cuts], "o-", color=C8, label="Gradient boosting, all tasks")
    bx.plot(cx, [c["gbm"]["auc_top_user_removed_from_both_sides"] for c in cuts], "o--", color=C8, alpha=0.55,
            label="Gradient boosting, largest user removed")
    bx.plot(cx, [c["logreg"]["auc_all"] for c in cuts], "s-", color=C2, label="Logistic regression, all tasks")
    bx.axhline(0.5, color="#333333", lw=0.9, ls="--")
    bx.set_xticks(cx)
    bx.set_ylim(0.2, 0.9)
    bx.set_xlabel("Cut position (% of tasks used for training)")
    bx.set_ylabel("Test ROC AUC (0.5 = chance)")
    bx.set_title("Time-ordered split at six cut positions", fontsize=10)
    bx.legend(frameon=False, fontsize=7.5, loc="lower left")
    a.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out / "time_split_shape.png", **SAVE); plt.close(fig)
    print("wrote", a.out / "time_split_shape.png")


if __name__ == "__main__":
    main()
