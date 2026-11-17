#!/usr/bin/env python3
"""R1 step 6: figures from results/*.json (aggregates only).

Style copied from Paper 1's scripts/charts.py (paper1-ref): rcParams font 10, no top/right spines,
grid alpha 0.3, dpi 200; colours #444444 (all tasks), #2c7fb8 (2-GPU machines / first series),
#d95f02 (8-GPU machines / second series), #999999 (reference); legends frameon=False; n= labels.

Writes PNGs to figures/:
  auc_by_split.png         every grid cell (dots) + mean (bar) per split x model
  machine_class.png        failure rate by cap_gpu: Paper 1 (published), KANVAS-comparable, S3, S3 w/o top user
  plan_gpu_by_class.png    Paper 1 chart 2 re-drawn on the raw trace: three panels (S3, S3 w/o top user,
                           KANVAS-comparable), plan_gpu 10/25/50/100 x cap_gpu 2/8
  user_concentration.png   cumulative share of tasks / failures vs share of users (S3)
  funnel.png               task counts per funnel stage, recounted from the Parquet frame

Usage: python scripts/figures.py [--results DIR] [--out DIR]
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


def auc_by_split(grid, out):
    order = ["random", "user_folds", "time", "leave_top_user_out"]
    names = {"random": "Random 75/25\n(5 seeds)", "user_folds": "User-grouped\n(5 folds x 3 seeds)",
             "time": "Time-ordered\n(first 75% -> last 25%)", "leave_top_user_out": "Leave top\nuser out"}
    cells = [c for c in grid["cells"] if c["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.38
    for j, (model, col, lab) in enumerate([("logreg", C2, "Logistic regression"),
                                           ("gbm", C8, "Gradient boosting")]):
        for i, s in enumerate(order):
            v = [c["auc"] for c in cells if c["split"] == s and c["model"] == model]
            x = i + (j - 0.5) * w
            if not v:
                ax.text(x, 0.52, "not run", ha="center", color=C_REF, fontsize=8, rotation=90)
                continue
            ax.bar(x, np.mean(v) - 0.5, bottom=0.5, width=w - 0.04, color=col, alpha=0.45,
                   label=lab if i == 0 else None)
            ax.scatter(np.full(len(v), x), v, s=12, color=col, zorder=3)
            ax.annotate(f"{np.mean(v):.3f}", (x, max(v) + 0.008), ha="center", fontsize=7.5, color="#555555")
    ax.axhline(0.5, color="#333333", lw=0.9, ls="--")
    ax.set_xticks(range(len(order)), [names[s] for s in order], fontsize=8.5)
    ax.set_ylabel("Test ROC AUC (0.5 = chance)")
    lo = min([c["auc"] for c in cells] + [0.5])
    hi = max([c["auc"] for c in cells] + [0.6])
    ax.set_ylim(min(0.45, lo - 0.03), min(1.0, hi + 0.06))
    ax.set_title("Request features only: failure-prediction AUC by split\n(bar = mean, dots = individual cells)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "auc_by_split.png", **SAVE); plt.close(fig)


def machine_class(rec, out):
    series = []
    ref = rec.get("paper1_published_reference")
    if ref:
        series.append(("Paper 1, KANVAS sample (published)", C_REF,
                       {k.split(".")[0]: (v["failure_rate_pct"], v["tasks"]) for k, v in ref["rate_by_cap_gpu"].items()}))
    for key, col, lab in [("kanvas_comparable", C_ALL, "Raw trace, KANVAS-comparable"),
                          ("full_S3", C2, "Raw trace, all resolved GPU tasks"),
                          ("S3_without_top_user", C8, "Raw trace, without top user")]:
        if isinstance(rec.get(key), dict):
            series.append((lab, col, {r["cap_gpu"]: (r["failure_rate_pct"], r["tasks"])
                                      for r in rec[key]["machine_class_cap_gpu"]}))
    caps = ["2", "8"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.8 / len(series)
    for j, (lab, col, d) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(caps))]
        vals = [d.get(c, (np.nan, 0))[0] for c in caps]
        bars = ax.bar(xs, vals, w * 0.95, color=col, label=lab)
        for b, c in zip(bars, caps):
            if c in d:
                ax.annotate(f"{d[c][0]:.1f}%\nn={d[c][1]:,}", (b.get_x() + b.get_width() / 2, b.get_height() + 0.8),
                            ha="center", fontsize=6.5, color="#555555")
    ax.set_xticks(range(len(caps)), ["2-GPU machines", "8-GPU machines"])
    ax.set_xlabel("Machine class (cap_gpu: GPUs installed on the assigned machine)")
    ax.set_ylabel("Task failure rate (%)")
    top = np.nanmax([v[0] for _, _, d in series for v in d.values()] + [10])
    ax.set_ylim(0, top * 1.35)
    ax.set_title("Failure rate by machine class: Paper 1 vs the raw trace")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "machine_class.png", **SAVE); plt.close(fig)


def plan_gpu_by_class(rec, out):
    panels = [(k, t) for k, t in [("full_S3", "All resolved GPU tasks"),
                                  ("S3_without_top_user", "Without the top user"),
                                  ("kanvas_comparable", "KANVAS-comparable subset")]
              if isinstance(rec.get(k), dict)]
    vals = [10, 25, 50, 100]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    ymax = 0
    for ax, (key, title) in zip(axes, panels):
        rows = {(r["plan_gpu"], r["cap_gpu"]): r for r in rec[key]["plan_gpu_x_cap_gpu_paper1_values"]}
        for cap, color, lab in [(2, C2, "2-GPU machines"), (8, C8, "8-GPU machines")]:
            rates = [rows[(v, cap)]["failure_rate_pct"] if rows[(v, cap)]["tasks"] else np.nan for v in vals]
            ns = [rows[(v, cap)]["tasks"] for v in vals]
            bars = ax.bar([i + (-0.19 if cap == 2 else 0.19) for i in range(len(vals))], rates, 0.38,
                          color=color, label=lab)
            for bar, n in zip(bars, ns):  # Paper 1 stratified.py reports cells with n >= 100 only
                if n < 100:
                    bar.set_alpha(0.3); bar.set_hatch("//")
            for b, n, r in zip(bars, ns, rates):
                if n:
                    ax.annotate(f"n={n:,}", (b.get_x() + b.get_width() / 2, (0 if np.isnan(r) else r) + 1.2),
                                ha="center", fontsize=6.5, color="#555555", rotation=90, va="bottom")
            ymax = max(ymax, np.nanmax(rates + [0]))
        ax.set_xticks(range(len(vals)), ["0.10", "0.25", "0.50", "1.00"])
        ax.set_xlabel("GPU requested (plan_gpu / 100)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Task failure rate (%)")
    axes[0].set_ylim(0, ymax * 1.45 + 5)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Paper 1's request-shape chart, re-drawn on the raw trace "
                 "(faded, hatched bars: n < 100 tasks)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "plan_gpu_by_class.png", **SAVE); plt.close(fig)


def concentration(rec, out):
    cc = rec["user_concentration_curve_S3"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for k, col, lab in [("by_tasks", C2, "Share of tasks"), ("by_failed", C8, "Share of failures")]:
        ax.plot(np.array(cc[k]["user_fraction"]) * 100, np.array(cc[k]["cumulative_share"]) * 100,
                color=col, lw=1.8, label=lab)
    ax.plot([0, 100], [0, 100], "k--", lw=0.9, label="Equal shares")
    ax.set_xlabel("Users, largest first (% of users)")
    ax.set_ylabel("Cumulative share (%)")
    ax.set_title("Tasks and failures by user, all resolved GPU tasks")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "user_concentration.png", **SAVE); plt.close(fig)


def funnel(recon, out):
    st = recon["stages_from_parquet"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(st))
    ax.barh(y, [s["tasks"] for s in st], color=C_ALL, height=0.6)
    for yi, s in zip(y, st):
        pct = 100 * s["failed"] / s["tasks"] if s["tasks"] else 0
        ax.annotate(f"  {s['tasks']:,}  ({pct:.2f}% failed)", (s["tasks"], yi), va="center",
                    fontsize=7, color="#555555")
    ax.set_yticks(y, [s["stage"] for s in st], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(s["tasks"] for s in st) * 1.45)
    ax.set_xlabel("Tasks")
    ax.set_title("Funnel: raw task table to telemetry-joinable tasks")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(out / "funnel.png", **SAVE); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=R1 / "results")
    ap.add_argument("--out", type=pathlib.Path, default=R1 / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    done, missing = [], []
    gp, rp, fp = (a.results / "eval_grid.json", a.results / "paper1_recheck.json",
                  a.results / "frame_reconciliation.json")
    if gp.exists():
        auc_by_split(json.loads(gp.read_text()), a.out); done.append("auc_by_split.png")
    else:
        missing.append(f"auc_by_split.png ({gp.name} missing)")
    if rp.exists():
        rec = json.loads(rp.read_text())
        machine_class(rec, a.out)
        plan_gpu_by_class(rec, a.out)
        concentration(rec, a.out)
        done += ["machine_class.png", "plan_gpu_by_class.png", "user_concentration.png"]
    else:
        missing.append(f"machine_class/plan_gpu_by_class/user_concentration ({rp.name} missing)")
    if fp.exists():
        funnel(json.loads(fp.read_text()), a.out); done.append("funnel.png")
    else:
        missing.append(f"funnel.png ({fp.name} missing)")
    print("wrote:", ", ".join(done))
    if missing:
        print("MISSING:", "; ".join(missing))


if __name__ == "__main__":
    main()
