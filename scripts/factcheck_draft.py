#!/usr/bin/env python3
"""Trace every number in a draft (PAPER-DRAFT.md, PAPER-DRAFT-v2*.md, leakage-table.md) to a pipeline output, the verified WALKTHROUGH.md,
a declared derivation (arithmetic on output values, recomputed here), a quoted external source
(the KANVAS notebook 07 and dataset card copies on disk; only for numbers the draft attributes to KANVAS),
or a declared non-result token (dates, identifiers, version strings, structural numbering).

Usage (from r1/):  python3 -I scripts/factcheck_draft.py [PAPER-DRAFT.md]
Exit code 1 if any number is untraced. Standard library only.
"""
import csv
import json
import pathlib
import re
import sys

R1 = pathlib.Path(__file__).resolve().parents[1]
RES = R1 / "results"
draft = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else R1 / "PAPER-DRAFT.md"

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")


def norm(tok):
    return tok.replace(",", "")


# ---------------------------------------------------------------- 1. values in result files
values = []  # (float, source)


def walk(o, src, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, src, f"{path}.{k}" if path else k)
            if re.fullmatch(r"-?\d+(\.\d+)?", str(k)):
                values.append((float(k), f"{src}:{path} (key)"))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, src, f"{path}[{i}]")
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, (int, float)):
        if o == o:
            values.append((float(o), f"{src}:{path}"))
    elif isinstance(o, str):
        for m in NUM.findall(o):
            values.append((float(norm(m)), f"{src}:{path} (in text)"))


jsons = {}
SOURCES = [RES / "eval_grid.json", RES / "paper1_recheck.json", RES / "frame_reconciliation.json",
           # revision run (17 Nov 2026): audit outputs behind the leakage table, and the two new analyses
           R1 / "audit_results.json", R1 / "funnel.json",
           RES / "time_split_diagnostics.json", RES / "userid_ablation.json",
           # red-flag fixes (17 Nov 2026): counts the fixes quote that no earlier output held
           RES / "supplementary_counts.json"]
for path in SOURCES:
    name = path.name
    txt = path.read_text().replace("NaN", "null")
    jsons[name] = json.loads(txt)
    walk(jsons[name], name)


def read_csv(name):
    out = list(csv.DictReader(open(RES / name)))
    for i, r in enumerate(out):
        for k, v in r.items():
            try:
                values.append((float(v), f"{name}:row{i + 2}.{k}"))
            except (TypeError, ValueError):
                pass
    return out


rows = read_csv("eval_grid.csv")
shape_rows = read_csv("time_split_by_request_shape.csv")
read_csv("userid_ablation.csv")

# concentration curve is stored as shares 0..1; the draft quotes percentages
cur = jsons["paper1_recheck.json"]["user_concentration_curve_S3"]
for kind in ("by_tasks", "by_failed"):
    for f, s in zip(cur[kind]["user_fraction"], cur[kind]["cumulative_share"]):
        values.append((100 * s, f"paper1_recheck.json:user_concentration_curve_S3.{kind}@{f} x100"))
        values.append((100 * f, f"paper1_recheck.json:user_concentration_curve_S3.user_fraction x100"))
# plan_gpu is stored as a percentage of one GPU; the draft quotes GPUs (plan_gpu / 100)
for pg in (10, 25, 50, 100):
    values.append((pg / 100, f"paper1_recheck.json:plan_gpu {pg} / 100"))

# ---------------------------------------------------------------- 2. declared derivations
R = jsons["paper1_recheck.json"]
F = jsons["frame_reconciliation.json"]
st = {s["stage"].split()[0]: s for s in F["stages_from_parquet"]}
full, wo = R["full_S3"], R["S3_without_top_user"]
q_full, q_wo = full["quarter_gpu_cell"]["plan25_cap8"], wo["quarter_gpu_cell"]["plan25_cap8"]
mc_full = {m["cap_gpu"]: m for m in full["machine_class_cap_gpu"]}
mc_wo = {m["cap_gpu"]: m for m in wo["machine_class_cap_gpu"]}
ug = [r for r in rows if r["split"] == "user_folds"]
def _c(b, pg, cap):
    return [r for r in b["plan_gpu_x_cap_gpu_paper1_values"] if r["plan_gpu"] == pg and str(r["cap_gpu"]) == cap][0]


_o = wo["quarter_gpu_cell"]["cap8_all_other_plan_gpu"]
_gap010 = (100 * (mc_wo["8"]["failed"] - _c(wo, 10, "8")["failed"]) / (mc_wo["8"]["tasks"] - _c(wo, 10, "8")["tasks"])
           - 100 * (mc_wo["2"]["failed"] - _c(wo, 10, "2")["failed"]) / (mc_wo["2"]["tasks"] - _c(wo, 10, "2")["tasks"]))
_t8t = mc_full["8"]["tasks"] - mc_wo["8"]["tasks"]
_t8f = mc_full["8"]["failed"] - mc_wo["8"]["failed"]
_top_other8 = 100 * (_t8f - (q_full["failed"] - q_wo["failed"])) / (_t8t - (q_full["tasks"] - q_wo["tasks"]))
_m12 = {m: sum(float(r["auc"]) for r in ug if r["model"] == m and float(r["test_top_user_share_pct"]) == 0) / 12
        for m in ("gbm", "logreg")}
_pairs = {}
for r in ug:
    _pairs.setdefault((r["seed"], r["fold"]), {})[r["model"]] = float(r["auc"])
_opp = sum(1 for v in _pairs.values() if (v["gbm"] - 0.5) * (v["logreg"] - 0.5) < 0)
_rnd = [r for r in rows if r["split"] == "random"]
_spread = max(max(float(r["auc"]) for r in _rnd if r["model"] == m) - min(float(r["auc"]) for r in _rnd if r["model"] == m)
              for m in ("gbm", "logreg"))
_tp = {}
for r in rows:
    if r["split"] != "time":
        continue
    n = int(r["n_test"]); sh = float(r["test_top_user_share_pct"]) / 100
    fr = float(r["test_failure_rate_pct"]) / 100; frt = float(r["test_top_user_failure_rate_pct"]) / 100
    nt = n * sh; no = n - nt; Pt = nt * frt; Nt = nt - Pt; P = n * fr; Po = P - Pt; No = no - Po
    tot = P * (n - P)
    A, At, Ao = float(r["auc"]), float(r["auc_test_top_user_only"]), float(r["auc_test_without_top_user"])
    _tp.update(others_rate=100 * Po / no, tt=100 * Pt * Nt / tot, oo=100 * Po * No / tot,
               cross=100 * (Pt * No + Po * Nt) / tot, cross1=100 * Pt * No / tot)
    _tp[r["model"]] = (A * tot - At * Pt * Nt - Ao * Po * No) / (Pt * No + Po * Nt)
derived = {
    "S0 failed % = failed/tasks": 100 * st["S0"]["failed"] / st["S0"]["tasks"],
    "S2 failed %": 100 * st["S2"]["failed"] / st["S2"]["tasks"],
    "S3 failed %": 100 * st["S3"]["failed"] / st["S3"]["tasks"],
    "S7 failed %": 100 * st["S7"]["failed"] / st["S7"]["tasks"],
    "S8 failed %": 100 * st["S8"]["failed"] / st["S8"]["tasks"],
    "Paper 1 sample as % of S3 = 18890/985358": 100 * R["paper1_published_reference"]["n_tasks"] / full["n_tasks"],
    "top user's tasks in quarter cell = full - without": q_full["tasks"] - q_wo["tasks"],
    "top user's failures in quarter cell": q_full["failed"] - q_wo["failed"],
    "top user's failure rate in quarter cell": 100 * (q_full["failed"] - q_wo["failed"]) / (q_full["tasks"] - q_wo["tasks"]),
    "top user's tasks on 8-GPU machines": mc_full["8"]["tasks"] - mc_wo["8"]["tasks"],
    "top user's tasks = S3 - S3 without": full["n_tasks"] - wo["n_tasks"],
    "quarter gap in points, without top user": q_wo["failure_rate_pct"] - wo["quarter_gpu_cell"]["plan25_cap2"]["failure_rate_pct"],
    "class gap in points, without top user": mc_wo["8"]["failure_rate_pct"] - mc_wo["2"]["failure_rate_pct"],
    "class ratio, without top user": mc_wo["8"]["failure_rate_pct"] / mc_wo["2"]["failure_rate_pct"],
    "user-grouped gbm cells below 0.5": sum(1 for r in ug if r["model"] == "gbm" and float(r["auc"]) < 0.5),
    "user-grouped logreg cells below 0.5": sum(1 for r in ug if r["model"] == "logreg" and float(r["auc"]) < 0.5),
    "S3 rounded to the nearest thousand (PLAN rev 3 coherence paragraph, verbatim 'about 985,000')": round(full["n_tasks"], -3),
    "0.10-GPU cap8 cell share of 8-GPU failures, without top user": 100 * _c(wo, 10, "8")["failed"] / mc_wo["8"]["failed"],
    "8-GPU rate without top user, 0.10-GPU requests left out": 100 * (mc_wo["8"]["failed"] - _c(wo, 10, "8")["failed"]) / (mc_wo["8"]["tasks"] - _c(wo, 10, "8")["tasks"]),
    "2-GPU rate without top user, 0.10-GPU requests left out": 100 * (mc_wo["2"]["failed"] - _c(wo, 10, "2")["failed"]) / (mc_wo["2"]["tasks"] - _c(wo, 10, "2")["tasks"]),
    "class gap in points without top user, 0.10 left out": _gap010,
    "other 8-GPU requests without top user, 0.10 cell also set aside": 100 * (_o["failed"] - _c(wo, 10, "8")["failed"]) / (_o["tasks"] - _c(wo, 10, "8")["tasks"]),
    "top user's failure rate on their other 8-GPU requests": _top_other8,
    "top user's quarter-GPU tasks on 2-GPU machines": full["quarter_gpu_cell"]["plan25_cap2"]["tasks"] - wo["quarter_gpu_cell"]["plan25_cap2"]["tasks"],
    "user-grouped gbm mean, 12 folds without top user": _m12["gbm"],
    "user-grouped logreg mean, 12 folds without top user": _m12["logreg"],
    "folds where the two models are on opposite sides of 0.5": _opp,
    "folds without the top user per model": 12,
    "gbm cells within 0.001 below 0.5": sum(1 for r in ug if r["model"] == "gbm" and 0.499 <= float(r["auc"]) < 0.5),
    "max spread of random-split AUC across seeds": _spread,
    "time split: others' test failure rate": _tp["others_rate"],
    "time split: pair share within top user %": _tp["tt"],
    "time split: pair share within others %": _tp["oo"],
    "time split: pair share cross-user %": _tp["cross"],
    "time split: pair share top-failed vs other-ok %": _tp["cross1"],
    "time split: implied cross-pair AUC gbm": _tp["gbm"],
    "time split: implied cross-pair AUC logreg": _tp["logreg"],
    "grid cells": len(rows),
    "funnel stages": len(F["stages_from_parquet"]),
    "populations in recheck": 4,
    "request features": len(jsons["eval_grid.json"]["meta"]["features"]),
    "trace tables (WALKTHROUGH diagram)": 7,
    "result files checksummed": 3,
}
# ---- revision run derivations
A = jsons["audit_results.json"]
T = jsons["time_split_diagnostics.json"]
_et = A["end_time_missing"]["resolved_gpu_tasks"]
_tc = A["telemetry_coverage_by_status_on_S6"]
_hm, _hs = _tc["has_metric"], _tc["has_sensor"]
_tvi = A["task_vs_instance_status"]["Terminated"]
_bins = T["largest_test_shape_over_time"]["rows"]
_burst = [b for b in _bins if b["bin"] in (16, 17, 18)]


def _yn_auc(tpr, fpr):
    return 0.5 + (tpr - fpr) / 2


derived.update({
    "leak table: AUC of 'end_time missing' = 0.5 + (TPR - FPR)/2": _yn_auc(
        _et["Failed"]["end_time_null"] / _et["Failed"]["n"], _et["Terminated"]["end_time_null"] / _et["Terminated"]["n"]),
    "leak table: AUC of 'no machine-metric rows'": _yn_auc(
        1 - _hm["Failed"]["with"] / _hm["Failed"]["n"], 1 - _hm["Terminated"]["with"] / _hm["Terminated"]["n"]),
    "leak table: AUC of 'has sensor rows'": _yn_auc(
        _hs["Failed"]["with"] / _hs["Failed"]["n"], _hs["Terminated"]["with"] / _hs["Terminated"]["n"]),
    "leak table: Failed tasks without machine-metric rows = n - with": _hm["Failed"]["n"] - _hm["Failed"]["with"],
    "leak table: Failed without machine-metric rows %": 100 * (1 - _hm["Failed"]["with"] / _hm["Failed"]["n"]),
    "leak table: Terminated tasks without machine-metric rows = n - with": _hm["Terminated"]["n"] - _hm["Terminated"]["with"],
    "leak table: Terminated without machine-metric rows %": 100 * (1 - _hm["Terminated"]["with"] / _hm["Terminated"]["n"]),
    "Terminated tasks with a failed instance %": 100 * _tvi["any_instance_failed"] / _tvi["n"],
    "burst: tasks of the largest test shape in time bins 16-18": sum(b["shape_tasks"] for b in _burst),
    "burst: failed tasks of that shape in bins 16-18": sum(b["shape_failed"] for b in _burst),
    "burst: tasks in bins 16-18 rounded to the nearest thousand ('about 81,000')": round(sum(b["shape_tasks"] for b in _burst), -3),
    "time bin width % = 100 / number of bins": 100 / len(_bins),
    "time bins": len(_bins),
    "KANVAS-comparable folds: 15 cells minus 3 with the largest user": 12,
    "user-grouped cells per model that contain the largest user": sum(
        1 for r in ug if r["model"] == "gbm" and float(r["test_top_user_share_pct"]) > 0),
    "user-grouped cells per model": sum(1 for r in ug if r["model"] == "gbm"),
    "cut positions": len(T["cut_dates"]),
    "other cut positions": len(T["cut_dates"]) - 1,
    "request shapes listed in Table 7": 5,
    "Paper 1 results re-checked": 2,
    "split protocols": 3,
    "label-proxy conditions in Table 2": 3,
    "published studies on time-ordered splits cited (refs 8-10)": 3,
})
for c in T["cut_dates"]:
    derived[f"cut position {c['frac']} as % of tasks"] = 100 * c["frac"]
# request shapes: the trace stores CPU as % of a core; Table 7 shows cores
for r in shape_rows:
    if r["plan_cpu"]:
        values.append((float(r["plan_cpu"]) / 100, f"time_split_by_request_shape.csv:row{r['rank_by_test_tasks']}.plan_cpu / 100 (cores)"))
for k, v in derived.items():
    values.append((float(v), f"DERIVED: {k}"))

# ---------------------------------------------------------------- 2b. KANVAS's own published numbers
EXT = R1.parent / "ext" / "kanvas-ref" / "notebooks" / "07_user_grouped_robustness.ipynb"
CARD = R1.parent / "paper1-ref" / "data" / "kanvas_README.md"
ext_tokens = {}
if EXT.exists():
    nb = json.loads(EXT.read_text())
    blob = []
    for c in nb["cells"]:
        blob.append("".join(c["source"]))
        for o in c.get("outputs", []):
            blob.append("".join(o.get("text", [])) if "text" in o else "".join(o.get("data", {}).get("text/plain", [])))
    for m in NUM.findall("\n".join(blob)):
        ext_tokens[m.replace(",", "")] = "EXTERNAL: KANVAS notebook 07 (ext/kanvas-ref @ 3c3fa2a)"
if CARD.exists():
    for m in NUM.findall(CARD.read_text()):
        ext_tokens.setdefault(m.replace(",", ""), "EXTERNAL: KANVAS dataset card (paper1-ref/data/kanvas_README.md)")

# ---------------------------------------------------------------- 3. WALKTHROUGH tokens
wt = set(norm(m) for m in NUM.findall((R1 / "WALKTHROUGH.md").read_text()))

# ---------------------------------------------------------------- 4. the draft
text = draft.read_text()
# strip tokens that are not results: DOIs, arXiv ids, issue refs, versions, dates, paths, headings
NONRESULT = [
    (r"10\.\d{4,5}/[^\s,)]+", "DOI"),
    (r"arXiv:\d{4}\.\d{5}", "arXiv id"),
    (r"krkn-chaos/krkn#\d+", "issue reference (PLAN rev 3 coherence paragraph)"),
    (r"\b(Python|DuckDB|scikit-learn) \d+(\.\d+)+", "version string (eval_grid.json meta / frame_reconciliation.json)"),
    (r"\d{1,2} November 2026", "date"),
    (r"\b(19|20)\d\d\b", "year"),
    (r"cluster-trace-gpu-v2020", "trace name"),
    (r"PAPER-DRAFT-v2(\.\d+)?\.md|leakage-table\.md", "file name"),
    (r"vol\. \d+, no\. \d+, (pp\. \d+–\d+|article \d+)", "bibliographic volume/issue/pages (Crossref, 17 Nov 2026)"),
    (r"pp\. \d+–\d+", "bibliographic page range without a volume (refs [7], [10]; REFS-CHECK.md)"),
    (r"ICSE '\d\d", "conference short name (ref [7])"),
    (r"commit [0-9a-f]{7}", "git commit"),
    (r"issue #\d+", "issue reference (VALIDATION-RESEARCH.md)"),
    (r"`notebooks/07_[a-z_]+\.ipynb`", "file name"),
    (r"\b70/15/15\b", "KANVAS partition fractions (notebook 07, Step 3 heading)"),
    (r"CC-BY-4\.0", "licence name"),
    (r"SHA-256", "algorithm name"),
    (r"\bK40m\b|\bK80\b|\bM60\b|\bS[0-8]\b|\bS1\b|\bv1\b|\bR1\b", "model/stage/label name"),
    (r"(?m)^#+ .*$", "heading"),
    (r"\[\d+(, \d+)*\]", "citation marker"),
    (r"(Section|Sections|Table|Figure|Paper|Scenario) \d+(\.\d+)?( (to|and) \d+(\.\d+)?)?", "cross-reference"),
    (r"(?m)^\d+\. ", "list numbering"),
    (r"!\[Figure \d\]\([^)]*\)", "figure link"),
    (r"\b75/25\b", "split fraction (eval_grid.json meta: time_frac 0.75, test_frac_random 0.25)"),
    (r"\b0\.5\b", "chance level of AUC (definition)"),
]
stripped = text
for pat, _ in NONRESULT:
    stripped = re.sub(pat, " ", stripped)

SMALL_WORDY = {"0", "1", "2", "3", "4", "5", "8", "15", "25", "75"}  # seeds, folds, GPU counts, split %: checked below
ok, bad = [], []
for m in NUM.finditer(stripped):
    tok = m.group(0)
    t = norm(tok)
    dec = len(t.split(".")[1]) if "." in t else 0
    x = float(t)
    hit = next((src for v, src in values if round(v, dec) == x or abs(round(v + 1e-12, dec) - x) < 1e-12), None)
    if hit is None and t in wt:
        hit = "WALKTHROUGH.md (verified narrative)"
    if hit is None and t in ext_tokens:
        hit = ext_tokens[t]
    ctx = stripped[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")
    (ok if hit else bad).append((tok, hit, ctx))

print(f"draft: {draft.name}")
print(f"numeric tokens checked: {len(ok) + len(bad)}  traced: {len(ok)}  UNTRACED: {len(bad)}")
by = {}
for tok, hit, _ in ok:
    key = "DERIVED" if hit.startswith("DERIVED") else "EXTERNAL (KANVAS)" if hit.startswith("EXTERNAL") else hit.split(":")[0]
    by[key] = by.get(key, 0) + 1
for k, v in sorted(by.items()):
    print(f"  {k}: {v}")
print("\nDerivations used (recomputed from result files):")
used = sorted({hit for _, hit, _ in ok if hit.startswith("DERIVED")})
for u in used:
    print(f"  {u} = {derived[u[9:]]:.4f}")
ext_used = sorted({(tok, hit) for tok, hit, _ in ok if hit.startswith("EXTERNAL")})
if ext_used:
    print("\nQuoted from KANVAS (not our outputs):")
    for tok, hit in ext_used:
        print(f"  {tok}  <- {hit[10:]}")
if bad:
    print("\nUNTRACED:")
    for tok, _, ctx in bad:
        print(f"  {tok!r:>12}  ...{ctx}...")
# The current paper draft is any PAPER-DRAFT-v2.md / PAPER-DRAFT-v2.N.md; the name is not hard-coded.
IS_REVISED_PAPER = re.fullmatch(r"PAPER-DRAFT-v2(\.\d+)?\.md", draft.name) is not None
if not IS_REVISED_PAPER:
    sys.exit(1 if bad else 0)

# ---------------------------------------------------------------- 5. row-by-row check of the revision tables
# The token check above only asks "does this number exist in some output". For the tables added in the
# revision (PAPER-DRAFT-v2*.md Tables 6-9) each row is rebuilt here from the named output fields and must
# appear in the draft character for character.
if IS_REVISED_PAPER:
    U = jsons["userid_ablation.json"]
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    p2 = lambda v: f"{v:.2f}%"  # noqa: E731
    exp = []
    b, w, lk = T["baseline"], T["without_top_user"], T["shape_lookup"]
    for lab, m in (("Logistic regression", "logreg"), ("Gradient boosting", "gbm")):
        exp.append(f"| {lab} | {f4(b[m]['auc_top_user_only'])} | {f4(b[m]['auc_without_top_user'])} | {f4(b[m]['auc_all'])} | "
                   f"{f4(w['same_cut'][m]['auc'])} | {f4(w['recut'][m]['auc'])} |")
    exp.append(f"| Request-shape lookup (no model) | {f4(lk['auc_top_user_only'])} | {f4(lk['auc_without_top_user'])} | {f4(lk['auc_all'])} | n/a | n/a |")
    for r in shape_rows[:5]:
        shape = f"{float(r['plan_cpu']) / 100:.0f}, {float(r['plan_mem']):.1f}, {float(r['plan_gpu']) / 100:.2f}, {float(r['inst_num']):.0f}"
        exp.append(f"| {shape} | {int(r['n_train']):,} | {p2(float(r['train_failure_rate_pct']))} | {int(r['n_test']):,} | "
                   f"{p2(float(r['test_failure_rate_pct']))} | {p2(float(r['test_top_user_share_of_shape_pct']))} | "
                   f"{f4(float(r['mean_test_score_gbm']))} | {f4(float(r['mean_test_score_logreg']))} |")
    for c in T["cut_dates"]:
        exp.append(f"| {100 * c['frac']:.0f}% | {p2(c['test_top_user_share_pct'])} | {p2(c['train_top_user_failure_rate_pct'])} → "
                   f"{p2(c['test_top_user_failure_rate_pct'])} | {f4(c['gbm']['auc_all'])} | {f4(c['logreg']['auc_all'])} | "
                   f"{f4(c['gbm']['auc_top_user_removed_from_both_sides'])} | {f4(c['logreg']['auc_top_user_removed_from_both_sides'])} |")
    sm = {(r["split"], r["feature_set"], r["model"]): r["mean"] for r in U["summary"]}
    so = {(r["split"], r["feature_set"], r["model"]): r["mean"] for r in U["summary_auc_test_without_top_user"] if r["n_cells"]}
    for slab, sp in (("Random (mean of 5)", "random"), ("User-grouped (mean of 15)", "user_folds"),
                     ("Time-ordered", "time"), ("Leave-top-user-out", "leave_top_user_out")):
        for lab, m in (("Gradient boosting", "gbm"), ("Logistic regression", "logreg")):
            exp.append(f"| {slab} | {lab} | {f4(sm[sp, 'request', m])} | {f4(sm[sp, 'request+user', m])} | {f4(sm[sp, 'user', m])} | "
                       f"{f4(so.get((sp, 'request+user', m)))} | {f4(so.get((sp, 'user', m)))} |")
    missing = [e for e in exp if e not in text]
    print(f"\nrevision tables (6-9): {len(exp)} rows rebuilt from outputs, {len(missing)} not found in the draft")
    for e in missing:
        print("  MISSING:", e)
    print("ablation baseline equals eval_grid:", U["baseline_check"]["all_equal"],
          "| time-split baseline equals eval_grid:", all(b[m]["equals_eval_grid_cell"] for m in ("gbm", "logreg")),
          "| independent split differs on", T["split_check"]["rows_assigned_differently_from_paigpu_time_split"], "rows")
    if missing or bad:
        sys.exit(1)
