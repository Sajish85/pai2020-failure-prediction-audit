#!/usr/bin/env python3
"""R1 day-1 audit of the Alibaba PAI GPU 2020 trace.

Loads the seven checksum-verified CSVs (no header row; column names come from
the .header files shipped with the trace mirror) into a local DuckDB file and
writes:
  r1/audit_results.json  schema / status / key-integrity / end_time checks
  r1/funnel.json         raw trace -> resolved GPU tasks -> telemetry-joinable

Nothing here writes trace rows anywhere except the local (never-published)
DuckDB file under r1/data/.

Usage: r1/.venv/bin/python r1/scripts/audit.py
"""
import json
import pathlib
import time

import duckdb

R1 = pathlib.Path(__file__).resolve().parents[1]
RAW = R1 / "data" / "raw"
CSV = R1 / "data" / "csv"
DB = R1 / "data" / "pai2020.duckdb"
TABLES = [
    "pai_job_table",
    "pai_task_table",
    "pai_instance_table",
    "pai_sensor_table",
    "pai_group_tag_table",
    "pai_machine_spec",
    "pai_machine_metric",
]
KANVAS_TASKS = 18890  # KANVAS card / Paper 1; used for scale comparison only

t0 = time.time()
con = duckdb.connect(str(DB))
con.execute("SET threads TO 6")
con.execute("SET memory_limit = '8GB'")

out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "duckdb_version": duckdb.__version__, "tables": {}}


def q(sql):
    return con.execute(sql).fetchall()


def q1(sql):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------- load + schema
for t in TABLES:
    cols = (RAW / f"{t}.header").read_text().strip().split(",")
    names = "[" + ",".join(f"'{c}'" for c in cols) + "]"
    con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute(
        f"CREATE TABLE {t} AS SELECT * FROM read_csv('{CSV / (t + '.csv')}', "
        f"header=false, names={names}, sample_size=-1)"
    )
    desc = q(f"DESCRIBE {t}")
    n = q1(f"SELECT count(*) FROM {t}")
    nulls = {}
    for name, typ, *_ in desc:
        nulls[name] = q1(f'SELECT count(*) FROM {t} WHERE "{name}" IS NULL')
    out["tables"][t] = {
        "rows": n,
        "header_columns": cols,
        "columns": [{"name": d[0], "type": d[1], "nulls": nulls[d[0]],
                     "null_pct": round(100 * nulls[d[0]] / n, 4) if n else None}
                    for d in desc],
    }
    print(f"loaded {t}: {n} rows, {len(desc)} cols  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- status values
status = {}
for t in ["pai_job_table", "pai_task_table", "pai_instance_table"]:
    status[t] = {s if s is not None else "<NULL>": c for s, c in
                 q(f"SELECT status, count(*) FROM {t} GROUP BY 1 ORDER BY 2 DESC")}
out["status_values"] = status

# ---------------------------------------------------------------- key integrity
ki = {}
ki["job_rows"] = q1("SELECT count(*) FROM pai_job_table")
ki["job_distinct_job_name"] = q1("SELECT count(DISTINCT job_name) FROM pai_job_table")
ki["job_distinct_inst_id"] = q1("SELECT count(DISTINCT inst_id) FROM pai_job_table")
ki["job_names_with_gt1_inst_id"] = q1(
    "SELECT count(*) FROM (SELECT job_name FROM pai_job_table GROUP BY 1 HAVING count(DISTINCT inst_id) > 1)")
ki["task_rows"] = q1("SELECT count(*) FROM pai_task_table")
ki["task_distinct_job_task"] = q1(
    "SELECT count(*) FROM (SELECT DISTINCT job_name, task_name FROM pai_task_table)")
ki["task_rows_job_not_in_job_table"] = q1(
    "SELECT count(*) FROM pai_task_table t WHERE NOT EXISTS "
    "(SELECT 1 FROM pai_job_table j WHERE j.job_name = t.job_name)")
ki["jobs_without_task"] = q1(
    "SELECT count(*) FROM pai_job_table j WHERE NOT EXISTS "
    "(SELECT 1 FROM pai_task_table t WHERE t.job_name = j.job_name)")
ki["instance_rows"] = q1("SELECT count(*) FROM pai_instance_table")
ki["instance_rows_task_not_in_task_table"] = q1(
    "SELECT count(*) FROM pai_instance_table i WHERE NOT EXISTS "
    "(SELECT 1 FROM pai_task_table t WHERE t.job_name = i.job_name AND t.task_name = i.task_name)")
ki["instance_distinct_worker_name"] = q1("SELECT count(DISTINCT worker_name) FROM pai_instance_table")
ki["instance_rows_null_machine"] = q1("SELECT count(*) FROM pai_instance_table WHERE machine IS NULL")
ki["instance_machines_not_in_spec"] = q1(
    "SELECT count(DISTINCT machine) FROM pai_instance_table i WHERE machine IS NOT NULL AND NOT EXISTS "
    "(SELECT 1 FROM pai_machine_spec m WHERE m.machine = i.machine)")
ki["machine_spec_rows"] = q1("SELECT count(*) FROM pai_machine_spec")
ki["machine_spec_distinct_machine"] = q1("SELECT count(DISTINCT machine) FROM pai_machine_spec")
ki["sensor_distinct_worker_name"] = q1("SELECT count(DISTINCT worker_name) FROM pai_sensor_table")
ki["metric_distinct_worker_name"] = q1("SELECT count(DISTINCT worker_name) FROM pai_machine_metric")
ki["group_tag_distinct_inst_id"] = q1("SELECT count(DISTINCT inst_id) FROM pai_group_tag_table")
ki["group_tag_user_mismatch_vs_job"] = q1(
    "SELECT count(*) FROM pai_group_tag_table g JOIN pai_job_table j USING (inst_id) WHERE g.user <> j.user")
out["key_integrity"] = ki

# ---------------------------------------------------------------- time ranges
tr = {}
for t in ["pai_job_table", "pai_task_table", "pai_instance_table"]:
    lo, hi = con.execute(f"SELECT min(start_time), max(start_time) FROM {t}").fetchone()
    tr[t] = {"start_time_min": lo, "start_time_max": hi,
             "span_days": round((hi - lo) / 86400, 2) if lo is not None else None}
out["time_ranges_seconds"] = tr

# ---------------------------------------------------------------- end_time missing vs status
et = {}
for t in ["pai_job_table", "pai_task_table", "pai_instance_table"]:
    rows = q(f"SELECT status, count(*) AS n, count(*) FILTER (WHERE end_time IS NULL) AS miss, "
             f"count(*) FILTER (WHERE start_time IS NULL) AS miss_start "
             f"FROM {t} GROUP BY 1 ORDER BY 2 DESC")
    et[t] = {(s if s is not None else "<NULL>"): {"n": n, "end_time_null": m,
                                                   "end_time_null_pct": round(100 * m / n, 4),
                                                   "start_time_null": ms}
             for s, n, m, ms in rows}
    tot = q1(f"SELECT count(*) FROM {t}")
    totm = q1(f"SELECT count(*) FROM {t} WHERE end_time IS NULL")
    et[t]["_all"] = {"n": tot, "end_time_null": totm, "end_time_null_pct": round(100 * totm / tot, 4)}
# Same check on the analysis base: resolved GPU tasks
rows = q("SELECT status, count(*), count(*) FILTER (WHERE end_time IS NULL) FROM pai_task_table "
         "WHERE status IN ('Failed','Terminated') AND plan_gpu > 0 GROUP BY 1")
et["resolved_gpu_tasks"] = {s: {"n": n, "end_time_null": m, "end_time_null_pct": round(100 * m / n, 4)}
                            for s, n, m in rows}
missing_total = sum(v["end_time_null"] for v in et["resolved_gpu_tasks"].values())
et["resolved_gpu_tasks"]["_share_of_missing_that_is_Failed"] = (
    round(100 * et["resolved_gpu_tasks"].get("Failed", {}).get("end_time_null", 0) / missing_total, 4)
    if missing_total else None)
out["end_time_missing"] = et

# ---------------------------------------------------------------- label cross-checks
# Task status vs status of its instances (does a Failed task have a Failed instance?)
con.execute("""
CREATE OR REPLACE TEMP TABLE inst_agg AS
SELECT job_name, task_name,
       count(*) AS n_inst,
       count(*) FILTER (WHERE status = 'Failed') AS n_inst_failed,
       count(*) FILTER (WHERE status = 'Terminated') AS n_inst_term
FROM pai_instance_table GROUP BY 1, 2
""")
rows = q("""
SELECT t.status,
       count(*) AS n,
       count(*) FILTER (WHERE a.n_inst IS NULL) AS no_instances,
       count(*) FILTER (WHERE a.n_inst_failed > 0) AS any_inst_failed,
       count(*) FILTER (WHERE a.n_inst IS NOT NULL AND a.n_inst_failed = 0) AS no_inst_failed
FROM pai_task_table t LEFT JOIN inst_agg a USING (job_name, task_name)
WHERE t.status IN ('Failed','Terminated') GROUP BY 1
""")
out["task_vs_instance_status"] = {s: {"n": n, "no_instances": ni, "any_instance_failed": af,
                                      "no_instance_failed": nf} for s, n, ni, af, nf in rows}
# Job status vs task statuses for single-task jobs
rows = q("""
WITH one AS (SELECT job_name FROM pai_task_table GROUP BY 1 HAVING count(*) = 1)
SELECT j.status AS job_status, t.status AS task_status, count(*)
FROM pai_job_table j JOIN pai_task_table t USING (job_name) JOIN one USING (job_name)
GROUP BY 1, 2 ORDER BY 3 DESC
""")
out["single_task_job_status_vs_task_status"] = [
    {"job_status": a, "task_status": b, "n": c} for a, b, c in rows]

# ---------------------------------------------------------------- funnel
con.execute("""
CREATE OR REPLACE TEMP TABLE first_inst AS
SELECT job_name, task_name, worker_name, machine FROM (
  SELECT job_name, task_name, worker_name, machine,
         row_number() OVER (PARTITION BY job_name, task_name
                            ORDER BY start_time NULLS LAST, worker_name) AS rn
  FROM pai_instance_table) WHERE rn = 1
""")
con.execute("CREATE OR REPLACE TEMP TABLE sensor_w AS SELECT DISTINCT worker_name FROM pai_sensor_table")
con.execute("CREATE OR REPLACE TEMP TABLE metric_w AS SELECT DISTINCT worker_name FROM pai_machine_metric")
con.execute("""
CREATE OR REPLACE TEMP TABLE f AS
SELECT t.*, j.user,
       fi.worker_name AS fi_worker, fi.machine AS fi_machine,
       (ms.machine IS NOT NULL) AS in_spec,
       (sw.worker_name IS NOT NULL) AS has_sensor,
       (mw.worker_name IS NOT NULL) AS has_metric
FROM pai_task_table t
LEFT JOIN pai_job_table j USING (job_name)
LEFT JOIN first_inst fi USING (job_name, task_name)
LEFT JOIN pai_machine_spec ms ON ms.machine = fi.machine
LEFT JOIN sensor_w sw ON sw.worker_name = fi.worker_name
LEFT JOIN metric_w mw ON mw.worker_name = fi.worker_name
""")
stages = [
    ("S0 all task rows", "TRUE"),
    ("S1 job found (user known)", "user IS NOT NULL"),
    ("S2 resolved: status Failed or Terminated", "user IS NOT NULL AND status IN ('Failed','Terminated')"),
    ("S3 GPU requested: plan_gpu > 0", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0"),
    ("S4 has >=1 instance row", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0 AND fi_worker IS NOT NULL"),
    ("S5 first instance has a machine", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0 AND fi_machine IS NOT NULL"),
    ("S6 machine in pai_machine_spec", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0 AND fi_machine IS NOT NULL AND in_spec"),
    ("S7 first instance has sensor rows", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0 AND fi_machine IS NOT NULL AND in_spec AND has_sensor"),
    ("S8 first instance has machine-metric rows", "user IS NOT NULL AND status IN ('Failed','Terminated') AND plan_gpu > 0 AND fi_machine IS NOT NULL AND in_spec AND has_sensor AND has_metric"),
]
funnel = []
for name, cond in stages:
    n, nf, nu = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE status = 'Failed'), count(DISTINCT user) FROM f WHERE {cond}"
    ).fetchone()
    funnel.append({"stage": name, "condition_sql": cond, "tasks": n, "failed": nf,
                   "failed_pct": round(100 * nf / n, 4) if n else None, "distinct_users": nu})
base = funnel[3]["tasks"]
for row in funnel:
    row["pct_of_S3"] = round(100 * row["tasks"] / base, 4) if base else None

# Concentration and GPU-type mix on S3 (resolved GPU tasks)
S3 = stages[3][1]
users = q(f"SELECT user, count(*) AS n, count(*) FILTER (WHERE status='Failed') AS nf FROM f WHERE {S3} "
          f"GROUP BY 1 ORDER BY n DESC")
tot_n = sum(u[1] for u in users)
tot_f = sum(u[2] for u in users)
top5_n = sum(u[1] for u in users[:5])
by_fail = sorted(users, key=lambda u: -u[2])
conc = {
    "users": len(users),
    "top_user_by_tasks": {"task_share_pct": round(100 * users[0][1] / tot_n, 4),
                          "failure_share_pct": round(100 * users[0][2] / tot_f, 4)},
    "top5_users_by_tasks_task_share_pct": round(100 * top5_n / tot_n, 4),
    "top_user_by_failures_failure_share_pct": round(100 * by_fail[0][2] / tot_f, 4),
    "top_user_by_tasks_is_top_by_failures": users[0][0] == by_fail[0][0],
}
gpu_mix = {g if g is not None else "<NULL>": c for g, c in
           q(f"SELECT gpu_type, count(*) FROM f WHERE {S3} GROUP BY 1 ORDER BY 2 DESC")}

# Telemetry coverage by label on S6 (each table separately, not chained),
# to see whether a telemetry join selects on the label.
S6 = stages[6][1]
cov = {}
for flag in ["has_sensor", "has_metric"]:
    rows = q(f"SELECT status, count(*), count(*) FILTER (WHERE {flag}) FROM f WHERE {S6} GROUP BY 1")
    cov[flag] = {s: {"n": n, "with": w, "with_pct": round(100 * w / n, 4)} for s, n, w in rows}
out["telemetry_coverage_by_status_on_S6"] = cov
out["machine_metric_duplicate_worker_rows"] = q1(
    "SELECT count(*) - count(DISTINCT worker_name) FROM pai_machine_metric")
# Instance-level: are machine_metric rows present for Failed instances at all?
rows = q("""SELECT i.status, count(*), count(*) FILTER (WHERE mw.worker_name IS NOT NULL)
            FROM pai_instance_table i LEFT JOIN metric_w mw USING (worker_name)
            WHERE i.worker_name IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""")
out["instance_metric_coverage_by_status"] = {s: {"n": n, "with_metric": w,
                                                 "with_pct": round(100 * w / n, 4)} for s, n, w in rows}

funnel_out = {
    "generated_utc": out["generated_utc"],
    "definition": "Task-level funnel. Join keys: task->job on job_name (gives user); "
                  "task->instance on (job_name, task_name); first instance = earliest start_time, "
                  "ties by worker_name; instance->machine_spec on machine; "
                  "instance->sensor and ->machine_metric on worker_name.",
    "label": "Failed = 1, Terminated = 0 (trace README: only 'Terminated' tasks are successful). "
             "Running and Waiting are unresolved and excluded from S2 on.",
    "stages": funnel,
    "kanvas_tasks_for_scale": KANVAS_TASKS,
    "kanvas_pct_of_S3": round(100 * KANVAS_TASKS / base, 4) if base else None,
    "kanvas_note": "KANVAS's own filter is not reproduced here; the comparison is scale only.",
    "user_concentration_on_S3": conc,
    "gpu_type_mix_on_S3": gpu_mix,
    "gpu_type_MISC_pct_on_S3": round(100 * gpu_mix.get("MISC", 0) / base, 4) if base else None,
}

out["runtime_seconds"] = round(time.time() - t0, 1)
(R1 / "audit_results.json").write_text(json.dumps(out, indent=2, default=str))
(R1 / "funnel.json").write_text(json.dumps(funnel_out, indent=2, default=str))
print(json.dumps(funnel_out, indent=2, default=str))
print(f"done in {time.time()-t0:.0f}s")
