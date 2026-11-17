"""CSV -> Parquet conversion step for the Alibaba PAI GPU 2020 trace (R1, step 2).

Builds ONE task-level frame (every task row, S0) with the joins fixed by
r1/AUDIT.md and writes it as Parquet. Then it recounts every funnel stage from
the Parquet file and reconciles the counts against r1/funnel.json.

Joins (from AUDIT.md / funnel.json "definition"):
  task -> job               on job_name                 (gives user)
  task -> first instance    on (job_name, task_name)    (default)
                            or (inst_id, task_name)     (--alt-keys: job_name->inst_id via job table)
  first instance = earliest start_time, NULLS LAST, ties by worker_name
  first instance -> pai_machine_spec on machine
  first instance -> sensor / machine_metric on worker_name (presence flags only)

Columns deliberately NOT carried into the frame (label leaks, AUDIT findings 3-4):
  end_time / duration, sensor values, machine-metric values.
has_sensor / has_metric are kept as flags for funnel reconciliation only;
paigpu.REQUEST_FEATURES never includes them.

The Parquet output lives under r1/data/ (git-ignored, DO-NOT-PUBLISH): it holds
trace rows. Only results/frame_reconciliation.json (aggregates) is publishable.

Usage:
  python -m paigpu.convert [--slice FRAC] [--alt-keys] [--out-dir DIR] [--results-dir DIR]
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import duckdb

R1 = pathlib.Path(__file__).resolve().parents[1]
RAW = R1 / "data" / "raw"
CSV = R1 / "data" / "csv"
FRAME_NAME = "tasks_frame.parquet"

# Same stage conditions as r1/scripts/audit.py (copied, not imported, so the
# reconciliation is an independent recount over the Parquet file).
STAGES = [
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


def csv_rel(con, table):
    cols = (RAW / f"{table}.header").read_text().strip().split(",")
    names = "[" + ",".join(f"'{c}'" for c in cols) + "]"
    return (f"read_csv('{CSV / (table + '.csv')}', header=false, names={names}, "
            f"sample_size=-1)")


def slice_predicate(frac):
    """Deterministic user-level slice: keep users whose md5 bucket < frac."""
    if frac is None or frac >= 1:
        return "TRUE"
    cut = int(round(frac * 10000))
    return f"(hash_bucket < {cut})"


def build(out_dir, results_dir, frac=None, alt_keys=False):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()  # in-memory; nothing persisted except the Parquet file
    con.execute("SET threads TO 4")
    con.execute("SET preserve_insertion_order = false")  # final COPY has its own ORDER BY
    con.execute("SET memory_limit = '5GB'")
    con.execute(f"SET temp_directory = '{R1 / 'data' / 'duckdb_tmp'}'")

    for t in ["pai_job_table", "pai_task_table", "pai_machine_spec"]:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM {csv_rel(con, t)}")
        print(f"loaded {t} ({time.time()-t0:.0f}s)", flush=True)
    # instance table: only the columns the joins use (it is ~2 GB of CSV; keeps memory down)
    con.execute(f"CREATE TABLE pai_instance_table AS SELECT job_name, task_name, worker_name, inst_id, "
                f"machine, start_time FROM {csv_rel(con, 'pai_instance_table')}")
    print(f"loaded pai_instance_table ({time.time()-t0:.0f}s)", flush=True)
    con.execute(f"CREATE TABLE sensor_w AS SELECT DISTINCT worker_name FROM {csv_rel(con, 'pai_sensor_table')}")
    con.execute(f"CREATE TABLE metric_w AS SELECT DISTINCT worker_name FROM {csv_rel(con, 'pai_machine_metric')}")
    print(f"loaded sensor/metric worker lists ({time.time()-t0:.0f}s)", flush=True)

    if alt_keys:
        # Alternative keys named in AUDIT.md: job_name <-> inst_id is 1:1.
        inst_join = """
        CREATE TABLE inst_k AS
        SELECT j.job_name, i.task_name, i.worker_name, i.machine, i.start_time
        FROM pai_instance_table i JOIN pai_job_table j ON j.inst_id = i.inst_id"""
    else:
        inst_join = """
        CREATE TABLE inst_k AS
        SELECT job_name, task_name, worker_name, machine, start_time FROM pai_instance_table"""
    con.execute(inst_join)
    con.execute("""
    CREATE TABLE first_inst AS
    SELECT job_name, task_name, worker_name, machine, start_time FROM (
      SELECT *, row_number() OVER (PARTITION BY job_name, task_name
                                   ORDER BY start_time NULLS LAST, worker_name) AS rn
      FROM inst_k) WHERE rn = 1""")

    # ------------------------------------------------- KANVAS-comparable flag
    # KANVAS's filter, read from github.com/Swapnil007-Curious/KANVAS kanvas/data.py
    # (build_kanvas_dataset / _lookback_average), applied to ALL labeled tasks instead of
    # KANVAS's stratified random sample: status Failed/Terminated; first instance has a
    # machine and a start time t0; >=1 pai_machine_metric row on that machine whose
    # end_time is in [t0-3600, t0) and whose worker is not one of the task's own job's
    # workers; the 5 window means and plan_cpu/plan_mem/plan_gpu/inst_num/cap_gpu not null.
    # Not reproduced: the stratified sample, the 1st/99th-percentile clipping, and KANVAS's
    # first-instance tie-break by file order (here: by worker_name, as in funnel.json).
    con.execute(f"""
    CREATE TABLE mm AS
    SELECT worker_name, machine, end_time, machine_cpu_usr, machine_gpu, machine_cpu_kernel,
           machine_load_1, machine_net_receive, floor(end_time / 3600)::BIGINT AS hb
    FROM {csv_rel(con, 'pai_machine_metric')}""")
    con.execute("""
    CREATE TABLE kt AS
    SELECT fi.job_name, fi.task_name, fi.machine, fi.start_time AS t0,
           unnest([floor(fi.start_time / 3600)::BIGINT - 1, floor(fi.start_time / 3600)::BIGINT]) AS hb
    FROM first_inst fi JOIN pai_task_table t USING (job_name, task_name)
    WHERE t.status IN ('Failed','Terminated') AND fi.machine IS NOT NULL AND fi.start_time IS NOT NULL""")
    con.execute("CREATE TABLE job_workers AS SELECT DISTINCT job_name, worker_name FROM inst_k WHERE worker_name IS NOT NULL")
    con.execute("DROP TABLE inst_k"); con.execute("DROP TABLE pai_instance_table")  # free memory
    # Window join in N_PART partitions by machine: bounds the join's working set (the full join
    # ran out of memory/spill on a nearly full disk, 17 Nov). Partitioning does not change results.
    N_PART = 8
    con.execute("CREATE TABLE kwin (job_name VARCHAR, task_name VARCHAR, kanvas_window_rows BIGINT, "
                "a1 DOUBLE, a2 DOUBLE, a3 DOUBLE, a4 DOUBLE, a5 DOUBLE)")
    for part in range(N_PART):
        con.execute(f"""
    INSERT INTO kwin
    SELECT kt.job_name, kt.task_name, count(*) AS kanvas_window_rows,
           avg(mm.machine_cpu_usr) AS a1, avg(mm.machine_gpu) AS a2, avg(mm.machine_cpu_kernel) AS a3,
           avg(mm.machine_load_1) AS a4, avg(mm.machine_net_receive) AS a5
    FROM kt
    JOIN mm ON mm.machine = kt.machine AND mm.hb = kt.hb
    LEFT JOIN job_workers jw ON jw.job_name = kt.job_name AND jw.worker_name = mm.worker_name
    WHERE mm.end_time >= kt.t0 - 3600 AND mm.end_time < kt.t0 AND jw.worker_name IS NULL
      AND hash(kt.machine) % {N_PART} = {part}
    GROUP BY kt.job_name, kt.task_name""")
    print(f"built KANVAS lookback windows ({time.time()-t0:.0f}s)", flush=True)

    con.execute(f"""
    CREATE TABLE frame AS
    SELECT
      t.job_name, t.task_name, j.user,
      t.status,
      CASE t.status WHEN 'Failed' THEN 1 WHEN 'Terminated' THEN 0 END::TINYINT AS label,
      t.start_time,
      t.inst_num, t.plan_cpu, t.plan_mem, t.plan_gpu, t.gpu_type,
      fi.worker_name AS fi_worker, fi.machine AS fi_machine, fi.start_time AS fi_start_time,
      ms.gpu_type AS spec_gpu_type, ms.cap_cpu, ms.cap_mem, ms.cap_gpu,
      (ms.machine IS NOT NULL) AS in_spec,
      (sw.worker_name IS NOT NULL) AS has_sensor,
      (mw.worker_name IS NOT NULL) AS has_metric,
      coalesce(j.user IS NOT NULL AND t.status IN ('Failed','Terminated') AND t.plan_gpu > 0, false) AS in_s3,
      coalesce(kw.kanvas_window_rows, 0)::INTEGER AS kanvas_window_rows,
      coalesce(t.status IN ('Failed','Terminated') AND kw.kanvas_window_rows > 0
               AND kw.a1 IS NOT NULL AND kw.a2 IS NOT NULL AND kw.a3 IS NOT NULL
               AND kw.a4 IS NOT NULL AND kw.a5 IS NOT NULL
               AND t.plan_cpu IS NOT NULL AND t.plan_mem IS NOT NULL AND t.plan_gpu IS NOT NULL
               AND t.inst_num IS NOT NULL AND ms.cap_gpu IS NOT NULL, false) AS kanvas_cmp,
      (md5_number(coalesce(j.user, '')) % 10000)::INTEGER AS hash_bucket
    FROM pai_task_table t
    LEFT JOIN pai_job_table j USING (job_name)
    LEFT JOIN first_inst fi USING (job_name, task_name)
    LEFT JOIN pai_machine_spec ms ON ms.machine = fi.machine
    LEFT JOIN sensor_w sw ON sw.worker_name = fi.worker_name
    LEFT JOIN metric_w mw ON mw.worker_name = fi.worker_name
    LEFT JOIN kwin kw ON kw.job_name = t.job_name AND kw.task_name = t.task_name
    """)
    pred = slice_predicate(frac)
    out = out_dir / FRAME_NAME
    con.execute(f"COPY (SELECT * EXCLUDE (hash_bucket) FROM frame WHERE {pred} "
                f"ORDER BY job_name, task_name) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------- reconciliation
    rows = []
    for name, cond in STAGES:
        n, nf, nu = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE status='Failed'), count(DISTINCT user) "
            f"FROM read_parquet('{out}') WHERE {cond}").fetchone()
        rows.append({"stage": name, "tasks": n, "failed": nf, "distinct_users": nu})
    is_full = (frac is None or frac >= 1)
    rec = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "duckdb_version": duckdb.__version__,
           "join_keys": "alt: instance on (inst_id->job_name, task_name)" if alt_keys
                        else "default: instance on (job_name, task_name)",
           "slice_fraction": frac if not is_full else 1.0,
           "frame_rows": con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0],
           "frame_columns": [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()],
           "stages_from_parquet": rows}
    kn, kf, ku = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE status='Failed'), count(DISTINCT user) "
        f"FROM read_parquet('{out}') WHERE kanvas_cmp").fetchone()
    rec["kanvas_comparable"] = {"tasks": kn, "failed": kf, "distinct_users": ku,
                                "definition": "KANVAS filter without its stratified sample; see paigpu/convert.py"}
    if is_full:
        ref = json.loads((R1 / "funnel.json").read_text())["stages"]
        diffs = []
        for mine, theirs in zip(rows, ref):
            for k in ["tasks", "failed", "distinct_users"]:
                if mine[k] != theirs[k]:
                    diffs.append({"stage": mine["stage"], "field": k,
                                  "parquet": mine[k], "funnel_json": theirs[k],
                                  "diff": mine[k] - theirs[k]})
        rec["reconciliation_vs_funnel_json"] = {"all_match": not diffs, "differences": diffs}
    else:
        rec["reconciliation_vs_funnel_json"] = "not applicable: slice run"
    (results_dir / "frame_reconciliation.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps(rec["reconciliation_vs_funnel_json"], indent=2))
    print(f"done ({time.time()-t0:.0f}s)")
    if is_full and not rec["reconciliation_vs_funnel_json"]["all_match"]:
        # Brief step 2: continue with the frame, flagging the gap (do not abort make).
        print("WARNING: FRAME DOES NOT RECONCILE WITH funnel.json; see frame_reconciliation.json. "
              "Retry with --alt-keys, then STOP-NOTE if still off.", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slice", type=float, default=None, help="keep this fraction of users (0-1)")
    ap.add_argument("--alt-keys", action="store_true", help="join instances via inst_id (AUDIT alt key)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=R1 / "data" / "parquet")
    ap.add_argument("--results-dir", type=pathlib.Path, default=R1 / "results")
    a = ap.parse_args(argv)
    return build(a.out_dir, a.results_dir, a.slice, a.alt_keys)


if __name__ == "__main__":
    sys.exit(main())
