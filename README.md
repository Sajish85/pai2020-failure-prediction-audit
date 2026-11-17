# One User Dominates: An Audit and User-Grouped Re-check of Task-Failure Prediction on the Alibaba PAI GPU Trace (cluster-trace-gpu-v2020)

Sajish Krishnan, independent researcher. Paper: [paper.pdf](paper.pdf) ([Markdown](paper.md)).

## Summary

The Alibaba PAI 2020 GPU trace is used to train and evaluate task-failure predictors. One user submits 28.71% of its 985,358 resolved GPU tasks and 49.21% of their failures, and some columns (task end time, a task's own machine-metric rows) nearly copy the outcome. Predicting failure from the resource request alone, gradient boosting has a mean AUC of 0.7950 on random splits and 0.5452 with whole users held out; on a time-ordered split it scores 0.2811, which traces to one request shape from the largest user. The paper also re-checks two results from an earlier report ("Paper 1", [machine-class-is-the-message](https://github.com/Sajish85/machine-class-is-the-message)) on the full trace, with and without that user.

## What is here

| Path | What |
|---|---|
| `paper.pdf`, `paper.md` | the paper |
| `paigpu/` | module that builds the task frame and the splits (`load_trace()`, `user_folds(k, seed)`, `time_split(frac)`) |
| `scripts/` | the analysis scripts |
| `Makefile` | the pipeline |
| `results/` | aggregate outputs (JSON, CSV) and their SHA-256 checksum files |
| `audit_results.json`, `funnel.json` | the audit of the raw tables |
| `figures/` | the paper's figures |
| `leakage-table.md` | which trace columns leak the label |
| `requirements.txt`, `requirements.lock.txt` | pinned environment (Python 3.14.3) |

## Reproduce

```
python3 -m venv .venv
make env
make reproduce
make revision
make supplementary
```

`make reproduce` downloads the seven trace tables if they are missing, verifies them, builds the frame under `data/` (git-ignored), runs the Paper 1 re-checks and the evaluation grid, draws the figures and writes checksums of the result files. `make revision` and `make supplementary` produce the remaining result files. Compare your checksums with `results/SHA256SUMS*`.

Two inputs are not in this repo, because nothing from the trace distribution is redistributed here. Put them in place before `make reproduce`:

- `data/SHA256SUMS.tar.README` and `data/SHA256SUMS.csv.README`: the SHA-256 lists for the seven `.tar.gz` files and the seven `.csv` files, copied from the trace README, in `shasum -c` format.
- `data/raw/<table>.header` for each of the seven tables: the column-name files from `github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data`.

`make factcheck` was run on the paper under its working file name and does not run against `paper.md` as shipped.

## Data

The trace is Alibaba's cluster-trace-gpu-v2020, available from Alibaba: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020. This repo redistributes no trace rows. `results/` and the two audit files hold aggregates only.

## Licence

- Code (`paigpu/`, `scripts/`, `Makefile`): MIT, see [LICENSE](LICENSE).
- Paper (`paper.md`, `paper.pdf`) and aggregates (`results/`, `figures/`, `audit_results.json`, `funnel.json`, `leakage-table.md`): CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/.
