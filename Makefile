# R1: Alibaba PAI GPU 2020 raw-trace re-check. Run from r1/.
#
#   make env                     install pinned deps into .venv, write requirements.lock.txt
#   make reproduce               download (if missing) -> verify -> frame -> re-checks -> grid -> figures -> checksums
#   make reproduce SLICE=0.02    same on a deterministic 2% user slice (outputs in *-slice/ dirs)
#   make rerun-check             clean-room re-run #1: fresh venv + fresh shell, diff result checksums
#   make revision                time-split diagnostics + user-identity ablation + their figure (needs make reproduce first)
#   make factcheck               trace every number in the current draft and leakage-table.md to an output file
#                                (DRAFT=PAPER-DRAFT-v2.md to check an older one; default is the newest PAPER-DRAFT-v2*.md)
#   make supplementary           counts quoted in the report that no other result file holds
#
# Trace rows stay under data/ (git-ignored, DO-NOT-PUBLISH). results/ and figures/ hold aggregates only.

SHELL   := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY      ?= .venv/bin/python
P1REF   ?= ../paper1-ref
SLICE   ?=
TABLES  := pai_job_table pai_task_table pai_instance_table pai_sensor_table pai_group_tag_table pai_machine_spec pai_machine_metric
URL     := https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces

ifeq ($(strip $(SLICE)),)
  SFX    :=
  SLICEA :=
else
  SFX    := -slice
  SLICEA := --slice $(SLICE)
endif
FRAME_DIR := data/parquet$(SFX)
RES_DIR   := results$(SFX)
FIG_DIR   := figures$(SFX)
FRAME     := $(FRAME_DIR)/tasks_frame.parquet
COMPARED  := $(RES_DIR)/paper1_recheck.json $(RES_DIR)/eval_grid.json $(RES_DIR)/eval_grid.csv

.PHONY: env download verify frame recheck grid figures checksums reproduce rerun-check clean-results revision factcheck supplementary

env:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip freeze --all > requirements.lock.txt

# Tarballs from the URLs in the trace README (data/pai2020_readme.md l.78-84).
# The .header files come from the README's alternative mirror; they are not fetched here
# (exact raw path not verified); download fails loudly if one is missing.
download:
	@mkdir -p data/raw data/csv
	@for t in $(TABLES); do \
	  test -f data/csv/$$t.csv && continue; \
	  test -f data/raw/$$t.tar.gz || curl -fL --retry 3 -o data/raw/$$t.tar.gz $(URL)/$$t.tar.gz || exit 1; \
	  (cd data/raw && grep " $$t.tar.gz$$" ../SHA256SUMS.tar.README | shasum -a 256 -c -) || exit 1; \
	  tar -xzf data/raw/$$t.tar.gz -C data/csv || exit 1; \
	done
	@for t in $(TABLES); do test -f data/raw/$$t.header || { echo "missing data/raw/$$t.header (get it from github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data)"; exit 1; }; done

verify: download
	cd data/csv && shasum -a 256 -c ../SHA256SUMS.csv.README

frame: verify
	$(PY) -m paigpu.convert $(SLICEA) --out-dir $(FRAME_DIR) --results-dir $(RES_DIR)

recheck: frame
	$(PY) scripts/recheck.py --frame $(FRAME) --out $(RES_DIR)/paper1_recheck.json $(if $(wildcard $(P1REF)/scripts/eda_results.json),--paper1-ref $(P1REF))

grid: frame
	$(PY) scripts/grid.py --frame $(FRAME) --out-dir $(RES_DIR)

figures: recheck grid
	$(PY) scripts/figures.py --results $(RES_DIR) --out $(FIG_DIR)

checksums: recheck grid
	shasum -a 256 $(COMPARED) > $(RES_DIR)/SHA256SUMS

reproduce: figures checksums
	@echo "reproduce done: $(RES_DIR)/ $(FIG_DIR)/"

# Clean-room re-run #1 (brief step 7). Keeps the first run's checksums, rebuilds a fresh venv
# from requirements.lock.txt, re-runs `make reproduce` under `env -i` (fresh shell, no inherited
# env), then diffs checksums. Never edits outputs. Log: REPRO-1.log
rerun-check:
	test -f results/SHA256SUMS || { echo "run make reproduce first"; exit 1; }
	cp results/SHA256SUMS results/SHA256SUMS.run1
	rm -rf .venv-clean
	$(PY) -m venv .venv-clean
	.venv-clean/bin/python -m pip install -r requirements.lock.txt
	rm -f results/paper1_recheck.json results/eval_grid.json results/eval_grid.csv results/SHA256SUMS
	env -i HOME="$$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin /bin/bash -c 'make reproduce PY=.venv-clean/bin/python' 2>&1 | tee REPRO-1.log
	{ echo "== run 1 vs clean-room run =="; diff results/SHA256SUMS.run1 results/SHA256SUMS && echo "ALL CHECKSUMS MATCH"; } 2>&1 | tee -a REPRO-1.log

# Revision analyses (17 Nov 2026). Read results/eval_grid.{json,csv} to confirm their baseline cells
# equal the grid's. Checksums go to a separate file so `rerun-check` is unchanged.
REVISED := $(RES_DIR)/time_split_diagnostics.json $(RES_DIR)/time_split_by_request_shape.csv $(RES_DIR)/userid_ablation.json $(RES_DIR)/userid_ablation.csv
revision:
	test -f $(RES_DIR)/eval_grid.json || { echo "run make reproduce first"; exit 1; }
	$(PY) scripts/time_split_diag.py --frame $(FRAME) --out-dir $(RES_DIR)
	$(PY) scripts/userid_ablation.py --frame $(FRAME) --out-dir $(RES_DIR)
	$(PY) scripts/figures_revision.py --results $(RES_DIR) --out $(FIG_DIR)
	shasum -a 256 $(REVISED) > $(RES_DIR)/SHA256SUMS.revision

# The draft under check is a parameter: `make factcheck DRAFT=PAPER-DRAFT-v2.1.md`.
# Default: the highest-numbered PAPER-DRAFT-v2*.md present (version sort).
DRAFT ?= $(shell ls PAPER-DRAFT-v2*.md | sort -V | tail -1)
# Counts quoted in the report that no other result file holds (17 Nov 2026). Own checksum file,
# so `rerun-check` and SHA256SUMS.revision are unchanged.
supplementary:
	$(PY) scripts/supplementary_counts.py --frame $(FRAME) --out-dir $(RES_DIR)
	shasum -a 256 $(RES_DIR)/supplementary_counts.json > $(RES_DIR)/SHA256SUMS.supplementary

factcheck:
	$(PY) -I scripts/factcheck_draft.py $(DRAFT)
	$(PY) -I scripts/factcheck_draft.py leakage-table.md

clean-results:
	rm -rf results-slice figures-slice data/parquet-slice
