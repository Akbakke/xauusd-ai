# Makefile for GX1 XAUUSD Project

.PHONY: help optuna-entry optuna-entry-refine farm-status farm-exit-baseline farm-exit-policy-analysis farm-doc farm-entry-meta-baseline farm-entry-policy-sim farm-v2-hygiene-run farm-v2-hygiene-check farm-v2-hygiene-run farm-v2-hygiene-check

help:
	@echo "Available targets:"
	@echo "  make optuna-entry         - Run Optuna sweep for entry gates (120 trials, 16 jobs AGGRESSIVE)"
	@echo "  make optuna-entry-refine  - Refine Optuna sweep (240 trials, 20 jobs MAX)"
	@echo "  make farm-status          - Print FARM baseline status and configuration"
	@echo "  make farm-exit-baseline    - Train baseline AI exit model on FARM dataset"
	@echo "  make farm-exit-policy-analysis - Full AI exit overlay policy analysis"
	@echo "  make farm-doc                 - View FARM exit final documentation"
	@echo "  make farm-entry-meta-baseline - Train FARM entry meta-model baseline"
	@echo "  make farm-entry-policy-sim    - Simulate FARM entry policy grid"
	@echo "  make farm-v2-hygiene-run     - Run FARM_V2 hygiene replay (3-6 months)"
	@echo "  make farm-v2-hygiene-check   - Check FARM_V2 hygiene results"
	@echo "  make farm-v2-hygiene-run     - Run FARM_V2 hygiene replay (3-6 months)"
	@echo "  make farm-v2-hygiene-check   - Check FARM_V2 hygiene results"

# Optuna entry gates optimization
# AGGRESSIVE: Set thread limits to 1 per worker, use 2x CPU count for parallelization
optuna-entry:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 GX1_XGB_THREADS=1 XGB_N_JOBS=1 python -m gx1.tuning.optuna_entry_gates --phase-a-only --trials-phase-a 60 --jobs 6

optuna-entry-refine:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 GX1_XGB_THREADS=1 XGB_N_JOBS=1 python -m gx1.tuning.optuna_entry_gates --trials 240 --jobs 20

# FARM baseline status
farm-status:
	python3 -m gx1.scripts.farm_status

# FARM AI exit baseline training
farm-exit-baseline:
	python3 -m gx1.models.exit_tabular_baseline \
		--dataset gx1/wf_runs/FARM_EXIT_DATA/farm_exit_paths.parquet \
		--test-ratio 0.2 \
		--out-json gx1/wf_runs/FARM_EXIT_AI/baseline_metrics.json \
		--out-dir gx1/wf_runs/FARM_EXIT_AI/

# FARM AI exit overlay policy analysis
farm-exit-policy-analysis:
	python3 -m gx1.analysis.farm_exit_policy_analysis \
		--dataset gx1/wf_runs/FARM_EXIT_DATA/farm_exit_paths.parquet \
		--thresholds 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
		--min-bars 1 \
		--out-dir gx1/wf_runs/FARM_EXIT_AI/

# View FARM exit final documentation
farm-doc:
	@cat gx1/docs/FARM_EXIT_V1_FINAL.md

# FARM entry meta-model baseline training
farm-entry-meta-baseline:
	python3 -m gx1.models.farm_entry_meta_baseline \
		--dataset gx1/wf_runs/FARM_ENTRY_DATA/farm_entry_dataset_v1.parquet \
		--test-ratio 0.2 \
		--out-json gx1/wf_runs/FARM_ENTRY_AI/entry_meta_baseline_metrics.json

# FARM entry policy simulator
farm-entry-policy-sim:
	python3 -m gx1.analysis.farm_entry_policy_sim \
		--dataset gx1/wf_runs/FARM_ENTRY_DATA/farm_entry_dataset_v1.parquet \
		--p-long-grid 0.75,0.8,0.85 \
		--p-profitable-grid 0.5,0.55,0.6,0.65,0.7 \
		--out-csv gx1/wf_runs/FARM_ENTRY_AI/entry_policy_grid.csv \
		--out-json gx1/wf_runs/FARM_ENTRY_AI/entry_policy_grid.json \
		--out-report gx1/wf_runs/FARM_ENTRY_AI/FARM_ENTRY_POLICY_V2_CANDIDATES.md

# FARM_V2 hygiene run (3-6 months, moderate CPU)
farm-v2-hygiene-run:
	bash scripts/run_farm_v2_hygiene.sh

# FARM_V2 hygiene check (analyze results)
farm-v2-hygiene-check:
	python3 -m gx1.analysis.farm_v2_hygiene_check \
		--trade-log gx1/wf_runs/FARM_V2_HYGIENE/trade_log.csv \
		--output-dir gx1/wf_runs/FARM_V2_HYGIENE
