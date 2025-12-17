# Baseline Lineage Report

**Generated:** 2025-12-16T20:39:20.535099+00:00

## Executive Summary

- **Total Unique Bundles Found:** 112
- **Bundles with role=PROD_BASELINE:** 14

- **Recommended One True Baseline:** `b692c56822495c76`
  - Source: prod_snapshot
  - Policy Path: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
  - Score: 150

## Top 3 Candidates

| Bundle ID | Source | Role | Policy Hash | Router | Entry Model | Router Hash | Manifest Hash | Last Run | Mode | Trades/Day |
|-----------|--------|------|-------------|--------|-------------|-------------|---------------|----------|------|------------|
| `b692c56822495c76` | prod_snapshot | PROD_BASELINE | `a560cfcd6683...` | HYBRID_ROUTER_V3 (cutoff=1.0) | N/A | `N/A...` | `N/A...` | None | None | N/A |
| `c69fa71ec925636b` | wf_run | PROD_BASELINE | `a2b67a9860a0...` | HYBRID_ROUTER_V3 | N/A | `N/A...` | `N/A...` | FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR | None | 0.56 |
| `1ac0f9b532a4d650` | wf_run | PROD_BASELINE | `2259b9a3b63c...` | HYBRID_ROUTER_V3 | N/A | `N/A...` | `N/A...` | GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR | None | 1.03 |

## Conflicts

Multiple bundles claim PROD_BASELINE but have different fingerprints:

### Conflict: `b692c56822495c76` vs `c6b9862957a38032`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `c69fa71ec925636b`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (1.0 vs None)

### Conflict: `b692c56822495c76` vs `1ac0f9b532a4d650`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (1.0 vs None)

### Conflict: `b692c56822495c76` vs `7b0d36cf2277b38c`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `4e11d9f08fb779c4`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `70646b9f43e65d94`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `17ff85e103f39808`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `916fcd35c0d11861`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `b692c56822495c76` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `c69fa71ec925636b`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (1.0 vs None)

### Conflict: `c6b9862957a38032` vs `1ac0f9b532a4d650`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (1.0 vs None)

### Conflict: `c6b9862957a38032` vs `7b0d36cf2277b38c`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `4e11d9f08fb779c4`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `70646b9f43e65d94`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `17ff85e103f39808`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `916fcd35c0d11861`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c6b9862957a38032` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `c69fa71ec925636b` vs `1ac0f9b532a4d650`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `c69fa71ec925636b` vs `7b0d36cf2277b38c`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `4e11d9f08fb779c4`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `70646b9f43e65d94`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `c69fa71ec925636b` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `7b0d36cf2277b38c`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `4e11d9f08fb779c4`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `70646b9f43e65d94`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `1ac0f9b532a4d650` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective, guardrail_cutoff (None vs 1.0)

### Conflict: `7b0d36cf2277b38c` vs `4e11d9f08fb779c4`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `70646b9f43e65d94`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `7b0d36cf2277b38c` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `70646b9f43e65d94`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `4e11d9f08fb779c4` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `3b71503c1a7ade56`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `70646b9f43e65d94` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_144157/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `58a7b7205fcf8c4a`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `3b71503c1a7ade56` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_145122/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `58a7b7205fcf8c4a` vs `8d927c4b02edddd9`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `58a7b7205fcf8c4a` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `58a7b7205fcf8c4a` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `58a7b7205fcf8c4a` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `58a7b7205fcf8c4a` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS/parallel_chunks/policy_chunk_5.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `8d927c4b02edddd9` vs `92cbb7e982ae3de9`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `8d927c4b02edddd9` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `8d927c4b02edddd9` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `8d927c4b02edddd9` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `92cbb7e982ae3de9` vs `17ff85e103f39808`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `92cbb7e982ae3de9` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `92cbb7e982ae3de9` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211622/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `17ff85e103f39808` vs `916fcd35c0d11861`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `17ff85e103f39808` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211653/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

### Conflict: `916fcd35c0d11861` vs `0c1cc17f4b14934e`
- Bundle 1: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251216_120044/parallel_chunks/policy_chunk_0.yaml`
- Bundle 2: `gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211546/parallel_chunks/policy_chunk_0.yaml`
- Differences: policy_effective

## Run Lineage

### Bundle `c69fa71ec925636b` (wf_run)

| Run Tag | Mode | Period | N Workers | Trades/Day |
|---------|------|--------|-----------|------------|
| `FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR` | None | None | None | 0.56 |

### Bundle `1ac0f9b532a4d650` (wf_run)

| Run Tag | Mode | Period | N Workers | Trades/Day |
|---------|------|--------|-----------|------------|
| `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR` | None | None | None | 1.03 |

### Bundle `7b0d36cf2277b38c` (wf_run)

| Run Tag | Mode | Period | N Workers | Trades/Day |
|---------|------|--------|-----------|------------|
| `FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_GUARDRAIL_FULLYEAR` | None | None | None | 0.56 |

## Action Plan

### Freeze Baseline

Recommended baseline: `b692c56822495c76`

```yaml
Policy Path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml
Policy Hash: a560cfcd668325e30adbe594ef7af029d5153702c872f71bae73f036087f3cb5
Router Model Hash: None
Feature Manifest Hash: None
```

### Fix Parity

To achieve parity between FULLYEAR and CANARY:

1. Use the same policy bundle for both runs
2. Ensure same entry/exit configs
3. Use same router model and feature manifest
4. Run both with `n_workers=1` for determinism

### Deprecate

Consider archiving or marking as obsolete:
- Policies not in prod_snapshot
- Bundles with score < 50
- Bundles with no runs

---

*Report generated by `gx1/analysis/baseline_lineage_scan.py`*