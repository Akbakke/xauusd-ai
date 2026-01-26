# Dummy Context (Benchmark Only)

**Single Source of Truth for dummy context semantics and usage.**

## What is Dummy Context?

Dummy context consists of zero-filled tensors:
- `ctx_cat`: `[0, 0, 0, 0, 0]` (int64, shape [5])
- `ctx_cont`: `[0.0, 0.0]` (float32, shape [2])

These tensors satisfy the shape contract required by the model but contain no meaningful information.

## What is Dummy Context Used For?

**DataLoader / IO throughput benchmarking only.**

When benchmarking DataLoader performance (e.g., `--benchmark_workers`), we need to measure pure data loading speed without the overhead of context feature building. Dummy ctx allows us to:
1. Satisfy the model's input shape contract
2. Measure DataLoader throughput without context-building overhead
3. Compare different `num_workers` settings accurately

## When is Dummy Context Allowed?

Dummy ctx is allowed **ONLY** when **ALL** of the following conditions are met:

1. **`mode == "benchmark"`** (dataset is in benchmark mode)
2. **`allow_dummy_ctx == True`** (explicitly enabled via CLI flag)
3. **`--benchmark_only` or `--benchmark_workers` is set** (benchmark mode is active)

All three conditions must be true. If any condition is false, dummy ctx is forbidden.

## When is Dummy Context Forbidden?

Dummy ctx is **FORBIDDEN** in:

- **Training** (`mode == "train"`)
- **Replay** (runtime replay evaluation)
- **Production** (live trading)
- **Evaluation** (model evaluation / validation)
- **Any non-benchmark context**

**Dummy ctx must never be used for training, evaluation, or live trading.**

## Why Does Dummy Context Exist?

1. **Shape Contract**: The model requires `ctx_cat` and `ctx_cont` tensors with specific shapes. Dummy ctx satisfies this contract without requiring context feature building.

2. **Throughput Measurement**: Benchmarking DataLoader performance requires isolating IO/loading overhead from feature-building overhead. Dummy ctx allows pure IO benchmarking.

3. **Prebuilt Dataset Support**: If a prebuilt dataset lacks `ctx_cat`/`ctx_cont` columns, dummy ctx allows benchmark to proceed (with explicit opt-in).

## Implementation Guards

The implementation enforces dummy ctx restrictions through:

1. **Runtime Assert**: `allow_dummy_ctx == True` + `mode != "benchmark"` → hard fail
2. **CLI Validation**: `--benchmark_allow_dummy_ctx` can only be used with `--benchmark_only` or `--benchmark_workers`
3. **Code Comments**: Explicit warnings in code where dummy ctx is created
4. **Unit Tests**: Tests verify that dummy ctx fails in non-benchmark contexts

## Contract

- **Dummy ctx is benchmark-only**
- **Dummy ctx is never used for model training or inference**
- **Dummy ctx is never used in production or evaluation**
- **Dummy ctx requires explicit opt-in via `--benchmark_allow_dummy_ctx`**
- **Dummy ctx is guarded by multiple runtime checks**

No exceptions. No silent fallbacks. No misuse.
