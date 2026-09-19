# Local training efficiency review — 2026-09-08

The operator resumed the local efficiency objective on 2026-09-08: measure the
existing RTX 3090 under the signed 160 W guard before spending on cloud.
This review is preparation and analysis, not a launch recipe. Full training,
TEST, purchase, power-limit increases and canonical model changes are outside
this wave. The current executable handover still returns BLOCK.

## Evidence and first result

The CPU-only report command is now part of `entry_next_edge_control.sh` as
`model-native-training-efficiency-report`. It requires explicit SHA-256 inputs
for the recipe and both logs, an expected bundle commit, and a new output
folder. It reuses the bundle-commit verifier, checks recipe/source/run identity
in metadata and lock, and requires successful guard exit and publication of
that exact bundle. It never imports the trainer, loads model tensors, opens
TRAIN/VAL/TEST payloads, or launches GPU work. Publication uses the existing
atomic no-replace directory owner.

Real measurement report:
`/var/tmp/gx1-local-efficiency-20260908/historical_smoke_report/report.json`
and its generated `report.md`. Reproducible argument vector:
`/var/tmp/gx1-local-efficiency-20260908/report_command.json`.

These are measurements of the completed September 8 smoke from source
`efa99b2b3105d2fb44a041de404d5a66b41158f8`, not a fresh run of this worktree.
The report preserves the exact bundle commit and hashes of the supplied logs.
Those sidecar logs were not themselves included in the historical bundle
manifest; the report states this distinction.

The measured first-batch Exit computation is the largest timed phase, while
Exit episode materialization is a small fraction. This is one cold batch,
not a steady-state attribution or a CUDA kernel profile. The complete guarded
smoke spends most wall time outside its brief TRAIN loop; using the complete
launch duration as GPU training throughput would be misleading.

The report labels its coarse log-derived throughput as including warmup.
GPU resident peak from the guard, PyTorch first-batch allocated peak and the
maximum of sparse process-RSS samples are distinct measurements. No average
power, GPU utilization, RAM peak or full-epoch ETA is invented from sparse
heartbeats or four optimizer updates.

## Source inspection

| Area | Owner and observed implementation | What remains to measure |
| --- | --- | --- |
| Trainer | `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`; one shared Entry/Exit learning path | Current-source fixed-step TRAIN/VAL runtime |
| Input dimensions and sequences | `entry_model_native_signal_v1.py`, actual bundle metadata, per-TF recipe lengths; original request's approximate feature count and seq513 name do not describe actual tensors | No shape change is justified |
| Feature groups | All existing specialist groups are listed from the bundle's exact routing metadata in the generated JSON | TRAIN/VAL-only usefulness and masking ablations, not gate-weight rankings |
| Architecture | `EntryV10CtxHybridTransformer`; local encoder, per-family encoders, MTF encoders, fusion, shared Entry/Exit representations and heads | Attention/FFN/projection runtime and full-call multiplicities |
| Attention | PyTorch `TransformerEncoderLayer`/`TransformerEncoder`; GELU, pre-norm, nested tensor disabled | Actual installed-kernel dispatch and representative profiler trace; SDPA/Flash use not established |
| Data storage | Scalar Parquet columns loaded separately; streamed Arrow batches, exact cache/surface binding, selected-row storage and bounded memmaps | Preflight, disk/page faults, repeated-start overhead |
| Batch construction | Existing index map, Pandas scalar row access, tensor copies and writable contiguous per-sample MTF buffers | Whether eliminating any copy materially improves measured loader wait |
| Loader | Fixed zero workers, no persistent workers/prefetch; local FP32 pinning disabled | Only benchmark pinned transfer if measured GPU starvation supports it |
| H2D | Local inputs use non-blocking `.to`; MTF/Exit transfers live inside their existing owners | Separate all H2D from computation before attributing cost to PCIe |
| Optimizer | AdamW, separate task-weight parameter group without decay; finite global-norm clipping; no explicit fused/foreach selection | Optimizer update cost and individually benchmarked low-risk variants |
| Loss | Existing masked raw-bps fitted-Q MSE, learned homoscedastic weights, auxiliary objectives | Loss/gradient trajectories and VAL parity for every precision variant |
| Gradients | Streamed Exit backward and returned Entry-representation gradients; main backward; accumulation support with source-owned remainder handling | Same-effective-batch comparisons require an explicit experimental recipe |
| Precision | Local deterministic FP32, TF32/compile/autocast off; BF16 policy restricted to Hopper | RTX 3090 AMP needs its own explicit experimental policy and parity proof; Hopper metadata cannot be reused |
| Activation memory | CUDA retains activations under allocator fence; CPU uses non-reentrant checkpointing with RNG preservation | Activation peaks and recomputation tradeoff under the physical guard |
| Scheduler and EMA | Cosine over declared epochs; EMA horizon derived from effective epoch rows | A sampled smoke has a different EMA horizon; smoke VAL does not establish candidate parity |
| Checkpoints/resume | Existing hash-bound two-slot sessions, model/teacher/optimizer/EMA/scheduler/RNG/order/progress and validation accumulator | Current variant restart proof; prior production next-batch evidence remains historical |
| Validation/early stop | Existing candidate full-population validation, economic selection monitor and frozen patience/max-epoch policy | Full-VAL cost and compute-normalized improvements after one real epoch |
| Safety | Canonical capped runner, shared exclusive job lock, cgroup RAM/swap/tasks, numerical threads, signed telemetry/watchdog | Fresh source-bound recipe and launch preflight before any new CUDA |

## Instrumentation implemented in the existing trainer

- `TRAIN_EFFICIENCY_BATCH`: cold first-batch JSON separating initial fetch,
  initial local-input H2D, Entry online/target forwards, Exit work, main backward
  and optimizer update. The optimizer interval includes gradient checks,
  clipping, AdamW, zeroing and EMA; it is not advertised as pure AdamW time.
  The existing nested Exit phase profile is retained. MTF/Exit H2D remains
  included in its owning phase and is not claimed as separately measured.
- `TRAIN_EFFICIENCY_WINDOW`: synchronized post-warmup throughput. Local
  multi-step canonical smoke excludes its first optimizer update; Hopper
  retains its contract-owned warmup count. Row and optimizer counts belong to
  the measured interval. One discarded step is an initial bounded diagnostic,
  not proof that performance has reached a stable plateau.
- `TRAIN_EFFICIENCY_VAL`: synchronized validation including EMA swap where
  active, with the actual loader dataset row count and full-trajectory flag.
- `TRAIN_EFFICIENCY_CHECKPOINT`: serialized state size and real write-plus-fsync
  time, before hashing/publication. It does not claim to measure resumable
  optimizer/session checkpoint I/O or the whole bundle export.

These additions do not change input/target bytes, model dimensions, optimizer
settings, batch size, power, safety limits or candidate stopping policy. The
measurement path is wired through the existing trainer and recipe, not a
second CUDA runner. Real-data current-source numerical/runtime parity remains
unproved until the new source can pass the canonical execution chain.

## Concrete next benchmark

After the full source package is reviewed, tested and committed, materialize a
new immutable canonical local smoke recipe using the existing materializer.
Keep all data bindings and batch/precision settings from the verified local
recipe. Use a bounded 512-row TRAIN/VAL sample, matching the existing historical
throughput experiment's sampling budget. Run exactly once through the capped
runner and signed guard. The local timing policy discards the first update;
report the remaining measured row/step counts, never the full sample divided
by only the post-warmup time. Do not run the old smoke again under changed
source hashes and do not relabel an old bundle.

Additional measurement work still needed before a qualified baseline:
GPU utilization and TRAIN-scoped power integration, host CPU/cgroup peaks,
complete loader/H2D attribution, representative kernel profiling, checkpoint
resume equivalence and full-epoch/full-VAL timing. Longer benchmark windows
must be justified by the first measurement, not started blindly.

Priority hypotheses, not accepted optimizations:

1. Precision/attention kernels in repeated online and teacher Exit passes.
2. Safe micro-batch/Exit-chunk geometry under the existing VRAM limit.
3. Optimizer implementation if its newly separated interval is material.
4. Avoiding repeated preflight work through the existing exact resume path.
5. Data-copy/pinning changes only if post-warmup loader/H2D measurements show
   starvation. Feature deletion remains later and requires VAL ablations.

## Capacity-gate defect found during regression

The unfinished cloud code combined a fixed measured-step count with a
600-second lower limit on the TRAIN measurement. With the exact physical
TRAIN population in the smoke lineage, even the largest permitted cloud
batch could not satisfy the 43.2-hour ceiling under that duration floor.
This is algebra from source and lineage, not a cloud hardware measurement.

The gate now uses the immutable owner's exact warmup/measured step counts and
strictly positive finite synchronized time. It no longer rejects a completed
fixed-step sample for being fast. It does not pad time, invent throughput or
change the 43.2-hour admission, 48-hour deadline, NOK 2,500 budget or hardware
limits. Regression fixtures explicitly distinguish a synthetic fast sample
from the preserved slow 600-second sample: both batch sizes must still fail
the time gate for the latter. No cloud host is thereby qualified or purchased.

Two resume tests also lacked the newer required explicit precision argument,
and the Hopper recipe fixture omitted its required fixed-step sampling
geometry. Their inputs now follow the existing owners; production constraints
were not weakened to satisfy these tests.

## Remaining execution boundary

The canonical worktree already contained a substantial uncommitted cloud
package and documentation changes when this wave began. They are preserved.
The source is not clean/committed, additional cloud integration coverage and
full-package review remain, and the active handover still blocks launch.
The user's local benchmark request is in scope; the missing executable
source/recipe chain is not solved by another permission question or by
silently clearing a hold. No fresh CUDA benchmark, full epoch, AMP experiment,
power sweep, feature ablation or cloud purchase ran in this wave.
