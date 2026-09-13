<!-- GX1_DOCUMENT_CLASS: CANONICAL | current status -->
# Current GX1 status

The user explicitly authorized VAL performance items 1–4 on 2026-09-13.
Frozen source: 39bdb3ce327b2ba408e43573b3e535fdc30bea06 at /home/andre2/src/GX1_VAL_CPU_PIPELINE_V38.
Branch: fix/native-val-cpu-pipeline-20260913 (pushed and remote HEAD verified).
Use CURRENT_NATIVE_RUN.json for the exact recipe, campaign and runtime paths.
The documentation checkout is not the training checkout.

The successor avoids rebuilding/copying cached market input, batches economic
row lookup and reuses identical HOLD cost calculations, prepares dynamic
position input with four CPU-only spawned workers inside the same capped job,
and admits all 19 WSL vCPUs with normal CPU priority. Numerical libraries in
the parent retain eight threads; workers have one. Data, features, all five
timeframes/eight families, both VAL sides, FP32, TRAIN batch 16, Exit VAL batch
128, 30-epoch maximum and patience 5 remain unchanged. No confidence cutoff,
hold cap, reduced VAL cohort or precision relaxation was introduced.

The prior 0e81f5b8 window ended RESUMABLE with both process exit codes zero and
guard PASS. It saved 1,344,280 views / 10,691 forwards, state cursor 244/640,
progress SHA 96331710b34d464ce690a60edf9a6565fea21ecb803d9d36f9b1db7aab021b51.
The new recipe binds that exact progress file and its existing immutable EMA
snapshot. The evaluator admits those accumulators only when model checkpoint,
Entry policy, data/rollout identity and prior execution contract still match.
Only the CPU-execution contract identity changes. TRAIN remains checkpoint 309
and 19,588 optimizer steps; no completed TRAIN or smoke is repeated.

Nineteen focused cases passed, including unchanged TRAIN gradients, compact
cache miss/hit/mixed model outputs/actions, CPU states around the 512-row
boundary, exact costs/slice hashes across entries/sides, source rejection,
checkpoint preservation and progress migration rejecting identity drift.
Required source commit hooks passed. On the actual GPU, first-batch checks
must additionally confirm states/actions/costs and report full batch pipeline
time before claiming throughput improvement. Campaign is prepared; actual
startup/resume and sustained throughput still require observation.

Runtime limits remain 20 GiB RAM, 512 MiB swap, 128 tasks, 300 W driver cap,
310 W draw stop, 85 C core / 80 C memory and 12 GiB VRAM. Signed local safety
telemetry remains frequent; model observations approximately every 15 minutes.
TEST stays sealed. Positive main-model net Bps and live readiness are unproven.
