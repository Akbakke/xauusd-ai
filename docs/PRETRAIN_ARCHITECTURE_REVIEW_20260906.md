# Integration note

This is the source review snapshot, not a claim that later repairs remain open.
Source references are normalized to the verified WSL repository; line numbers
refer to the reviewed snapshot and can move with repairs. Current execution,
fix status and tests are in `PRETRAIN_READINESS_REPAIR_20260906.md`.

# GX1 architecture and optimization review — 2026-09-06

## Decision for the parent

**Retain the current research candidate unchanged. Do not enlarge it or buy compute on architectural intuition.** GX1 is modest in individual hidden width but complex in the number of dense temporal routes, training tasks, target-network passes and qualification mechanisms. There is no evidence here that it is too small, that all its complexity earns its cost, or that another model would trade better.

Most actionable findings:

1. **P1, existing qualification gap:** research economics and capacity-forced Exit termination are not production economics/long-lived serving semantics. More capacity does not repair them.
2. **P2, source-proven integration gap:** the structure-preserving Entry/Exit usefulness engine exists, but its CLI only validates a pre-existing report; its computation functions have no production caller in the inspected tree. The selective evaluator explicitly disables feature-mask ablation. “Ablation exists” must not mean “economic usefulness has been measured.”
3. **P2, optimization opportunity, speedup unmeasured:** repeated device-to-host checks/statistics and repeated identical Exit timeframe-scale computation are concrete avoidable work to profile before changing architecture.
4. **P3, confirmed dead state:** four retired static Exit modules remain registered without a forward use. Source algebra gives **562,688 parameters / 2.1465 MiB FP32 per model copy**, not an explanation for the entire training time. Remove only in a separately approved successor with explicit checkpoint/bundle migration.
5. **Experiments, not findings:** first compare a regularized linear Q regressor, one histogram-tree Q regressor and one shallow shared neural challenger under chronological, equal-information/economics controls. Do not launch a model zoo, five expensive seeds per idea, a large foundation model or a new RL stack.

## Evidence maturity: the current decision is not architecture replacement

The parent/user supplied the following current status explicitly: **313,399 TRAIN rows spanning approximately five years; 5,509 VAL rows from June 2026; the candidate remains in its first epoch at 9,664 optimizer steps; the active technical smoke had only four updates.** These are supplied actual-run facts, not new measurements by this subagent. No remote status or artifact read was used to refresh them.

| Evidence available | What it supports now | What it cannot support |
| --- | --- | --- |
| 313,399 chronological TRAIN rows, about five years | A substantial historical fit population and a reason to preserve the ongoing reference experiment | An iid sample size of 313,399, sufficient effective capacity/data, or out-of-time edge; overlapping histories/outcomes reduce independence |
| 5,509 June-2026 VAL rows | A specific held-out-month research comparison on its declared population | Multi-year/regime robustness, reliable inference from every thin coverage slice, or repeated unrestricted architecture searching |
| Candidate: 9,664 steps, incomplete first epoch | Operational progress and, where recorded, training mechanics | Its first complete TRAIN/VAL result, a stable learning curve, convergence, overfitting/underfitting diagnosis, or a capacity verdict |
| Smoke: four updates | Technical routing, finite outputs, export/reload and diagnostic mechanics where actually checked | Family utility, an expert winner quota, useful learned task weighting, or justification to remove seven never-top-ranked families |

Under the stated batch-eight/one-step-per-batch geometry, one full epoch has `ceil(313399 / 8) = 39175` batches; 9,664 steps is approximately **24.7% of that first epoch**, not a completed first candidate assessment. This is arithmetic conditional on that geometry, not a refreshed progress measurement. The five-year TRAIN span does not compensate for a one-month outer VAL span. Conversely, a one-month VAL limitation does not prove the model is bad or too large. Neither a larger model nor a smaller model resolves that evidential limitation by itself.

**Authorization distinction:** review, identified fixes and scoped build/remove opportunities are authorized; automatic architecture replacement, feature/task removal, a new training campaign and any purchase are not. The baseline shortlist below is a later decision menu, not an implementation backlog to execute now. Treat its first comparison as a request for a separately bounded experiment only if existing candidate evidence leaves a concrete question unanswered.

**Parent verification update:** all **42 existing `test_entry_v10_ctx_model_shapes` tests passed on WSL CPU through the 4G capped runner**, as explicitly reported by the parent. This covers joint eight-family gradient reachability, exact-contract shape, token/context influence, incremental carry/restart and train/eval normalization controls. Evidence class: **M — actual CPU execution on synthetic mechanical fixtures**, not real-market measurement, family-benefit ablation or profitable behavior. This subagent did not independently inspect the test result artifact. No GPU training or new VAL inference was started. The result materially strengthens the mechanical-readiness milestone; it does not authorize replacing the architecture or repeat-running that same completed test group unnecessarily.

### Decision milestones, not a large new-model project

1. **Restore trustworthy evidence and operations.** Parent completes the HAC semantic-version correction, semantic seed comparison, retention/event safety and host-clock work. The 42-test model-shape/gradient/control group is now reported passed; retain that evidence and run only still-relevant outstanding checks. Publish new immutable evidence without changing thresholds to recover a pass. These fixes and mechanical passes do not establish model quality.
2. **Reach the first meaningful candidate checkpoint/VAL milestone.** Preserve the current session and let the parent decide safe continuation through the unchanged approved recipe. Obtain the first full epoch and complete bound VAL/Exit evidence before diagnosing model size. A single first epoch still does not prove convergence, especially with epoch-boundary fitted-Q teachers.
3. **Read the learning evidence before proposing a learner change.** Review Q errors and bias, actual policy outcomes, task precisions, valid population counts and per-family gradients across the permitted training/early-stop progression. Keep the frozen epoch/patience/selection rules; do not add new stopping thresholds after seeing VAL. Distinguish technical failure, lack of economic signal, estimation noise and expensive compute.
4. **Decide whether usefulness wiring or optimization earns a small build.** F2 is an existing-owner integration opportunity, and F3 is a profile-guided optimization opportunity. Approve a bounded scope and source/session compatibility plan before either touches source. Four-update gate dominance is not a removal criterion. F4 stays a later migration/cleanup decision, not a reason to restart.
5. **Approve at most the next informative baseline experiment.** If the current reference is valid but its complexity remains unjustified, choose the cheapest arm that answers the diagnosed question, predeclare its data/compute budget and keep TEST sealed. Do not implement all listed options or run five seeds per idea automatically. A negative or inconclusive result is retained, not explained away by another model rewrite.
6. **Separate research selection from final admission.** Only after a coherent candidate/limited comparison is frozen should the parent seek the separate economics, terminal, portfolio, same-bundle parity and untouched-TEST milestones. No architecture decision bypasses them or manufactures a longer honest OOS history.

## Authority, identity and evidence classes

- Reviewed only the staging source at `/home/andre2/src/GX1_ENGINE`, originating from the user-verified WSL commit `d4d459c13ec1235ed46fbb5bceb7a72f97057368`. This is not the old Mac repository.
- Read staging `GX1_RULES.md`, `AGENTS.md`, `CLAUDE.md`, relevant system/data/handover sections, and `/private/tmp/GX1_BEFORE_CLOUD_TRAINING_20260906.md:1` plus `/private/tmp/GX1_REPO_REVIEW_20260906.md:1`. Current instructions override historical launch permissions. No handover/remote operation was run.
- **S — source proof:** inspected executable control/data flow, static references or data-independent algebra. This proves implementation properties, not profitable behavior.
- **D — actual-data measurement:** none newly made by this subagent. The earlier report's real smoke-VAL observations are context only, not independently recomputed here. They do not describe the unfinished candidate's quality.
- **M — mechanical fixture evidence:** relevant existing test source was read; no tests or model forwards were executed by this subagent. The parent explicitly reports all 42 model-shape tests passed on capped WSL CPU/4G. Their random/synthetic inputs establish mechanics, never trading benefit; the result artifact was not independently inspected here.
- **U — unproven:** architecture benefit, live fidelity, absolute/relative throughput, useful capacity, model quality, any claimed economic improvement.
- Only this report was written. No source/configuration changes, imports of the training model, GPU use, training, installs, SSH, TEST access, artifact/checkpoint mutation, commits, branches or financial actions occurred.

## 1. What the architecture actually does

### Dense all-family processing, not sparse expert routing — S

The model's internal configuration fixes hidden width 128, four attention heads and a three-layer generic local Transformer; specialist/MTF depths and fusion magnitudes are explicit constructor/recipe inputs. See `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:243` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:472`. I did not instantiate the production model or measure its total/active parameter count. Signal/context/MTF widths should still be resolved from their owners, not copied from historical documents.

The architecture owner declares Entry local M5/96 rows plus M15/H1/H4/D1 and Exit native M1/480-row initial context plus M5/M15/H1/H4/D1. The respective MTF windows are code-owned, including the 252-row daily lane. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_exit_production_architecture_v1.py:50`.

Entry processing:

- Generic local sequence projection/Transformer and snapshot/context fusion.
- Eight family-specific sequence projections/Transformers, then cross-family attention and softmax-weighted residual fusion.
- On each of four higher timeframes, all eight family temporal encoders run. Feature-conditioned gates, family/timeframe attention and further learned fusion follow.
- The Q head sees local, final fused, MTF and family-context representations. Auxiliaries are not post-model voters.

Execution proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1826`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1850`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1940`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2012` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:3150`.

**The gates do not skip any family computation.** Local eight-way and Entry MTF 32-family/timeframe evaluations occur before weighting. Small gate weights do not provide sparse-MoE speed savings, and top-rank frequency is not an ablation result. Cross-attention also mixes information before the reported post-attention gate, so its weight is not an isolated raw-family attribution.

### Shared model does not mean identical Entry and Exit temporal encoders — S

Exit shares local/MTF projections, normalization, categorical embeddings and context owners with Entry, but uses dedicated causal recurrent modules: one global GRU, eight local-family GRUs, eight MTF-family GRUs shared across five clocks, and one path GRU. Thus there are 18 registered active GRU modules and, by loop expansion, 50 GRU invocations per full episode-batch forward. Online and frozen-target Exit forwards both run during a training batch. This is source arithmetic, not a measured kernel/latency profile.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1282`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2203`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2364` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2480`.

Each MTF history is scanned causally, then gathered at each state's closed-timeframe index. Family and timeframe attention run across tokens **within a state**, not bidirectionally across future Exit states. The same state-level work is repeated over the episode; the path scan preserves temporal history. Both sides share market computation before side/path fusion. Therefore “replace repeated 480-by-480 Exit temporal Transformers with a GRU” describes an optimization already implemented, not new work.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2377`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2422` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2486`. The stale Transformer-cost explanation at `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:1054` should not drive a hardware estimate.

## 2. Objectives, targets and gradient checks

### Retain the coherent decision semantics — S; economic adequacy U

- Entry LONG/SHORT targets are detached values from the frozen target model at each side's first authoritative post-fill Exit state. FLAT has target zero; valid-action masking is explicit. This is expected-return regression, not calibrated direction probability or classification accuracy. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_fitted_q_v1.py:74` and `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_fitted_q_v1.py:250`.
- Exit uses current executable research PnL for EXIT_NOW and the frozen target's next-state maximum for HOLD, with gamma one and no intermediate HOLD reward. The pathwise hindsight optimum is explicitly not a training label. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/unified_exit_fitted_q_v1.py:58` and `/home/andre2/src/GX1_ENGINE/gx1/contracts/unified_exit_fitted_q_v1.py:180`.
- TRAIN supplies the frozen teacher. A new session initially copies the initial model; the teacher is refreshed only after an epoch boundary, not per minibatch and not by validation optimization. Initial Entry targets therefore do not constitute evidence of an already learned Exit policy. Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:10880` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:11368`.
- The ten-task contract covers the two Q tasks, side MAE, trendline events, masked sizing, dip, forecast return, dip timing, tail risk and forward volatility. The trainer uses raw-bps masked MSE for decisions, with BCE/L1/pinball/MSE on the relevant auxiliaries. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_model_native_joint_task_weighting_v1.py:22` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:403`.
- Scalarization is `sum(exp(-s_i) * L_i + s_i)` for supervised tasks. Empty sizing masks omit that task. Exit streams the precision-weighted loss gradient, adds its log-variance term once and explicitly reinjects the token gradient into Entry. A detached intermediate token does **not** mean the Entry-to-Exit gradient is lost. Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:330`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:6631`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:8120` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:8163`.
- Task log-variances are exempt from AdamW weight decay; otherwise decay would add an unadvertised pull toward equal weighting. Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:12622`.

Uncertainty weighting has a research rationale, not proof that these eight auxiliary tasks improve XAUUSD returns. Its cited evidence is scene understanding, not trading. The algebra balances loss scales; it does not detect harmful shared-gradient directions or guarantee useful forecasts. Monitor each task's loss, precision, gradient norm and its gradient alignment with the decision tasks before proposing task removal. [Kendall, Gal and Cipolla, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).

Checkpoint selection is full-VAL realized **gross/spread-inclusive mean bps per Entry row**, including FLAT zeros, not a calibrated-probability metric, not aggregate auxiliary loss, and not an overlap-constrained portfolio return. It selects actual model policy outcomes rather than simply agreement with a moving teacher. Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:10438`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:10552` and `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_candidate_checkpoint_policy_v1.py:19`.

### What connectivity evidence does and does not establish

| Check read | Evidence it can provide | Important limitation |
| --- | --- | --- |
| Production field-partition gradient test | Exact owner field order reaches eight local/MTF projections and each local gate-logit row after zero-init residuals open | Uses generated inputs, four-row histories and fixture normalization, not market-data gradient/utility evidence |
| Joint Entry/Exit family test | Eight shared projections and both sets of Exit-family GRU input matrices receive finite nonzero gradients | Small generated state/target fixtures; not per-family economic contribution |
| Prefix/append/carry parity tests | Future appends do not change prior Q values; incremental carry agrees with episode execution within stated tolerance | Generated inputs; not same-bundle real-data serve parity or production terminal qualification |
| Trainer movement proof | At least one parameter in each named component group changes | Reduction is a maximum across group members; “episode_family_encoders moved” alone does not prove every family moved |
| Entry VAL input influence | Input-margin gradients and family perturbations change Q values | Sensitivity/liveness, not favorable realized PnL or unique causal feature value |

Proof locations: `/home/andre2/src/GX1_ENGINE/tests/test_entry_v10_specialist_fusion_model.py:193`, `/home/andre2/src/GX1_ENGINE/tests/test_entry_v10_ctx_model_shapes.py:444`, `/home/andre2/src/GX1_ENGINE/tests/test_entry_v10_ctx_model_shapes.py:889`, `/home/andre2/src/GX1_ENGINE/tests/test_entry_v10_ctx_model_shapes.py:950`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:5606`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:7384` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:7492`.

These checks complement one another; no actual disconnected family was established in this pass. The parent now reports all 42 model-shape tests passed, including the joint eight-family, influence, carry/restart and normalization cases. Other named test files are read-only review unless separately reported executed. Next inspect per-family gradients/movement on a meaningful bound checkpoint rather than repeating the completed group. Do not force equal weights, add a top-rank winner quota, or treat a nonzero derivative as profitable information.

## 3. Concrete findings and bounded actions

### F1 — P1 before production: economics and terminal mismatch remain — S

The economics owner explicitly keeps net costs, financing, economic terminals, gap evidence and overlapping capital constraints unready. The research episode pack requires a 512-state capacity terminal; the model masks HOLD at the last complete-episode state. The runtime data contract describes a rolling tail and continuation beyond that boundary. These are separate train/serve decision conditions, not a mere tensor-shape issue.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_fitted_q_v1.py:115`, `/home/andre2/src/GX1_ENGINE/gx1/contracts/unified_exit_episode_pack_v1.py:108`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2512` and `/home/andre2/src/GX1_ENGINE/docs/DATA_CONTRACT.md:298`.

**Action:** keep the current candidate as a research reference; separately specify economic termination versus storage truncation, continuation targets, actual fills/costs/financing and capital-constrained replay. A target change is a successor experiment, not an unchanged resume. I did not reproduce erroneous trades or infer that the corrected Entry-notional target is still wrong: its signed LONG/SHORT denominators are correct in source at `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_causal_m1_outcomes_v1.py:278`.

### F2 — P2 before claiming family usefulness: existing engine is not end-to-end wired — S

The usefulness owner already specifies observed VAL block donors, coupled sequence/snapshot/alias swaps, valid categories, family/timeframe perturbations and paired loss/margin deltas. It explicitly grants no fit, retirement or admission authority. This is a better starting point than inventing another importance engine.

However, `audit_task_feature_usefulness` requires caller-supplied state arrays and a `predictor` callback. Its CLI accepts only `--validate-json`. Repository-wide reference searches under staging source/scripts/tests found its computation/build functions only in their own definition/export and tests, not in a production producer. The selective-edge evaluator sets `feature_mask = {"enabled": False}`. These are source findings; no assertion is made that an external historical script never called it.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_exit_feature_usefulness_v1.py:74`, `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_exit_feature_usefulness_v1.py:89`, `/home/andre2/src/GX1_ENGINE/gx1/scripts/audit_entry_exit_feature_usefulness_v1.py:572`, `/home/andre2/src/GX1_ENGINE/gx1/scripts/audit_entry_exit_feature_usefulness_v1.py:1011` and `/home/andre2/src/GX1_ENGINE/gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py:2185`.

**Action:** parent should scope an adapter through the existing owners that binds the actual loaded model, frozen target, VAL data, Entry token and episode-native Exit path. Prove the callback uses the real episode/carry implementation rather than reviving the retired static Exit interface. Stream under approved limits; do not materialize all overlapping dense Exit windows in RAM. Publish diagnostic deltas without automatic deletion/selection. Subsequently compare actual Entry/Exit policy outcomes; Bellman-target loss/margin deltas alone remain weaker evidence than realized economics.

### F3 — P2 performance opportunity: repeated scalar work and synchronization — S for work, U for impact

The Exit MTF loop evaluates the identical learned timeframe scale twice per family (history and current residual): 80 evaluations across five clocks/eight families per Exit forward, versus once per timeframe in Entry. Each call performs softplus and a finite check using Python truth conversion of CUDA reductions. Task reporting additionally performs repeated `.cpu().item()` reads per active task; many input/output guards also synchronize.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1580`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2377`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2400`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:79` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:363`.

**Action:** profile, then consider hoisting the scale once per timeframe and batching diagnostic transfers/reductions. Never cache the scale across optimizer steps, detach its gradient, omit per-field finite checks, or permit a bad gradient through an optimizer step. Shared computation is algebraically equivalent, but FP32 gradient reduction order can differ; actual forward/loss/all-active-gradient/optimizer/RNG parity is required before accepting an optimization. These changes alter bound source and cannot be silently substituted into this session. PyTorch explicitly documents scalar reads and CPU-dependent CUDA control flow as synchronization points. [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).

### F4 — P3 remove later: retired static Exit parameters are still serialized — S

`exit_path_encoder`, `exit_entry_query_norm`, `exit_entry_path_attention` and `exit_fuse` are constructed, but none is loaded/called through `self` elsewhere in the model AST. All-source name searches agree; the trainer explicitly acknowledges their retired status and excludes them from active movement proof. Active Exit uses `exit_episode_fuse` instead.

Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1255`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:2495`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:5511` and `/home/andre2/src/GX1_ENGINE/tests/test_unified_exit_parameter_movement.py:64`.

Data-independent parameter arithmetic uses width 128, feedforward 512, two retired Transformer layers and the listed Linear/LayerNorm/attention shapes: `2 × 198272 + 66048 + 256 + 99840 = 562688` parameters. The two-layer constant is at `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/direction_decision_contract.py:101`. This is 2,250,752 FP32 tensor bytes per model copy, excluding serialization overhead. Do not invent Adam state for unused parameters or call this a measured checkpoint-size/throughput improvement.

**Action:** preserve current keys/checkpoints; later remove dead registration and stale metadata only with a migration contract and strict reload/output/RNG checks. Removing constructors changes RNG consumption during fresh initialization, so a nominally equal seed is not by itself parity proof. This is cleanup, not a compelling reason to restart training.

### Additional interpretation risks, not newly proven training bugs

- Group-max movement can hide an individual dead member; augment the diagnosis with existing per-parameter deltas and genuine per-family gradients rather than interpreting the group boolean as complete proof.
- Model-width and attention-share diagnostics cannot estimate effective independent sample size. Overlapping M5 entries and hundreds of M1 states from the same episode are not independent trades.
- A frozen-target maximum can suffer value overestimation, but it was not measured here. The current owner explicitly marks Double-Q inactive; consider it only after directional Q-versus-realized bias is observed on chronological validation, as a separately declared target variant. Atari results are not XAUUSD evidence. [Van Hasselt, Guez and Silver, Double Q-learning](https://arxiv.org/abs/1509.06461).
- The parent owns HAC/seed fixes, host diagnostics and documentation integration. This report does not reclassify their patches, rerun their tests or imply those issues are resolved.

## 4. Retain / simplify / remove later / experiment

| Decision | Recommendation | Why / evidence boundary |
| --- | --- | --- |
| Retain now | All eight raw feature families, separate native clocks, closed HTF construction, exact aliases and TRAIN-only transforms | These preserve the problem definition and prevent known wiring/leakage failures. No measured family redundancy sufficient for removal was established |
| Retain now | One bundle, explicit token, unique masked Q argmax, signed bps targets, separate target snapshot and full trajectory evaluation | Coherent source semantics; not proven production adequacy |
| Retain now | Existing candidate/checkpoint, causal Exit GRUs, capped FP32 recipe and exact resume | Source already contains major memory/causality optimizations; restarting discards the reference without a demonstrated benefit |
| Simplify next, after profiling | Repeated scalar computations, redundant host synchronization and diagnostic transport | Concrete work can be reduced without designing a new model; actual speed/parity still unproven |
| Remove later | Four retired static Exit modules and their obsolete descriptions | Demonstrably unused forward state; small storage benefit; migration/RNG implications |
| Experiment first | Shallow shared neural temporal encoder/fusion with all eight owner inputs and unchanged heads/economics | Tests whether stacked family/axis fusion earns its cost; preserve every causal input/window in the first arm |
| Experiment only with evidence | Auxiliary-task ablation, fewer attention stages, patching, altered width, Double-Q | Changes the learner/target. One factor at a time, successor identities, explicit approval, no TEST selection |
| Do not add now | More indicators/families, sparse expert/router machinery, LLM agents, online adaptation, a second trading decision authority, generic PPO/IQL stack | No demonstrated failure requiring them; adds tuning, lineage and evaluation burden; several violate frozen scope |

Auxiliary **heads** are cheap Linear layers relative to temporal encoding; deleting them is unlikely to solve compute by itself. Their **gradient influence** can still be harmful or helpful. Separate those questions. Keep sizing mandatory in the current contract; do not convert a diagnostic fixed-size/no-trade reference into a fallback execution rule.

## 5. Small, relevant baseline shortlist — proposals only

No cited paper establishes a universally better trading model. Forecast MSE on public benchmarks is not cost-adjusted Entry/Exit performance. Each proposal needs explicit later experimental authorization; nothing was trained here.

### A. Regularized linear action-value regression: cheapest capacity check

Use Ridge on a declared causal state representation and the same frozen TRAIN-teacher raw-bps targets/valid masks. First use a declared present-state tabular projection as a **reduced-information diagnostic**; if comparing architecture fairly, also give the linear model the same eligible lag/window information through a bounded, explicitly declared representation. Do not flatten the entire population's overlapping windows into a huge in-memory array by default. Report the information restriction rather than concluding that temporal modeling failed.

This tests whether the Q mapping is already approximately linear and supplies a transparent regularization baseline. It does not independently validate the teacher or prove an end-to-end trading policy. Ridge is already available in the repository-pinned scikit-learn dependency; no additional package is needed for a future implementation. Proof: `/home/andre2/src/GX1_ENGINE/requirements.txt:12`. [scikit-learn 1.7 Ridge](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html).

### B. One histogram-boosted tree regressor: nonlinear tabular check

Use `HistGradientBoostingRegressor` rather than immediately adding multiple boosting libraries. Compare on the same declared input representation, targets, chronological selection budget and errors as A. The paper motivation is that tree methods remain strong on heterogeneous tabular benchmarks, not that those benchmarks prove superiority on dependent financial sequences. [Grinsztajn, Oyallon and Varoquaux, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf).

Important implementation trap: HGB's automatic early-stopping validation split is not the GX1 chronological contract. Disable it and externally select iterations, or pass explicit chronological `X_val`/`y_val` from the permitted inner window; the pinned 1.7 API supports these arguments. No unseen future window may participate in binning, feature selection or teacher fitting. [scikit-learn 1.7 HGB](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

Fitted-Q is not inherently neural: tree-based batch fitted-Q has primary research precedent. That justifies a later explicit tree-FQI research arm, not a drop-in replacement for GX1's shared neural token/bundle or an expectation of profit. [Ernst, Geurts and Wehenkel, JMLR 2005](https://www.jmlr.org/beta/papers/v6/ernst05a.html).

### C. Shallow shared MLP/temporal-linear neural arm: best first full-model simplification

Preserve all eight owner inputs, causal sequence extents, categorical handling, Entry/Exit actions/masks, path/token information and task contract, but compare a small shared temporal projection plus residual MLP/fusion against the current layered attention stack. Do not simultaneously remove families, change objectives and alter costs. A same-interface neural arm is closer to the current shared-model contract than a tree model, but still requires an explicit successor architecture identity.

A ResNet-like MLP is a serious tabular baseline, and the benchmark comparing it with Transformers/GBDTs explicitly finds no universal winner. [Gorishniy et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html). Linear temporal models are also a useful sanity check: LTSF-Linear/DLinear outperformed several Transformer forecasters on the authors' long-horizon forecasting benchmarks, which is not the same task as GX1 fitted-Q/optimal stopping. [Zeng et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089).

**Later, only if temporal attention is a measured bottleneck and useful:** a PatchTST-style patch representation is a bounded alternative to investigate, because patching reduces token count while retaining history. GX1's heterogeneous owner fields and sparse events are not interchangeable univariate channels; preserve event timing, closed clocks and all inputs, and measure any degradation. Its forecasting success is not trading evidence. [Nie et al., PatchTST](https://arxiv.org/abs/2211.14730).

### Fairness boundary for all baseline comparisons

A/B against one frozen neural teacher are **target-approximation/distillation diagnostics**, not independent joint-policy champions. A tree cannot simply replace Entry while the old neural Entry token is silently reused for Exit: that violates the exact decision-token source contract. Either stay explicitly diagnostic, or define a separately approved complete challenger preserving the same economic/action/information semantics and evaluate its own coherent Entry/Exit policy. Do not publish mixed-bundle pseudo-PnL.

The existing offline challenger owner accepts review-only rolling-OOS evidence and cannot launch or promote a model. It is a place to integrate legitimate future comparisons, not an already implemented baseline trainer. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_offline_challenger_v1.py:1` and `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_offline_challenger_v1.py:267`.

## 6. Honest experiment and performance plan

### Before any new model-quality experiment

1. Parent completes current evaluator/seed mechanics and focused source-contract/gradient tests. Preserve historical evidence and the active session. No smoke fixture may decide which family or objective to remove.
2. Recover the exact current split/recipe/teacher identities from bound artifacts. Use chronological expanding/rolling folds **inside TRAIN** for limited design/regularization selection. Refit every learned preprocessing/ranker/normalization/teacher only on that fold's training prefix; reusing a teacher fitted on the later inner-validation interval would leak even if the student's fit is chronological.
3. Apply the existing outcome-completeness/lifecycle boundary logic at every fold. Purge training outcomes that cross the next interval; retain legitimate causal feature warmup. The source already guards `crosses_split_end`; do not invent a blanket D1-length embargo or let auxiliary labels cross the fold boundary. Proof: `/home/andre2/src/GX1_ENGINE/gx1/contracts/unified_exit_lifecycle_v1.py:1328` and `/home/andre2/src/GX1_ENGINE/docs/DATA_CONTRACT.md:244`.
4. Predeclare a small shortlist, tuning/compute cap, economic estimand, tie/mask behavior and selection rule before inspecting its results. Keep all attempted arms in the record. Use current VAL only for allowed selection/confirmation; previously inspected VAL is not pristine discovery evidence. Never alter the old coverage grid/thresholds to recover a PASS.
5. Compare the same chronological decision population, same research fills/spread treatment, same exit terminal semantics and same sizing/capital convention. Once production economics changes, rerun **both** arms on that same successor contract; do not compare gross baseline numbers with net challenger numbers. Include the same coin-flip and no-trade reference as evaluation nulls, never model overrides.
6. Report per-action Q errors/bias, actual realized Entry-plus-Exit outcomes, coverage/FLAT frequency, drawdown, costs/exposure and time/regime slices. Preserve row/trade/episode counts. Use chronological dependence-aware uncertainty, accounting for irregular trade times and overlapping outcomes; many M1 states are not many independent samples. Do not infer significance from uncorrected model/coverage searching.
7. Only a shortlisted viable approach merits the separately authorized seed campaign; use the corrected semantic seed comparison. Freeze model/recipe/calibration before any separately authorized final TEST opening. TEST remains untouched throughout this work and is never reused to choose an architecture.

### Performance plan for the parent, not executed here

- First use current profiling hooks: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:6269`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:6464`, `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:6525` and `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:7997`. These use a synchronized monotonic clock, but the normal trainer instruments its first batch, which is not sufficient steady-state evidence.
- In an explicitly approved, separate bounded pilot with real TRAIN/VAL dimensions, time warmup, data fetch, episode materialization/hash checks, H2D, Entry online/teacher, Exit online/teacher, backward, optimizer/EMA, checkpoint I/O and full VAL separately. Preserve the actual FP32/B8/worker-zero source-bound recipe and every guard. No GPU purchase or pilot is authorized by this report.
- Use a brief scheduled profiler trace only if needed; it adds overhead and shape/stack recording can retain tensors. Compare unprofiled monotonic windows before/after. [PyTorch 2.6 profiler](https://docs.pytorch.org/docs/2.6/profiler.html).
- Rank opportunities by measured wall-time share: F3 redundant checks/scales first if significant; immutable CPU gather/copy work only if it dominates; architectural changes only afterward. Do not cache learned recurrent states across different episode origins or training steps. The frozen teacher changes at epoch boundaries, so an all-TRAIN target cache consumed only once per epoch is not an automatic saving and may add gigabytes of I/O.
- CUDA Transformer activation checkpointing is already disabled under the current bounded policy, while CPU uses recomputation. Do not propose “turn checkpointing off on CUDA” as new work. Proof: `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:142`.
- Preserve all population rows and normalize chunk losses over the same total valid cells; compare online/teacher outputs, masks, every active gradient, optimizer/EMA/RNG/resume state and final decisions. Mathematical equivalence alone is not bit-identical FP32 evidence. Never trade away hard finite/safety checks for a microbenchmark result.
- Measure peak host/VRAM, restart/preflight overhead, useful steps per second and full-VAL cost. Derive epoch/seed ETA from those measurements. Do not assume linear batch/GPU scaling, price a full campaign from marketing FLOPS, or add CPU workers/AMP/TF32/compile/DDP to the current run. No measured speedup or total ETA is claimed here.

## 7. Coverage and exclusions

Static inventory in staging counted 72 contract files, seven model files, 19 feature files, 67 GX1 script files, 14 execution files, three replay files, 209 test files, 12 top-level script files and two guard files. This is a source-tree inventory, not a claim to have reviewed every line. The trainer is 14,881 lines and the hybrid model 3,226 lines. Concurrent agents may subsequently change non-model files.

| Area | Reviewed in depth | Not established |
| --- | --- | --- |
| Model | Entry local/MTF fusion, Q/token/aux heads, episode/incremental structure, shared projections, retired modules | Full normalization/constructor validation audit, total active parameter count, real input/output parity |
| Trainer | Q target/Exit batching and gradient reinjection, task weighting, movement reductions, candidate teacher boundaries, VAL policy selection, existing profile hooks | Full 14,881-line review, complete optimizer/resume/export failure analysis, fresh CPU/GPU test results |
| Contracts | Production architecture, fitted-Q, task objective/weighting, episode terminal, relevant lifecycle/target boundaries, usefulness/challenger interfaces | Full economics implementation, calibration/portfolio acceptance, TEST sizing feasibility; parent/prior review owns those |
| Features | Inventory, specialist routing owner references, native-clock and alias/normalization ownership contracts | Fresh numerical fidelity audit of structure, SMC, trend, volatility/squeeze, momentum, session, geometry or candles; causal replay of every feature |
| Tests | Relevant test source; parent-reported 42/42 model-shape cases passed on capped WSL CPU/4G with synthetic fixtures | Independently inspected result artifact, other unreported suites, real-market usefulness or quality evidence |
| Runtime/replay | Inventory and the model's incremental path only | Broker/OANDA modules, collector/paper/live paths, full replay/serve integration; not launched or authorized |
| Host/data/artifacts | No inspection beyond explicitly named context reports and source snapshot comparison | Any remote process/temperature/clock status, checkpoint progress, raw TRAIN/VAL audit or TEST content |

Source agreement is not proof of benefit or exhaustive absence of bugs. No fresh real-data family-ablation result, meaningful candidate learning curve, portfolio replay, model-capacity experiment or cost benchmark was produced.

### Model-source identity checked in this pass

The following staging files were SHA-256 hashed and byte-compared successfully with the original read-only snapshot at `/private/tmp/gx1-review-d4d459c1.oPjcd2`. This verifies these named source bytes, not every concurrent staging change or the running WSL environment.

| Source | SHA-256 |
| --- | --- |
| `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:1` | `d6e8a2657f8bc000c357fe87b4019db759c2452ebf925203171bd7a507fc256f` |
| `/home/andre2/src/GX1_ENGINE/gx1/models/entry_v10/entry_v10_ctx_train_v3.py:1` | `fb4aa9744518ebda31e633145bd60463d53b532e639f09cc2f1c8d72247dcf7c` |
| `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_fitted_q_v1.py:1` | `ce77ca0d566c16a0f884a5098fb1ac4f9e6ecb848d9c3594ae6e3edc877e4fd1` |
| `/home/andre2/src/GX1_ENGINE/gx1/contracts/unified_exit_fitted_q_v1.py:1` | `7b55c1288b1808484260ee5155da846fb59f68802a216ff7f0f3034c7844d985` |
| `/home/andre2/src/GX1_ENGINE/gx1/contracts/entry_model_native_training_objective_v1.py:1` | `18c9f242fa1815f06a0b9cde0e372f72dd97459366c1edd493093843fc4c1211` |

**Bottom line:** the eight families are connected, but the benefit of dense multi-stage cooperation is unproved. Measure meaningful chronological evidence with the current reference, complete the existing usefulness path, remove confirmed dead state only later, and let a small controlled baseline comparison decide whether a simpler successor is preferable.

## Parent integration note: historical-status compaction

The parent additionally reports Windows Time is synchronized and plans no material architecture changes before benefit evidence. Those are parent-supplied operational facts/intent, not independent host verification here. Windows synchronization alone is not proof of stable WSL offset, monotonic event ordering or fresh guard telemetry.

Compacting duplicated history in root AGENTS/README is reasonable, with these source-audit cautions:

- Preserve binding scope/precedence and every still-applicable operational prohibition, evidence class, canonical command/owner, source/recipe identity requirement and TEST boundary. Do not accidentally turn a historical allowance into present launch authority or remove an obligation carried only in one root document.
- Keep one short current handoff pointing to executable status and machine-readable state. Do not transplant checkpoint numbers, feature widths, old thermal limits or the old 13-month VAL description into a new “current” summary. Record 42 CPU passes as synthetic-fixture mechanics, never “all families useful” or “model ready.”
- Move historical evidence to clearly historical, linked sections/documents if approved rather than rewriting its original result/authorization meaning. Preserve anchors/references and provenance used by incident reviews, tests and operators. Removing a documentary artifact reference is not authorization to delete that artifact or proof it is unreachable.
- Verify the actual recipe/source closure and source-hygiene/fingerprint consumers before assuming Markdown is outside identity checks. Even if a document is outside learning bytes, a dirty tracked file can affect clean-source admission. Never silently rebind the active session after a documentation edit.
- Run scoped stale-reference/anchor/status-literal checks and relevant documentation/handover tests through the parent's capped workflow. Keep the current model/trainer byte identities unchanged; this review proposes no architecture replacement.
