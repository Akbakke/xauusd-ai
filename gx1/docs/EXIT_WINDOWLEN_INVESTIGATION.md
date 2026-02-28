## EXIT Window Length Investigation

- **What is `window_len`?**  
  The number of bars (T) used to build the transformer exit input; shapes target `(B, T, D)` where `D` is the exit IO feature count.

- **Who decides?**  
  Model config (`exit_transformer_config.json`) is authoritative; policy YAML is a hint. We currently log both and report the effective value (prefer model config).

- **Effective T in practice**  
  Logged once per run via `[EXIT_WINDOWLEN_REPORT]` and `EXIT_WINDOWLEN_REPORT.json` in the run directory. Fields: policy_window_len, model_config_window_len, effective_window_len, model_input_dim, model_io_version, feature_names_hash, built_tensor_shape.

- **Mismatch handling (current behavior)**  
  No behavior change yet; report-only. Runtime currently uses whatever window_len the model config encodes. Any padding/truncation would occur inside feature prep/model input (to be reviewed when we wire transformer inference).

- **Risks: 8 vs 64**  
  - Short window (8): faster response, but may underfit long context and misalign with training if model trained on longer sequences.  
  - Long window (64): more context, higher latency/compute; mismatch with training shortens or pads improperly.

- **Suggested rule (not enforced yet)**  
  Treat model config as the contract; fail fast if policy and model disagree. Align training/eval/replay window_len before enabling strict enforcement.
