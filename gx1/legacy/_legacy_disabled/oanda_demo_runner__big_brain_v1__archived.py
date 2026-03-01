"""
ARCHIVED: BigBrain V1 removed from runtime; non-canonical.

This file preserves the BigBrain V1 integration that was removed from
`gx1/execution/oanda_demo_runner.py` to enforce CANONICAL_TRUTH_SIGNAL_ONLY_V1.
"""

# Optional Big Brain V1 imports (archived)
try:
    from gx1.big_brain.v1.runtime_v1 import BigBrainV1Runtime  # type: ignore[reportMissingImports]
    from gx1.big_brain.v1.entry_gater import BigBrainV1EntryGater, load_entry_gater, EntryAction  # type: ignore[reportMissingImports]
    BIG_BRAIN_AVAILABLE = True
except ImportError:
    BigBrainV1Runtime = None
    BigBrainV1EntryGater = None
    load_entry_gater = None
    EntryAction = None
    BIG_BRAIN_AVAILABLE = False

# Archived OpenPos field for BigBrain-adjusted exits
"""
@dataclass
class OpenPos:
    ...
    bb_exit: dict | None = None  # Big Brain V1 adjusted exit parameters (if available)
"""

# Archived TickWatcher adjustments (BE/soft-stop shaped by bb_exit)
"""
be_cfg = self.cfg.get("be", {})
be_activate_at_bps = int(pos.bb_exit.get("be_trigger_bps_adj", be_cfg.get("activate_at_bps", 50))) if pos.bb_exit else int(be_cfg.get("activate_at_bps", 50))
...
soft_stop_bps = int(pos.bb_exit.get("soft_stop_bps_adj", self.cfg.get("soft_stop_bps", 0))) if pos.bb_exit else int(self.cfg.get("soft_stop_bps", 0))
"""

# Archived BigBrain V1 runtime + entry-gater initialization
"""
bb_v1_config = self.policy.get("big_brain_v1", {})
bb_v1_enabled = bb_v1_config.get("enabled", False)
if bb_v1_enabled and BIG_BRAIN_AVAILABLE:
    model_path = Path(bb_v1_config.get("model_path", "models/BIG_BRAIN_V1/model.pt"))
    meta_path = Path(bb_v1_config.get("meta_path", "models/BIG_BRAIN_V1/meta.json"))
    ...
    self.big_brain_v1 = BigBrainV1Runtime(model_path=model_path, meta_path=meta_path)
    self.big_brain_v1.load()
elif bb_v1_enabled and not BIG_BRAIN_AVAILABLE:
    log.warning("[BIG_BRAIN_V1] Big Brain V1 is enabled in policy but modules are not available. Disabling Big Brain V1.")
else:
    log.info("[BIG_BRAIN_V1] Big Brain V1 shaping disabled in policy")

bb_v1_entry_gates_config = bb_v1_config.get("entry_gates", {})
bb_v1_entry_gates_enabled = bb_v1_entry_gates_config.get("enabled", False)
if bb_v1_entry_gates_enabled:
    if not BIG_BRAIN_AVAILABLE:
        log.warning("[BIG_BRAIN_V1_ENTRY] Entry gating enabled but Big Brain modules not available. Disabling entry gating.")
        self.big_brain_v1_entry_gater = None
    else:
        entry_gates_config_path = Path(bb_v1_entry_gates_config.get("config_path", "gx1/configs/big_brain_v1_entry_gates.yaml"))
        self.big_brain_v1_entry_gater = load_entry_gater(entry_gates_config_path)
        log.info("[BIG_BRAIN_V1_ENTRY] Entry gater loaded from %s", entry_gates_config_path)
"""

# Archived BigBrain V1 exit-shaping block (risk shaping, asymmetry, hysteresis, TP2/TP3 extension)
"""
# Big Brain V1 – Step 4: Exit-Aware Risk Shaping
# ...
be_trigger_bps_adjusted = max(int(be_trigger_bps_original * (1.0 - 0.20 * brain_risk)), int(be_trigger_bps_original * 0.10))
tp_bps_adjusted = max(int(tp_bps_original * (1.0 - 0.10 * brain_risk)), int(tp_bps_original * 0.005))
soft_stop_bps_adjusted = ...

# Step 5: Trend-Aware TP/SL Asymmetry
# tp_bps_asym/sl_bps_asym/soft_stop_bps_asym adjusted by trend and side

# Step 6: Exit Hysterese Dynamics
# trade_obj.extra["bb_exit_hyst"] = {...}

# Step 7: Adaptive TP2/TP3 Extension
# trade_obj.extra["bb_exit_tp_ext"] = {...}
# trade_obj.extra["bb_exit"] / ["bb_exit_asym"] populated for logging
"""

# Archived BigBrain V1 warmup handling in replay
"""
warmup_prices_path = bb_v1_config.get("warmup_prices_path")
lookback_bars = self.big_brain_v1.lookback
...  # load parquet/CSV, filter to replay period, feed self.big_brain_v1.feed_warmup(...)
# fallback: use first lookback_bars from replay file, log BIG_BRAIN_V1 warmup messages
"""
