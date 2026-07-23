#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Execution Smoke Test Trade Journal Schema.

Verifies that a "test trade" JSON has test_mode=true, execution_events
are appended in correct order, and client_ext_id format is correct.
"""
import math
import tempfile
import unittest
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.monitoring.trade_journal import TradeJournal
from tests.model_native_sizing_support import unverified_learned_sizing_authority
from tests.model_native_offline_rl_support import offline_rl_evidence
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)


class TestExecSmokeTradeJournalSchema(unittest.TestCase):
    """Test execution smoke test trade journal schema."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.journal_dir = self.temp_dir / "journal"
        self.journal_dir.mkdir(parents=True)
        
        # Create minimal run header
        self.run_header = {
            "timestamp": "2025-01-01T00:00:00Z",
            "run_tag": "EXEC_SMOKE_TEST",
            "meta": {
                "role": "TEST",
                "test_mode": True,
            },
            "artifacts": {},
        }
        
        self.trade_journal = TradeJournal(
            run_dir=self.temp_dir,
            run_tag="EXEC_SMOKE_TEST",
            header=self.run_header,
            enabled=True,
        )
        
        self.trade_id = "EXEC-SMOKE-1234567890-abcdefgh"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_execution_smoke_snapshot_has_no_direction_substitute(self):
        """A broker smoke record may be explicit, but cannot fake model evidence."""
        self.trade_journal.log_entry_snapshot(
            trade_id=self.trade_id,
            entry_time="2025-01-01T00:00:00Z",
            instrument="XAU_USD",
            side="long",
            entry_price=2000.0,
            model_evidence={},
        )

        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        entry = journal_data["entry_snapshot"]
        self.assertEqual(entry["model_evidence"], {})
        self.assertNotIn("entry_score", entry)
        self.assertNotIn("entry_filters_passed", entry)
        self.assertNotIn("test_mode", entry)
    
    def test_execution_events_order(self):
        """Test that execution events are appended in correct order."""
        # Log ORDER_SUBMITTED
        self.trade_journal.log_order_submitted(
            trade_id=self.trade_id,
            instrument="XAU_USD",
            side="long",
            units=1,
            order_type="MARKET",
            client_ext_id="GX1:EXEC_SMOKE:TEST:123",
        )
        
        # Log ORDER_FILLED
        self.trade_journal.log_order_filled(
            trade_id=self.trade_id,
            oanda_order_id="12345",
            fill_price=2000.0,
            fill_units=1,
        )
        
        # Log TRADE_OPENED_OANDA
        self.trade_journal.log_oanda_trade_update(
            trade_id=self.trade_id,
            event_type="TRADE_OPENED_OANDA",
            oanda_trade_id="67890",
        )
        
        # Verify order
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        events = journal_data.get("execution_events", [])
        
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["event_type"], "ORDER_SUBMITTED")
        self.assertEqual(events[1]["event_type"], "ORDER_FILLED")
        self.assertEqual(events[2]["event_type"], "TRADE_OPENED_OANDA")
    
    def test_client_ext_id_format(self):
        """Test that client_ext_id format is correct."""
        run_tag = "EXEC_SMOKE_TEST"
        trade_id = "EXEC-SMOKE-1234567890-abcdefgh"
        client_ext_id = f"GX1:EXEC_SMOKE:{run_tag}:{trade_id}"
        
        # Verify format
        self.assertTrue(client_ext_id.startswith("GX1:EXEC_SMOKE:"))
        self.assertIn(run_tag, client_ext_id)
        self.assertIn(trade_id, client_ext_id)
        
        # Log ORDER_SUBMITTED with client_ext_id
        self.trade_journal.log_order_submitted(
            trade_id=trade_id,
            instrument="XAU_USD",
            side="long",
            units=1,
            order_type="MARKET",
            client_ext_id=client_ext_id,
        )
        
        # Verify in journal
        journal_data = self.trade_journal._get_trade_journal(trade_id)
        events = journal_data.get("execution_events", [])
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "ORDER_SUBMITTED")
        self.assertEqual(events[0]["client_extensions"]["id"], client_ext_id)

    def test_journal_rejects_unadopted_model_native_sizing_evidence(self):
        """Complete Entry evidence cannot invent executable sizing authority."""
        logits = [2.0, 0.2, -1.0]

        def softmax(values):
            shifted = [value - max(values) for value in values]
            exp_values = [math.exp(value) for value in shifted]
            total = sum(exp_values)
            return [value / total for value in exp_values]

        def sigmoid(value):
            return 1.0 / (1.0 + math.exp(-value))

        def logit(value):
            return math.log(value / (1.0 - value))

        direction_probs = softmax(logits)
        public_logits = [2.0, -1.0]
        public_probs = softmax(public_logits)
        side_logits = [1.0, -0.5]
        side_probs = softmax(side_logits)
        side_bad_path_logits = [-2.0, 0.2]
        side_validity_logits = [1.2, -0.4]
        mtf_logits = [1.0, -0.2, 0.1]
        rail_logits = [0.1, 0.2, -0.1, 0.3, -0.2, 0.4]
        tf_logit = -0.2
        size_logit = 0.4
        path_log_var = -0.3
        evidence = {
            "decision_ts": "2026-07-08T17:55:00Z",
            "runtime_evidence_schema_version": MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
            "session_id": 3,
            "session": "US",
            "entry_vol_regime_id": 3,
            "entry_vol_regime": "HIGH",
            "entry_atr_bucket": 4,
            "entry_spread_bucket": 1,
            "entry_h4_trend_sign_cat": 0,
            "entry_trend_regime_id": 2,
            "entry_trend_regime": "TREND_UP",
            "decision_available_ts": "2026-07-08T18:00:00Z",
            "entry_signal_latency_sec": 0.0,
            "context_cutoff_ts": "2026-07-08T17:55:00Z",
            "context_age_m5_bars": 0,
            "raw_direction_logits": [2.185, 0.345, -1.15],
            "direction_logits": logits,
            "direction_probs": direction_probs,
            "model_direction_index": 0,
            "model_direction": "LONG",
            "selected_side": 0,
            "public_trade_flat_decision_logits": public_logits,
            "public_trade_flat_decision_probs": public_probs,
            "public_trade_flat_decision_index": 0,
            "public_trade_flat_decision": "TRADE",
            "model_native_logits": [0.5, -0.25, 0.1],
            "path_quality_raw": 1.25,
            "path_quality": 1.25,
            "path_quality_pred": 1.25,
            "mfe_first_n": 7.0,
            "mfe_first_n_pred": 7.0,
            "tradable_logit": logit(0.81),
            "tradable_prob": 0.81,
            "trade_logit": 0.7,
            "bad_path_logit_raw": logit(0.09),
            "bad_path_logit": logit(0.09),
            "bad_path_prob": 0.09,
            "clean_edge_logit": logit(0.76),
            "clean_edge_prob": 0.76,
            "survival_logit": logit(0.68),
            "survival_prob": 0.68,
            "dip_pred": [0.0] * 18,
            "forecast_pred": [0.0] * 4,
            "timing_pred": [0.0] * 12,
            "tail_risk_pred": [0.0] * 6,
            "vol_forecast_pred": [0.0] * 3,
            **offline_rl_evidence(),
            "p_trade": public_probs[0],
            "p_flat_hier": public_probs[1],
            "atr_bps": 12.5,
            "tf_agreement_logit": tf_logit,
            "tf_agreement_pred": 1.0 / (1.0 + math.exp(-tf_logit)),
            "path_quality_log_var": path_log_var,
            "path_quality_std": math.exp(0.5 * path_log_var),
            "position_size_logit": size_logit,
            "position_size_pred": sigmoid(size_logit),
            "sizing_authority_contract": unverified_learned_sizing_authority(),
            "p_long_given_trade": side_probs[0],
            "p_short_given_trade": side_probs[1],
            "side_logits": side_logits,
            "side_probs": side_probs,
            "side_utility": [18.5, -7.0],
            "side_bad_path_logit": side_bad_path_logits,
            "long_bad_path_prob": sigmoid(side_bad_path_logits[0]),
            "short_bad_path_prob": sigmoid(side_bad_path_logits[1]),
            "side_validity_logit": side_validity_logits,
            "long_validity_prob": sigmoid(side_validity_logits[0]),
            "short_validity_prob": sigmoid(side_validity_logits[1]),
            "side_mae": [4.0, 11.0],
            "mtf_dir_logits": mtf_logits,
            "mtf_dir_probs": softmax(mtf_logits),
            "mtf_trend_evidence": 0.69,
            "specialist_names": [
                "structure_swing_encoder",
                "smc_liquidity_encoder",
                "trend_ema_encoder",
                "vol_compression_encoder",
                "momentum_flow_encoder",
                "session_regime_encoder",
                "chart_geometry_encoder",
                "price_action_candle_encoder",
            ],
            "specialist_gate": [0.125] * 8,
            "trendline_rail_logits": rail_logits,
            "trendline_rail_probs": [sigmoid(value) for value in rail_logits],
            "geometry_channel_edge_pressure": 0.42,
            "geometry_rising_support_rail_long_pressure": 0.81,
            "geometry_rising_support_rail_short_trap_pressure": 0.77,
            "geometry_falling_resistance_rail_short_pressure": 0.02,
            "geometry_falling_resistance_rail_long_trap_pressure": 0.03,
            "calibration_version": "dircal_v2",
            "direction_calibration_enabled": True,
            "direction_calibration_temperature": 1.15,
            "direction_calibration_bias": [0.1, -0.1, 0.0],
            "path_calibration_enabled": True,
            "path_calibration": {
                "enabled": True,
                "version": "path-cal-v1",
                "path_quality_scale": 1.0,
                "path_quality_shift": 0.0,
                "bad_path_temperature": 1.0,
                "bad_path_bias": 0.0,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "TRADE_JOURNAL_ENTRY"):
            self.trade_journal.log_entry_snapshot(
                trade_id=self.trade_id,
                entry_time="2026-07-08T18:00:00Z",
                instrument="XAU_USD",
                side="long",
                entry_price=2360.2,
                model_evidence=evidence,
                entry_bid=2360.0,
                entry_ask=2360.2,
                entry_spread_bps=0.85,
                session="US",
                model_policy=MODEL_NATIVE_RUNTIME_POLICY,
                execution_checks=["fresh_quote", "learned_sizing_proof_bound"],
                capacity_units=1,
                reference_pre_round_units=1.0,
                pre_round_units=1.0,
                units=1,
                applied_size_multiplier=1.0,
                sizing_application={},
                atr_bps=12.5,
            )
