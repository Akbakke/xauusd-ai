#!/usr/bin/env python3
"""
integrate_exit_critic_runtime_v1.py

Runtime-integrasjon for ExitCritic V1.
Denne modulen skal brukes internt i SNIPER exit-løpet for å bestemme:

- EXIT_NOW        (full close)
- SCALP_PROFIT    (delvis close + trail resten)
- HOLD            (ingen endring, la exit-router bestemme videre)

Bruk:
    Fra exit-engine (f.eks exit_manager.py):

        from gx1.rl.exit_critic.integrate_exit_critic_runtime_v1 import ExitCriticRuntimeV1

        critic = ExitCriticRuntimeV1(
            model_path="models/exit_critic/exit_critic_xgb_v1.json",
            metadata_path="models/exit_critic/exit_critic_xgb_v1.meta.json"
        )

        decision = critic.evaluate(trade_snapshot_dict)

        if decision == "EXIT_NOW":  -> exit market
        if decision == "SCALP_PROFIT": -> partial exit + trail
        if decision == "HOLD": -> continue normal exit engine
"""

import json
import numpy as np
import xgboost as xgb
from typing import Dict, Any, List


ACTION_EXIT_NOW = "EXIT_NOW"
ACTION_SCALP_PROFIT = "SCALP_PROFIT"
ACTION_HOLD = "HOLD"


class ExitCriticRuntimeV1:
    """
    Laster trenet ExitCritic og predikerer exit-beslutninger basert på
    snapshot av aktiv trade.

    Input-format (trade_snapshot):
    {
        "p_long": float,
        "atr_bps": float,
        "current_pnl_bps": float,
        "mfe_bps": float,
        "mae_bps": float,
        "bars_held": int,
        "volatility_bps": float,
        "trend_score": float,
        "vol_regime": int,
        "trend_regime": int,
        ...
        feat_XXX: float  <── alle features bygges av exit_dataset
    }
    """

    def __init__(self, model_path: str, metadata_path: str, exit_now_threshold: float = 0.60,
                 scalp_threshold: float = 0.35):
        """
        exit_now_threshold:
            sannsynlighet for label==EXIT_NOW der vi tvangseksiterer

        scalp_threshold:
            sannsynlighet for label==SCALP_PROFIT der vi gjør halv-exit
        """

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        self.feature_cols: List[str] = meta["feature_cols"]
        self.exit_now_threshold = exit_now_threshold
        self.scalp_threshold = scalp_threshold

        print(f"[ExitCriticRuntime] Loaded model with {len(self.feature_cols)} features.")
        print(f"[ExitCriticRuntime] EXIT_NOW threshold = {exit_now_threshold}")
        print(f"[ExitCriticRuntime] SCALP_PROFIT threshold = {scalp_threshold}")

    def evaluate(self, snapshot: Dict[str, Any]) -> str:
        """
        Returnerer ACTION_* string basert på sannsynlighet.

        Merk:
        - snapshot må inneholde feature_cols
        - manglede nøkler blir fylt med 0 (defensive)
        """

        row = []
        for col in self.feature_cols:
            val = snapshot.get(col, 0.0)  # fallback for sikkerhetsnett
            row.append(val)

        X = np.array(row).reshape(1, -1)

        # probas: [hold_prob, exit_prob] eller multi-class softprob
        proba = self.model.predict_proba(X)[0]

        # to-klasser (EXIT_NOW=1 / HOLD=0)
        if len(proba) == 2:  
            p_exit = float(proba[1])
            if p_exit >= self.exit_now_threshold:
                return ACTION_EXIT_NOW
            elif p_exit >= self.scalp_threshold:
                return ACTION_SCALP_PROFIT
            else:
                return ACTION_HOLD

        # multi-class (0=HOLD,1=SCALP,2=EXIT eksempel)
        label = int(np.argmax(proba))

        if label == 2 and max(proba) >= self.exit_now_threshold:
            return ACTION_EXIT_NOW
        elif label == 1 and max(proba) >= self.scalp_threshold:
            return ACTION_SCALP_PROFIT
        else:
            return ACTION_HOLD
