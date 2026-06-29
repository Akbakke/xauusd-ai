"""Fail-closed guards for legacy Entry control surfaces."""
from __future__ import annotations

import os
import sys


LEGACY_RESEARCH_ACK_ENV = "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH"
LEGACY_RESEARCH_ACK_VALUE = "20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH"
V12_DATA_MAINTENANCE_ACK_ENV = "GX1_ALLOW_V12_DATA_MAINTENANCE"
V12_DATA_MAINTENANCE_ACK_VALUE = "20260627_ALLOW_V12_DATA_MAINTENANCE"
OANDA_ORDER_ACK_ENV = "GX1_ALLOW_OANDA_ORDER_PLACEMENT"
OANDA_ORDER_ACK_VALUE = "20260627_ALLOW_OANDA_ORDER_PLACEMENT"


def enforce_legacy_entry_research_ack(surface: str) -> None:
    """Require an explicit override before old Entry runner/replay paths run."""
    if os.environ.get(LEGACY_RESEARCH_ACK_ENV) == LEGACY_RESEARCH_ACK_VALUE:
        print(
            f"[entry-next-edge] legacy override acknowledged for {surface}: {LEGACY_RESEARCH_ACK_VALUE}",
            file=sys.stderr,
        )
        return

    print(
        "\n".join(
            [
                f"[ABORT] Active Entry next-edge plan blocks {surface}.",
                "        Current path is Entry foundation seq146 cleanup/audit/smoke-readiness.",
                "        Use:",
                "          scripts/entry_next_edge_control.sh verify",
                "          scripts/entry_next_edge_control.sh selftest",
                "          scripts/entry_next_edge_control.sh foundation-guardrails",
                "          scripts/entry_next_edge_control.sh worktree-hygiene",
                "          scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run",
                "          scripts/entry_next_edge_control.sh materialize-smoke",
                "          scripts/entry_next_edge_control.sh train-readiness",
                "        Override only with:",
                f"          {LEGACY_RESEARCH_ACK_ENV}={LEGACY_RESEARCH_ACK_VALUE} <command>",
            ]
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


def enforce_v12_data_maintenance_ack(surface: str) -> None:
    """Require an explicit override before direct V12 data-maintenance entrypoints run."""
    if os.environ.get(V12_DATA_MAINTENANCE_ACK_ENV) == V12_DATA_MAINTENANCE_ACK_VALUE:
        print(
            f"[entry-next-edge] V12 data-maintenance override acknowledged for {surface}: "
            f"{V12_DATA_MAINTENANCE_ACK_VALUE}",
            file=sys.stderr,
        )
        return

    print(
        "\n".join(
            [
                f"[ABORT] Active Entry next-edge plan blocks {surface}.",
                "        Current path is Entry foundation seq146 cleanup/audit/smoke-readiness.",
                "        Direct data-maintenance entrypoints can mutate the tape/prebuilts",
                "        used by foundation validation; run them only deliberately.",
                "        Use:",
                "          scripts/entry_next_edge_control.sh verify",
                "          scripts/entry_next_edge_control.sh selftest",
                "          scripts/entry_next_edge_control.sh foundation-guardrails",
                "          scripts/entry_next_edge_control.sh worktree-hygiene",
                "          scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run",
                "          scripts/entry_next_edge_control.sh materialize-smoke",
                "          scripts/entry_next_edge_control.sh train-readiness",
                "        Override only with:",
                f"          {V12_DATA_MAINTENANCE_ACK_ENV}={V12_DATA_MAINTENANCE_ACK_VALUE} <command>",
            ]
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


def enforce_oanda_order_placement_ack(surface: str) -> None:
    """Require an explicit override before any OANDA order mutation is submitted."""
    if os.environ.get(OANDA_ORDER_ACK_ENV) == OANDA_ORDER_ACK_VALUE:
        print(
            f"[entry-next-edge] OANDA order-placement override acknowledged for {surface}: "
            f"{OANDA_ORDER_ACK_VALUE}",
            file=sys.stderr,
        )
        return

    print(
        "\n".join(
            [
                f"[ABORT] Active Entry next-edge plan blocks OANDA order placement via {surface}.",
                "        Current path is Entry foundation seq146 cleanup/audit/smoke-readiness.",
                "        No live/practice order mutation is allowed before reviewed foundation training and replay gates.",
                "        Use:",
                "          scripts/entry_next_edge_control.sh verify",
                "          scripts/entry_next_edge_control.sh selftest",
                "          scripts/entry_next_edge_control.sh foundation-guardrails",
                "          scripts/entry_next_edge_control.sh worktree-hygiene",
                "          scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run",
                "          scripts/entry_next_edge_control.sh materialize-smoke",
                "          scripts/entry_next_edge_control.sh train-readiness",
                "        Override only with:",
                f"          {OANDA_ORDER_ACK_ENV}={OANDA_ORDER_ACK_VALUE} <command>",
            ]
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)
