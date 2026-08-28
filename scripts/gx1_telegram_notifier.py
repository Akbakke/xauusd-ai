#!/usr/bin/env python3
"""GX1 Telegram push-notifier — READ-ONLY trade alerts to your phone.

Watches the live paper journal + completed-trade records and pushes to Telegram:
  • 📈 Trade OPENED  (side · units · entry)
  • ✅/🔴 Trade CLOSED (realized bps · MFE/MAE · exit reason)
  • 📊 Daily summary  (first tick of a new UTC day: prior-day trades, realized bps, win-rate)

Decoupled from the live chain: it only READS files (same sources as the dashboard) —
never touches cement/contract/prebuilts/processes. Push-only (no command interface).
Token + chat_id come from .env (GX1_TELEGRAM_BOT_TOKEN / GX1_TELEGRAM_CHAT_ID), which
is gitignored — secrets never enter the repo. Stdlib only (urllib), no new deps.

Run:  /home/andre2/src/GX1_ENGINE/.venv/bin/python scripts/gx1_telegram_notifier.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from gx1.contracts.gx1_scope_v1 import require_offline_scope
RUNS = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
JOURNAL_GLOB = str(RUNS / "v12_paper_journal_*.jsonl")
TRADES_DIR = RUNS / "trade_journal" / "trades"
STATE = RUNS / "telegram_notifier_state.json"
POLL = int(os.environ.get("GX1_TG_POLL", "30"))


def _load_env():
    if not os.environ.get("GX1_TELEGRAM_BOT_TOKEN"):
        envf = REPO / ".env"
        if envf.is_file():
            for line in envf.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


TOKEN: str | None = None
CHAT: str | None = None


def send(text: str) -> bool:
    if not TOKEN or not CHAT:
        return False
    data = urllib.parse.urlencode({
        "chat_id": CHAT, "parse_mode": "HTML", "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    try:
        with urllib.request.urlopen(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=15) as r:
            return bool(json.loads(r.read()).get("ok"))
    except Exception as e:
        print(f"[notifier] send failed: {e}", flush=True)
        return False


def _latest_journal():
    fs = sorted(glob.glob(JOURNAL_GLOB))
    return fs[-1] if fs else None


def _tail_last_json(path, nbytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - nbytes))
            lines = f.read().decode("utf-8", "ignore").splitlines()
        for ln in reversed(lines):
            ln = ln.strip()
            if ln:
                try:
                    return json.loads(ln)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def current_open() -> dict:
    """id -> {side, units, entry_ts, bars} from the latest journal line."""
    jp = _latest_journal()
    d = _tail_last_json(jp) if jp else None
    out = {}
    if d:
        for t in (d.get("open_trade_records") or []):
            out[str(t.get("trade_id"))] = {
                "side": t.get("side"), "units": t.get("units"),
                "entry_ts": t.get("entry_ts"), "bars": t.get("bars_in_trade"),
            }
    return out


def read_trade(tid: str):
    p = TRADES_DIR / f"{tid}.json"
    if not p.is_file():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    es = d.get("exit_summary") or {}
    ent = d.get("entry_snapshot") or {}
    return {
        "side": ent.get("side"), "units": ent.get("units"), "entry_px": ent.get("entry_price"),
        "pnl": es.get("realized_pnl_bps"), "mfe": es.get("max_mfe_bps"),
        "mae": es.get("max_mae_bps"), "reason": es.get("exit_reason"),
        "exit_time": es.get("exit_time"), "exit_px": es.get("exit_price"),
    }


def daily_summary(date_str: str):
    n = 0
    bps = 0.0
    wins = 0
    now = time.time()
    for f in glob.glob(str(TRADES_DIR / "*.json")):
        try:
            if now - os.path.getmtime(f) > 3 * 24 * 3600:
                continue
            es = (json.load(open(f)).get("exit_summary") or {})
        except Exception:
            continue
        if str(es.get("exit_time", "")).startswith(date_str):
            n += 1
            p = es.get("realized_pnl_bps") or 0
            bps += p
            wins += 1 if p > 0 else 0
    return n, round(bps, 1), wins


def _load_state():
    if STATE.is_file():
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {}


def _save_state(s):
    tmp = str(STATE) + ".tmp"
    json.dump(s, open(tmp, "w"))
    os.replace(tmp, STATE)


def main() -> int:
    # Frozen offline scope rejects this legacy notifier before it may read a
    # secret, inspect journals, write state, or contact Telegram.
    require_offline_scope("telegram_notifier")
    global TOKEN, CHAT
    _load_env()
    TOKEN = os.environ.get("GX1_TELEGRAM_BOT_TOKEN")
    CHAT = os.environ.get("GX1_TELEGRAM_CHAT_ID")
    if not TOKEN or not CHAT:
        print("FATAL: GX1_TELEGRAM_BOT_TOKEN / GX1_TELEGRAM_CHAT_ID mangler i .env")
        return 1
    st = _load_state()
    notified_open = set(st.get("notified_open", []))
    last_summary = st.get("last_summary_date")
    seeded = st.get("seeded", False)
    print(f"[notifier] startet (poll {POLL}s, seeded={seeded})", flush=True)
    while True:
        try:
            opens = current_open()
            cur = set(opens.keys())
            if not seeded:
                # first ever run: adopt current opens silently (don't spam pre-existing)
                notified_open = set(cur)
                seeded = True
                st.update(notified_open=list(notified_open), seeded=True)
                _save_state(st)
            else:
                # NEW opens
                for tid in cur - notified_open:
                    o = opens[tid]
                    t = read_trade(tid) or {}
                    px = t.get("entry_px")
                    pxs = f" @ {px}" if px else ""
                    send(f"📈 <b>Trade ÅPNET</b> #{tid}\n"
                         f"{(o.get('side') or '?').upper()} · {o.get('units')}u{pxs}")
                    notified_open.add(tid)
                # CLOSES (was open, now gone) — natural retry until the record has a P&L
                for tid in list(notified_open - cur):
                    t = read_trade(tid)
                    if t and t.get("pnl") is not None:
                        w = t["pnl"] >= 0
                        send(f"{'✅' if w else '🔴'} <b>Trade LUKKET</b> #{tid}\n"
                             f"{(t.get('side') or '?').upper()} · "
                             f"<b>{'+' if w else ''}{t['pnl']:.1f} bps</b>\n"
                             f"mfe {t.get('mfe', 0):.0f} / mae {t.get('mae', 0):.0f} · {t.get('reason')}")
                        notified_open.discard(tid)
                    # if record not yet written, keep tid → retry next tick
                st.update(notified_open=list(notified_open))
                _save_state(st)

            # DAILY summary on the first tick of a new UTC day (summarize prior day)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if last_summary != today:
                if last_summary is not None:  # never summarize on the very first run
                    y = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
                    n, bps, wins = daily_summary(y)
                    if n > 0:
                        send(f"📊 <b>Dagsoppsummering {y}</b>\n"
                             f"trades: {n} · realisert: {'+' if bps > 0 else ''}{bps} bps · "
                             f"vinnere: {wins}/{n}")
                last_summary = today
                st.update(last_summary_date=today)
                _save_state(st)
        except Exception as e:
            print(f"[notifier] loop error: {e}", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    import sys
    sys.exit(main())
