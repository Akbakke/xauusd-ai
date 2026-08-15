#!/usr/bin/env python3
"""GX1 "Jarvis" live dashboard — zero-dependency local status page.

A single self-contained, READ-ONLY view of the XAUUSD paper-trading stack:
the model's exact LONG/SHORT/FLAT argmax and learned diagnostics, data freshness,
stack health, execution state, proof-bound learned sizing, and today's tally.

ONE-TRUTH: it reads the SAME live sources the operator inspects by hand —
  • the paper journal (exact model direction, hierarchy, path and sizing evidence)
  • live process status via pgrep (runner / collector)
  • the exact launch-bound immutable live-tail admission, revalidated on read
  • the journaled sizing application (mode, calibrated fraction, capacity, units)
It NEVER writes to any live file and NEVER touches contracts or feature data
or any process (pgrep/proc reads only). Stdlib only — no new deps.

Run:   /home/andre2/src/GX1_ENGINE/.venv/bin/python scripts/gx1_dashboard.py [--port 8787] [--host 0.0.0.0]
Open:  http://localhost:8787   (from the Windows browser; WSL2 forwards localhost)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.models.entry_v10.direction_decision_contract import (  # noqa: E402
    MODEL_DIRECTION_ACTION_BY_INDEX,
    MODEL_DIRECTION_NAME_BY_INDEX,
)
from gx1.contracts.live_tail_publication_v1 import (  # noqa: E402
    require_newest_live_tail_runtime_authority,
)

RUNS = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
JOURNAL_GLOB = str(RUNS / "v12_paper_journal_*.jsonl")
WANTS = Path.home() / ".config/systemd/user/default.target.wants/gx1-paper-runner.service"
LAUNCH_STATE = REPO / "PROJECT_STATE_xau_direction_launch.json"
_count_cache: dict[str, tuple[float, dict]] = {}


# ── read-only helpers ───────────────────────────────────────────────────────
def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return []


def _latest_journal(max_age_seconds: float = 3600.0):
    """Return only a recently written journal; never present an old run as live."""
    files = glob.glob(JOURNAL_GLOB)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    if time.time() - os.path.getmtime(latest) > max_age_seconds:
        return None
    return latest


def _tail_lines(path, nbytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "ignore").splitlines()
    except Exception:
        return []


def _last_decision(path):
    for line in reversed(_tail_lines(path)):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        vd = d.get("v12_decision") or {}
        if isinstance(vd, dict) and not vd.get("stub"):
            return d
    return None


def _today_counts(path):
    try:
        mt = os.path.getmtime(path)
    except Exception:
        return {}
    cached = _count_cache.get(path)
    if cached and cached[0] == mt:
        return cached[1]
    counts = {"SKIP": 0, "TAKE_LONG_NOW": 0, "TAKE_SHORT_NOW": 0, "total": 0}
    try:
        with open(path) as f:
            for line in f:
                if '"v12_decision"' not in line:
                    continue
                try:
                    vd = (json.loads(line).get("v12_decision") or {})
                except Exception:
                    continue
                if not isinstance(vd, dict) or vd.get("stub"):
                    continue
                a = vd.get("action", "?")
                counts[a] = counts.get(a, 0) + 1
                counts["total"] += 1
    except Exception:
        pass
    _count_cache[path] = (mt, counts)
    return counts


def _git_clean():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "status", "--short"],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and out.stdout.strip() == ""
    except Exception:
        return None


def _proc_cmdline_arg(pid, flag, default=None):
    """Read a CLI flag value (e.g. --max-trades 3) from a process cmdline."""
    if not pid:
        return default
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        parts = [p.decode() for p in parts if p]
        for i, p in enumerate(parts):
            if p == flag and i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        pass
    return default


def _parse_ts(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    # A naive timestamp (e.g. the daemon's "new cutoff: YYYY-MM-DD HH:MM:SS") is UTC by
    # convention here — pin it to UTC, never let astimezone() reinterpret it as local.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _live_tail_status() -> dict:
    """Revalidate the exact runtime admission; process presence is not proof."""
    try:
        state = json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "admitted": False,
            "status": f"BLOCK — launch-state unreadable: {type(exc).__name__}",
            "cutoff": None,
            "cutoff_lag_min": None,
        }
    if state.get("decision") != "ALLOW":
        return {
            "admitted": False,
            "status": "BLOCK — launch-state has no active live-tail authority",
            "cutoff": None,
            "cutoff_lag_min": None,
        }
    authority = state.get("new_entry_live_tail_authority")
    try:
        runtime = require_newest_live_tail_runtime_authority(authority)
        admission_path = Path(runtime["current_admission"]["path"])
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        child_path = Path(admission["child_publication"]["path"])
        child = json.loads(child_path.read_text(encoding="utf-8"))
        cutoff = child["timing"]["canonical_m5_cutoff_utc"]
        cutoff_dt = _parse_ts(cutoff)
        lag = (
            round(
                (datetime.now(timezone.utc) - cutoff_dt).total_seconds() / 60,
                1,
            )
            if cutoff_dt is not None
            else None
        )
        return {
            "admitted": True,
            "status": (
                "PASS — immutable admission "
                f"{runtime['current_admission']['pair_generation_id'][:12]}"
            ),
            "cutoff": cutoff,
            "cutoff_lag_min": lag,
        }
    except Exception as exc:
        return {
            "admitted": False,
            "status": f"BLOCK — live-tail authority invalid: {type(exc).__name__}",
            "cutoff": None,
            "cutoff_lag_min": None,
        }


TRADES_DIR = RUNS / "trade_journal" / "trades"
_trades_cache: dict = {}


def _trades_summary(open_records):
    """Open trades (from the journal) + last completed trade + today's realized P&L
    (from trade_journal/trades/<id>.json exit_summary). Cached by the trades-dir
    mtime so the per-trade JSONs are only re-read when a trade actually closes."""
    open_list = [{"side": t.get("side"), "units": t.get("units"),
                  "entry_ts": t.get("entry_ts"), "bars": t.get("bars_in_trade")}
                 for t in (open_records or [])]
    last, today_n, today_bps = None, 0, 0.0
    try:
        dmt = os.path.getmtime(TRADES_DIR)
    except Exception:
        dmt = None
    cached = _trades_cache.get("v")
    if cached and dmt is not None and cached[0] == dmt:
        last, today_n, today_bps = cached[1]
    else:
        import time as _t
        now = _t.time()
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent = sorted((os.path.getmtime(f), f) for f in glob.glob(str(TRADES_DIR / "*.json"))
                        if now - os.path.getmtime(f) < 30 * 3600)
        rows = []
        for _mt, f in recent:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            es = d.get("exit_summary") or {}
            ent = d.get("entry_snapshot") or {}
            if not es.get("exit_time"):
                continue
            rows.append({"id": d.get("trade_id"), "side": ent.get("side"), "units": ent.get("units"),
                         "pnl": es.get("realized_pnl_bps"), "mfe": es.get("max_mfe_bps"),
                         "mae": es.get("max_mae_bps"), "reason": es.get("exit_reason"),
                         "exit_time": es.get("exit_time")})
            if str(es.get("exit_time", "")).startswith(today_str):
                today_n += 1
                today_bps += (es.get("realized_pnl_bps") or 0)
        if rows:
            last = rows[-1]
        today_bps = round(today_bps, 1)
        _trades_cache["v"] = (dmt, (last, today_n, today_bps))
    return {"open": open_list, "last": last, "today_n": today_n, "today_bps": today_bps}


def build_status() -> dict:
    now = datetime.now(timezone.utc)
    rpid = (_pgrep("v12_paper_runner") or [None])[0]
    live_tail = _live_tail_status()

    jrnl = _latest_journal()
    dec = _last_decision(jrnl) if jrnl else None
    counts = _today_counts(jrnl) if jrnl else {}
    model = None
    market = None
    sizing = None
    dec_age = None
    if dec:
        vd = dec["v12_decision"]
        try:
            q_values = [float(value) for value in vd["entry_action_q_bps"]]
            if len(q_values) != 3 or not all(
                math.isfinite(value) for value in q_values
            ):
                raise ValueError(
                    "entry_action_q_bps must contain three finite values"
                )
            best_q = max(q_values)
            winners = [
                index for index, value in enumerate(q_values) if value == best_q
            ]
            if len(winners) != 1:
                raise ValueError("entry_action_q_bps has no unique argmax")
            direction_index = winners[0]
            direction = str(vd["model_direction"])
            if (
                direction != MODEL_DIRECTION_NAME_BY_INDEX[direction_index]
                or vd.get("model_direction_index") != direction_index
            ):
                raise ValueError("direction argmax/index/name parity failure")
            expected_action = MODEL_DIRECTION_ACTION_BY_INDEX[direction_index]
            if vd.get("action") != expected_action:
                raise ValueError("model action disagrees with direction argmax")
            required_scalars = {}
            for key in (
                "entry_action_q_margin_bps",
                "atr_bps",
                "position_size_pred",
            ):
                value = float(vd[key])
                if not math.isfinite(value):
                    raise ValueError(f"{key} is non-finite")
                required_scalars[key] = value
            model = {
                "contract_error": None,
                "direction": direction,
                "direction_index": direction_index,
                "action": expected_action,
                "q_long_bps": q_values[0],
                "q_short_bps": q_values[1],
                "q_flat_bps": q_values[2],
                "selected_q_bps": q_values[direction_index],
                "specialist_gate": vd["specialist_gate"],
                **required_scalars,
            }
        except (KeyError, TypeError, ValueError) as exc:
            model = {"contract_error": str(exc)}

        spread_value = dec.get("spread_bps")
        try:
            spread_value = float(spread_value)
            if not math.isfinite(spread_value) or spread_value < 0.0:
                raise ValueError
            spread_value = round(spread_value, 2)
        except (TypeError, ValueError):
            spread_value = None
        market = {
            "bid": dec.get("bid"), "ask": dec.get("ask"),
            "spread_bps": spread_value,
            "session": vd.get("session"),
            "action": vd.get("action"),
            "execution_status": dec.get("order_status") or dec.get("gate_reason"),
            "n_open_trades": dec.get("n_open_trades"),
        }
        sizing_application = dec.get("sizing_application")
        if isinstance(sizing_application, dict):
            sizing = {
                "mode": sizing_application.get("sizing_mode"),
                "fraction": sizing_application.get("calibrated_size_fraction"),
                "capacity_units": sizing_application.get("capacity_units"),
                "units": sizing_application.get("units"),
                "authorized_order": sizing_application.get("authorized_order"),
                "no_order_reason": sizing_application.get("no_order_reason"),
            }
        else:
            authority = ((vd.get("_v10_snapshot") or {}).get(
                "sizing_authority_contract"
            ) or {})
            sizing = {
                "mode": authority.get("adoption_mode"),
                "fraction": None,
                "capacity_units": None,
                "units": dec.get("units"),
                "authorized_order": False,
                "no_order_reason": dec.get("sizing_unavailable_evidence")
                or dec.get("sizing_no_order_reason"),
            }
        dt = _parse_ts(dec.get("ts_utc"))
        dec_age = round((now - dt).total_seconds() / 60, 1) if dt else None

    return {
        "now_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "health": {
            "runner": rpid is not None,
            "collector": bool(_pgrep("v12_oanda_data_collector")),
            "live_tail_publisher": live_tail["admitted"],
            "git_clean": _git_clean(),
            "auto_recover": WANTS.is_symlink(),
        },
        "model": model,
        "data": {
            "cutoff": live_tail["cutoff"],
            "cutoff_lag_min": live_tail["cutoff_lag_min"],
            "publisher_status": live_tail["status"],
            "last_decision_ts": dec.get("ts_utc") if dec else None,
            "last_decision_age_min": dec_age,
        },
        "market": market,
        "today": counts,
        "trades": _trades_summary(dec.get("open_trade_records") if dec else None),
        "op": {
            "direction": "exact_long_short_flat_argmax",
            "sizing": sizing,
            "max_trades": _proc_cmdline_arg(rpid, "--max-trades"),
        },
        "journal": os.path.basename(jrnl) if jrnl else None,
    }


PAGE = r"""<!doctype html><html lang="no"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GX1 · JARVIS — XAUUSD</title>
<style>
:root{--bg:#070b12;--card:#0e1521;--card2:#121b2b;--line:#1d2a40;--txt:#cfe3ff;--dim:#6c829e;
--grn:#27e8a7;--amb:#ffcf5c;--red:#ff5d6c;--cyan:#36d3ff;--vio:#a98bff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 700px at 70% -10%,#0d1830 0,var(--bg) 55%);
color:var(--txt);font:14px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;-webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto;padding:18px 18px 50px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}
.logo{font-weight:700;letter-spacing:2px;font-size:20px;color:#fff;text-shadow:0 0 18px rgba(54,211,255,.4)}
.logo b{color:var(--cyan)}.spacer{flex:1}
.clock{font-size:18px;color:#fff;letter-spacing:1px}.sub{color:var(--dim);font-size:12px}
.pulse{width:11px;height:11px;border-radius:50%;background:var(--grn);box-shadow:0 0 0 0 rgba(39,232,167,.6);animation:p 2s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(39,232,167,.55)}70%{box-shadow:0 0 0 9px rgba(39,232,167,0)}100%{box-shadow:0 0 0 0 rgba(39,232,167,0)}}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:16px;padding:16px 18px}
.card h3{margin:0 0 12px;font-size:11px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;font-weight:600}
.col12{grid-column:span 12}.col8{grid-column:span 8}.col4{grid-column:span 4}.col6{grid-column:span 6}.col3{grid-column:span 3}
@media(max-width:820px){.col8,.col4,.col6,.col3{grid-column:span 12}}
.hero .big{font-size:46px;font-weight:700;line-height:1.05;color:#fff;letter-spacing:.5px}
.hero .big.grn{color:var(--grn)}.hero .big.amb{color:var(--amb)}.hero .big.red{color:#ff8a93}
.gauge{position:relative;height:18px;border-radius:10px;background:#0a1018;border:1px solid var(--line);margin:16px 0 8px;overflow:hidden}
.gfill{position:absolute;left:0;top:0;bottom:0;border-radius:10px 0 0 10px;transition:width .5s ease,background .5s}
.rowflex{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-end}
.kv{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px dashed #16213400}
.kv .k{color:var(--dim)}.kv .v{color:#fff;font-weight:600}
.lean{height:10px;border-radius:6px;background:#13202f;display:flex;overflow:hidden;margin-top:6px}
.lean .l{background:linear-gradient(90deg,#1b8f6e,var(--grn))}.lean .s{background:linear-gradient(90deg,var(--red),#7a2330)}
.pills{display:flex;flex-wrap:wrap;gap:10px}
.pill{display:flex;align-items:center;gap:8px;background:#0b1320;border:1px solid var(--line);border-radius:999px;padding:7px 13px;font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--red)}.dot.ok{background:var(--grn);box-shadow:0 0 8px rgba(39,232,167,.7)}
.dot.warn{background:var(--amb)}
.big2{font-size:26px;font-weight:700;color:#fff}.unit{font-size:12px;color:var(--dim)}
.muted{color:var(--dim);font-size:12px}.mono{font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:3px 9px;border-radius:7px;font-size:12px;font-weight:600}
.tag.skip{background:#15314a;color:var(--cyan)}.tag.take{background:#143a2c;color:var(--grn)}
.foot{margin-top:16px;color:var(--dim);font-size:12px;display:flex;gap:18px;flex-wrap:wrap}
.off{opacity:.45}
</style></head><body><div class="wrap">
<header>
  <div class="pulse" id="pulse"></div>
  <div class="logo">◢◤ GX1 <b>JARVIS</b> <span class="sub">· XAUUSD live</span></div>
  <div class="spacer"></div>
  <div style="text-align:right"><div class="clock mono" id="clock">--:--:--</div><div class="sub" id="conn">kobler til…</div></div>
</header>
<div class="grid">

  <div class="card hero col8">
    <h3>Modellretning · eksakt LONG / SHORT / FLAT argmax</h3>
    <div class="rowflex">
      <div><div class="big" id="dirBig">—</div><div class="muted" id="dirSub">venter på modellbevis…</div></div>
      <div style="flex:1;min-width:220px">
        <div class="gauge"><div class="gfill" id="gfill"></div></div>
        <div class="muted mono" id="probScale"></div>
      </div>
    </div>
    <div class="rowflex" style="margin-top:14px">
      <div style="min-width:190px"><div class="muted">retningens sannsynligheter</div><div class="lean"><div class="l" id="leanL"></div><div class="s" id="leanS"></div></div><div class="muted mono" id="leanTxt"></div></div>
      <div><div class="muted">lærte støttehoder</div><div class="mono" id="modelEvidence" style="color:#fff">—</div></div>
    </div>
  </div>

  <div class="card col4">
    <h3>På minuttet · data</h3>
    <div class="big2 mono" id="lag">—</div><div class="unit" id="lagSub">cutoff-etterslep</div>
    <div class="kv"><span class="k">prebuilt cutoff</span><span class="v mono" id="cutoff">—</span></div>
    <div class="kv"><span class="k">siste beslutning</span><span class="v mono" id="decAge">—</span></div>
  </div>

  <div class="card col12">
    <h3>Stack-helse</h3>
    <div class="pills" id="pills"></div>
  </div>

  <div class="card col6">
    <h3>Marked</h3>
    <div class="rowflex">
      <div><div class="muted">pris (bid/ask)</div><div class="big2 mono" id="px">—</div></div>
      <div><div class="muted">spread</div><div class="big2 mono" id="spread">—</div></div>
      <div><div class="muted">sesjon</div><div class="big2" id="sess">—</div></div>
    </div>
    <div class="kv" style="margin-top:8px"><span class="k">execution-status</span><span class="v" id="gate">—</span></div>
    <div class="kv"><span class="k">åpne trades</span><span class="v mono" id="open">—</span></div>
  </div>

  <div class="card col6">
    <h3>I dag</h3>
    <div class="rowflex">
      <div><div class="muted">beslutninger</div><div class="big2 mono" id="cTot">—</div></div>
      <div><div class="muted">SKIP</div><div class="big2 mono" id="cSkip">—</div></div>
      <div><div class="muted">TAKE</div><div class="big2 mono" id="cTake">—</div></div>
    </div>
    <div class="kv" style="margin-top:8px"><span class="k">operating point</span><span class="v mono" id="op">—</span></div>
    <div class="kv"><span class="k">journal</span><span class="v mono" id="jrnl" style="font-size:11px">—</span></div>
  </div>

  <div class="card col12">
    <h3>Trades · P&amp;L</h3>
    <div class="rowflex">
      <div><div class="muted">åpen trade</div><div class="big2" id="openTrade">ingen</div></div>
      <div><div class="muted">trades i dag</div><div class="big2 mono" id="tToday">—</div></div>
      <div><div class="muted">realisert i dag</div><div class="big2 mono" id="tBps">—</div></div>
      <div style="flex:1;min-width:240px"><div class="muted">siste trade</div><div id="tLast" class="mono" style="color:#fff">—</div></div>
    </div>
  </div>

</div>
<div class="foot">
  <span id="updated">—</span><span class="muted">auto-oppdaterer hvert 3s · READ-ONLY (rører ingenting)</span>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
function fmtAge(m){if(m==null)return"—";if(m<1)return"akkurat nå";if(m<60)return m.toFixed(1)+" min siden";return (m/60).toFixed(1)+" t siden";}
function pct(v){return v==null?"—":(100*v).toFixed(1)+"%";}
function num(v,d=2){return v==null?"—":Number(v).toFixed(d);}
function dotClass(v){return v===true?"dot ok":(v===false?"dot":"dot warn");}
function tickClock(){const d=new Date();$("clock").textContent=d.toISOString().slice(11,19)+"Z";}
setInterval(tickClock,1000);tickClock();

async function refresh(){
 try{
  const r=await fetch('/status.json',{cache:'no-store'});const s=await r.json();
  $("conn").textContent="● tilkoblet";$("conn").style.color="var(--grn)";$("pulse").style.background="var(--grn)";

  // Exact model-native direction and learned evidence; no live threshold.
  const c=s.model;
  if(c && !c.contract_error){
    const cls=c.direction==="LONG"?"grn":(c.direction==="SHORT"?"red":"amb");
    $("dirBig").textContent=c.direction;$("dirBig").className="big "+cls;
    $("dirSub").textContent="modellens eneste rå-Q handling · margin "+num(c.entry_action_q_margin_bps)+" bps";
    const g=$("gfill");g.style.width="100%";
    g.style.background=c.direction==="LONG"?"linear-gradient(90deg,#1b8f6e,var(--grn))":(c.direction==="SHORT"?"linear-gradient(90deg,#6b1e2b,var(--red))":"linear-gradient(90deg,#8a6a1e,var(--amb))");
    $("probScale").textContent="valgt Q "+num(c.selected_q_bps)+" bps · margin "+num(c.entry_action_q_margin_bps)+" bps";
    $("leanL").style.width="0%";$("leanS").style.width="0%";
    $("leanTxt").textContent="Q long "+num(c.q_long_bps)+" · short "+num(c.q_short_bps)+" · flat "+num(c.q_flat_bps)+" bps";
    $("modelEvidence").textContent="ATR "+num(c.atr_bps)+" bps · size-head "+pct(c.position_size_pred)+" (evidens)";
  } else if(c && c.contract_error){
    $("dirBig").textContent="BLOCK";$("dirBig").className="big red";
    $("dirSub").textContent="modellkontrakt ugyldig: "+c.contract_error;
    $("gfill").style.width="0%";$("probScale").textContent="ingen bevist retning";$("modelEvidence").textContent="—";
  } else {$("dirBig").textContent="—";$("dirSub").textContent="ingen fersk modellbeslutning";$("gfill").style.width="0%";}

  // data freshness
  const d=s.data;const lag=d.cutoff_lag_min;
  $("lag").textContent=lag==null?"—":lag.toFixed(1)+" min";
  $("lag").style.color=lag==null?"#fff":(lag<8?"var(--grn)":(lag<20?"var(--amb)":"var(--red)"));
  $("cutoff").textContent=d.cutoff||d.publisher_status||"—";$("decAge").textContent=fmtAge(d.last_decision_age_min);

  // health pills
  const h=s.health;const labels={runner:"paper_runner",collector:"collector",live_tail_publisher:"live-tail publisher",git_clean:"git rent",auto_recover:"auto-recover"};
  $("pills").innerHTML=Object.keys(labels).map(k=>`<div class="pill"><span class="${dotClass(h[k])}"></span>${labels[k]}</div>`).join("");
  const allok=Object.values(h).every(v=>v===true);
  $("pulse").style.background=allok?"var(--grn)":"var(--amb)";

  // market
  const m=s.market;
  if(m){$("px").textContent=(m.bid!=null?m.bid:"—")+" / "+(m.ask!=null?m.ask:"—");
    $("spread").textContent=(m.spread_bps!=null?m.spread_bps:"—")+" bps";
    $("sess").textContent=m.session||"—";$("gate").textContent=m.execution_status||"—";
    $("open").textContent=(m.n_open_trades??"—")+" / "+(s.op.max_trades||3);}

  // today
  const t=s.today||{};
  $("cTot").textContent=t.total??"—";$("cSkip").textContent=t.SKIP??"—";
  $("cTake").textContent=((t.TAKE_LONG_NOW||0)+(t.TAKE_SHORT_NOW||0));
  const sz=s.op.sizing||{};const frac=sz.fraction==null?"—":(100*Number(sz.fraction)).toFixed(1)+"%";
  $("op").textContent="direction exact argmax · sizing "+(sz.mode||"unavailable")+" · fraction "+frac+" · "+(sz.units??0)+"u";
  $("jrnl").textContent=s.journal||"—";

  // trades / P&L
  const tr=s.trades||{};
  if(tr.open && tr.open.length){const o=tr.open[0];
    $("openTrade").innerHTML=`<span class="${o.side==='long'?'':'red'}" style="color:${o.side==='long'?'var(--grn)':'var(--red)'}">${(o.side||'?').toUpperCase()}</span> ${o.units||''}u · ${o.bars||0} bars`;
  } else {$("openTrade").textContent="ingen";$("openTrade").style.color="var(--dim)";}
  $("tToday").textContent=tr.today_n??"—";
  if(tr.today_bps!=null){$("tBps").textContent=(tr.today_bps>0?"+":"")+tr.today_bps+" bps";$("tBps").style.color=tr.today_bps>0?"var(--grn)":(tr.today_bps<0?"var(--red)":"#fff");}
  const L=tr.last;
  if(L){const w=(L.pnl||0)>=0;
    $("tLast").innerHTML=`#${L.id} ${(L.side||'').toUpperCase()} ${L.units||''}u → <b style="color:${w?'var(--grn)':'var(--red)'}">${w?'+':''}${(L.pnl!=null?L.pnl.toFixed(1):'?')} bps</b> · mfe ${L.mfe!=null?L.mfe.toFixed(0):'?'} / mae ${L.mae!=null?L.mae.toFixed(0):'?'} · ${L.reason||''} · ${(L.exit_time||'').slice(11,16)}Z`;
  } else {$("tLast").textContent="—";}
  $("updated").textContent="oppdatert "+s.now_utc+"Z";
 }catch(e){$("conn").textContent="● frakoblet";$("conn").style.color="var(--red)";$("pulse").style.background="var(--red)";}
}
refresh();setInterval(refresh,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status.json"):
            try:
                body = json.dumps(build_status()).encode()
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
            self._send(body, "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="GX1 Jarvis live dashboard (read-only)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GX1 Jarvis dashboard → http://localhost:{args.port}  (host bind {args.host}:{args.port})")
    print("READ-ONLY. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
