#!/usr/bin/env python3
"""GX1 "Jarvis" live dashboard — zero-dependency local status page.

A single self-contained, READ-ONLY view of the live XAUUSD paper-trading stack:
how far the bot is from a trade (conviction proximity vs the live gate), whether
we are "on the minute" (data freshness), stack health, market + today's tally.

ONE-TRUTH: it reads the SAME live sources the operator inspects by hand —
  • the paper journal (per-poll v12_decision: q-values, advantage-over-skip, V10 lean)
  • the canonical_incremental daemon log (prebuilt cutoff = data freshness)
  • live process status via pgrep (runner / collector / canonical / counterfactual)
  • the running runner's OWN env (/proc/PID/environ) for the live operating point
    (GX1_CONVICTION_THR / SKIP_ASIA / SIZING_MODE) — never a hardcoded guess
It NEVER writes to any live file and NEVER touches the cement/contract/prebuilts
or any process (pgrep/proc reads only). Stdlib only — no new deps.

Run:   /home/andre2/src/GX1_ENGINE/.venv/bin/python scripts/gx1_dashboard.py [--port 8787] [--host 0.0.0.0]
Open:  http://localhost:8787   (from the Windows browser; WSL2 forwards localhost)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path("/home/andre2/src/GX1_ENGINE")
RUNS = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
JOURNAL_GLOB = str(RUNS / "v12_paper_journal_*.jsonl")
CANON_LOG = RUNS / "logs" / "canonical_incremental.log"
WANTS = Path.home() / ".config/systemd/user/default.target.wants/gx1-paper-runner.service"
DEFAULT_THR = -37.71

_count_cache: dict[str, tuple[float, dict]] = {}


# ── read-only helpers ───────────────────────────────────────────────────────
def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return []


def _proc_env(pid: int | None, key: str, default=None):
    if not pid:
        return default
    try:
        for kv in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if kv.startswith(key.encode() + b"="):
                return kv.split(b"=", 1)[1].decode()
    except Exception:
        pass
    return default


def _latest_journal():
    files = sorted(glob.glob(JOURNAL_GLOB))
    return files[-1] if files else None


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
    counts = {"SKIP": 0, "TAKE_LONG": 0, "TAKE_SHORT": 0, "total": 0}
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


def _canon_cutoff():
    for line in reversed(_tail_lines(CANON_LOG, 20_000)):
        m = re.search(r"new[_ ]cutoff'?:?\s*'?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            return m.group(1)
    return None


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
    thr = DEFAULT_THR
    try:
        thr = float(_proc_env(rpid, "GX1_CONVICTION_THR", DEFAULT_THR))
    except Exception:
        pass

    jrnl = _latest_journal()
    dec = _last_decision(jrnl) if jrnl else None
    counts = _today_counts(jrnl) if jrnl else {}
    cutoff = _canon_cutoff()
    cutoff_dt = _parse_ts(cutoff)
    cutoff_lag = round((now - cutoff_dt).total_seconds() / 60, 1) if cutoff_dt else None

    conv = None
    market = None
    dec_age = None
    if dec:
        vd = dec["v12_decision"]
        al, ash = vd.get("advantage_over_skip_long"), vd.get("advantage_over_skip_short")
        if al is not None and ash is not None:
            raw = max(al, ash)
            conv = {
                "raw_adv": round(raw, 1), "thr": round(thr, 2), "dist": round(raw - thr, 1),
                "side": "LONG" if al >= ash else "SHORT", "would_take": raw >= thr,
                "p_long": vd.get("v10_p_long"), "p_short": vd.get("v10_p_short"),
                "q_skip": vd.get("q_skip"), "q_take_long": vd.get("q_take_long"),
                "q_take_short": vd.get("q_take_short"),
            }
        feats = dec.get("features", {})
        market = {
            "bid": dec.get("bid"), "ask": dec.get("ask"),
            "spread_bps": round(dec.get("spread_bps", 0), 2),
            "session": (feats.get("session") or "?").upper(),
            "vol_regime": feats.get("vol_regime"),
            "action": vd.get("action"), "gate_reason": dec.get("gate_reason"),
            "n_open_trades": dec.get("n_open_trades", 0),
        }
        dt = _parse_ts(dec.get("ts_utc"))
        dec_age = round((now - dt).total_seconds() / 60, 1) if dt else None

    return {
        "now_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "health": {
            "runner": rpid is not None,
            "collector": bool(_pgrep("v12_oanda_data_collector")),
            "canonical": bool(_pgrep("v12_canonical_incremental")),
            "counterfactual": bool(_pgrep("v12_daily_counterfactual")),
            "git_clean": _git_clean(),
            "auto_recover": WANTS.is_symlink(),
        },
        "conviction": conv,
        "data": {
            "cutoff": cutoff, "cutoff_lag_min": cutoff_lag,
            "last_decision_ts": dec.get("ts_utc") if dec else None,
            "last_decision_age_min": dec_age,
        },
        "market": market,
        "today": counts,
        "trades": _trades_summary(dec.get("open_trade_records") if dec else None),
        "op": {
            "conviction_thr": round(thr, 2),
            "skip_asia": _proc_env(rpid, "GX1_SKIP_ASIA", "?"),
            "sizing": _proc_env(rpid, "GX1_SIZING_MODE", "?"),
            "max_trades": _proc_cmdline_arg(rpid, "--max-trades", 3),
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
.gthr{position:absolute;top:-5px;bottom:-5px;width:2px;background:#fff;box-shadow:0 0 8px #fff}
.gthr span{position:absolute;top:-18px;left:50%;transform:translateX(-50%);font-size:10px;color:var(--dim);white-space:nowrap}
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
    <h3>Nærhet til trade · conviction-gate</h3>
    <div class="rowflex">
      <div><div class="big" id="convBig">—</div><div class="muted" id="convSub">venter på data…</div></div>
      <div style="flex:1;min-width:220px">
        <div class="gauge"><div class="gfill" id="gfill"></div><div class="gthr" id="gthr"><span>trigger</span></div></div>
        <div class="muted mono" id="convScale"></div>
      </div>
    </div>
    <div class="rowflex" style="margin-top:14px">
      <div style="min-width:160px"><div class="muted">V10 lener</div><div class="lean"><div class="l" id="leanL"></div><div class="s" id="leanS"></div></div><div class="muted mono" id="leanTxt"></div></div>
      <div><div class="muted">Q (skip / long / short)</div><div class="mono" id="qvals" style="color:#fff">—</div></div>
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
    <div class="kv" style="margin-top:8px"><span class="k">gate</span><span class="v" id="gate">—</span></div>
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
function dotClass(v){return v===true?"dot ok":(v===false?"dot":"dot warn");}
function tickClock(){const d=new Date();$("clock").textContent=d.toISOString().slice(11,19)+"Z";}
setInterval(tickClock,1000);tickClock();

async function refresh(){
 try{
  const r=await fetch('/status.json',{cache:'no-store'});const s=await r.json();
  $("conn").textContent="● tilkoblet";$("conn").style.color="var(--grn)";$("pulse").style.background="var(--grn)";

  // conviction
  const c=s.conviction;
  if(c){
    if(c.would_take){$("convBig").textContent="🟢 TRADE-KLAR";$("convBig").className="big grn";$("convSub").textContent="conviction over terskel — "+c.side;}
    else{const d=Math.abs(c.dist);$("convBig").textContent=d.toFixed(0)+" bps";$("convBig").className="big "+(d<20?"amb":(d<60?"":"red"));$("convSub").textContent="unna en trade ("+c.side+" nærmest · raw_adv "+c.raw_adv+" vs "+c.thr+")";}
    // gauge: map [thr-200, thr+60] -> 0..100
    const lo=c.thr-200, hi=c.thr+60, span=hi-lo;
    const pct=Math.max(0,Math.min(100,(c.raw_adv-lo)/span*100));
    const tpct=Math.max(0,Math.min(100,(c.thr-lo)/span*100));
    const g=$("gfill");g.style.width=pct+"%";
    g.style.background=c.would_take?"linear-gradient(90deg,#1b8f6e,var(--grn))":(Math.abs(c.dist)<20?"linear-gradient(90deg,#8a6a1e,var(--amb))":"linear-gradient(90deg,#3a1820,var(--red))");
    $("gthr").style.left=tpct+"%";
    $("convScale").textContent="skala "+lo.toFixed(0)+" … "+hi.toFixed(0)+" bps";
    const pl=(c.p_long||0)*100, ps=(c.p_short||0)*100, pf=Math.max(0,100-pl-ps);
    $("leanL").style.width=pl+"%";$("leanS").style.width=ps+"%";
    $("leanTxt").textContent="long "+pl.toFixed(0)+"% · short "+ps.toFixed(0)+"% · flat "+pf.toFixed(0)+"%";
    $("qvals").textContent=[c.q_skip,c.q_take_long,c.q_take_short].map(x=>x==null?"—":x.toFixed(0)).join("  /  ");
  } else {$("convBig").textContent="—";$("convSub").textContent="ingen fersk beslutning (runner laster?)";}

  // data freshness
  const d=s.data;const lag=d.cutoff_lag_min;
  $("lag").textContent=lag==null?"—":lag.toFixed(1)+" min";
  $("lag").style.color=lag==null?"#fff":(lag<8?"var(--grn)":(lag<20?"var(--amb)":"var(--red)"));
  $("cutoff").textContent=d.cutoff||"—";$("decAge").textContent=fmtAge(d.last_decision_age_min);

  // health pills
  const h=s.health;const labels={runner:"paper_runner",collector:"collector",canonical:"canonical_incr",counterfactual:"counterfactual",git_clean:"git rent",auto_recover:"auto-recover"};
  $("pills").innerHTML=Object.keys(labels).map(k=>`<div class="pill"><span class="${dotClass(h[k])}"></span>${labels[k]}</div>`).join("");
  const allok=Object.values(h).every(v=>v===true);
  $("pulse").style.background=allok?"var(--grn)":"var(--amb)";

  // market
  const m=s.market;
  if(m){$("px").textContent=(m.bid!=null?m.bid:"—")+" / "+(m.ask!=null?m.ask:"—");
    $("spread").textContent=(m.spread_bps!=null?m.spread_bps:"—")+" bps";
    $("sess").textContent=m.session||"—";$("gate").textContent=m.gate_reason||"—";
    $("open").textContent=m.n_open_trades+" / "+(s.op.max_trades||3);}

  // today
  const t=s.today||{};
  $("cTot").textContent=t.total??"—";$("cSkip").textContent=t.SKIP??"—";
  $("cTake").textContent=((t.TAKE_LONG||0)+(t.TAKE_SHORT||0));
  $("op").textContent="thr "+s.op.conviction_thr+" · skip-asia "+s.op.skip_asia+" · sizing "+s.op.sizing;
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
