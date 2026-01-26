#!/usr/bin/env python3
import csv
from datetime import datetime
from pathlib import Path
import json

def to_float(v):
    try:
        return float(v)
    except ValueError:
        return 0.0

def parse_time(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace('Z', '+00:00'))

run_dir = Path('gx1/wf_runs/SNIPER_OBS_Q1_2025_20251217_173932')
rows = []
with open(run_dir / 'trade_journal' / 'trade_journal_index.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

pnl = []
durations = []
exit_reason_counts = {}
eof_count = 0

for row in rows:
    pnl_val = to_float(row.get('pnl_bps', '0'))
    pnl.append(pnl_val)
    exit_reason = row.get('exit_reason', '')
    exit_reason_counts[exit_reason] = exit_reason_counts.get(exit_reason, 0) + 1
    if exit_reason == 'REPLAY_EOF':
        eof_count += 1
    entry = parse_time(row.get('entry_time', ''))
    exit_ = parse_time(row.get('exit_time', ''))
    if entry and exit_:
        durations.append((exit_ - entry).total_seconds() / 60.0)

total = len(rows)
wins = [x for x in pnl if x > 0]
losses = [x for x in pnl if x < 0]
win_rate = len(wins)/total if total else 0
avg_win = sum(wins)/len(wins) if wins else 0
avg_loss = sum(losses)/len(losses) if losses else 0
payoff = (avg_win / abs(avg_loss)) if losses else 0

ev = sum(pnl)/total if total else 0
pnl_sum = sum(pnl)
durations_sorted = sorted(durations)
quantile = lambda q: durations_sorted[min(max(int(q * len(durations_sorted)) - 1, 0), len(durations_sorted) - 1)] if durations_sorted else 0

duration_stats = {
    'p50': quantile(0.5),
    'p90': quantile(0.9),
    'p95': quantile(0.95),
}

time_market = sum(durations)
avg_duration = time_market / len(durations) if durations else 0

sessions = {}
for sess in ['EU', 'OVERLAP', 'US']:
    subset = [row for row in rows if row.get('session') == sess]
    if not subset:
        sessions[sess] = None
        continue
    stats = [to_float(r.get('pnl_bps', '0')) for r in subset]
    duration_vals = []
    wins_s = [x for x in stats if x > 0]
    losses_s = [x for x in stats if x < 0]
    win_rate_s = len(wins_s)/len(subset)
    avg_duration_s = 0
    for r in subset:
        entry = parse_time(r.get('entry_time', ''))
        exit_ = parse_time(r.get('exit_time', ''))
        if entry and exit_:
            avg_duration_s += (exit_ - entry).total_seconds()/60.0
    avg_duration_s = avg_duration_s/len(subset)
    payoff_s = (sum(wins_s)/len(wins_s))/(abs(sum(losses_s)/len(losses_s))) if wins_s and losses_s else 0
    sessions[sess] = {
        'trades': len(subset),
        'win_rate': win_rate_s,
        'ev': sum(stats)/len(stats),
        'payoff': payoff_s,
        'avg_duration': avg_duration_s,
    }

volumes = {}
for regime in ['LOW', 'MEDIUM', 'HIGH']:
    subset = [row for row in rows if row.get('vol_regime', '').upper() == regime]
    if not subset:
        volumes[regime] = None
        continue
    stats = [to_float(r.get('pnl_bps', '0')) for r in subset]
    wins_r = [x for x in stats if x > 0]
    losses_r = [x for x in stats if x < 0]
    win_rate_r = len(wins_r)/len(subset)
    avg_duration_r = 0
    for r in subset:
        entry = parse_time(r.get('entry_time', ''))
        exit_ = parse_time(r.get('exit_time', ''))
        if entry and exit_:
            avg_duration_r += (exit_ - entry).total_seconds()/60.0
    avg_duration_r = avg_duration_r/len(subset)
    payoff_r = (sum(wins_r)/len(wins_r))/(abs(sum(losses_r)/len(losses_r))) if wins_r and losses_r else 0
    volumes[regime] = {
        'trades': len(subset),
        'win_rate': win_rate_r,
        'ev': sum(stats)/len(stats),
        'payoff': payoff_r,
        'avg_duration': avg_duration_r,
    }

# concurrency
events = []
for r in rows:
    entry = parse_time(r.get('entry_time', ''))
    exit_ = parse_time(r.get('exit_time', ''))
    if entry and exit_:
        events.append((entry, 1))
        events.append((exit_, -1))
events.sort()
current = 0
max_concurrent = 0
for _, delta in events:
    current += delta
    max_concurrent = max(max_concurrent, current)

drawdown = {
    'p50': sorted([abs(x) for x in losses])[int(0.5 * len(losses)) - 1] if losses else 0,
    'p90': sorted([abs(x) for x in losses])[int(0.9 * len(losses)) - 1] if losses else 0,
}

print(json.dumps({
    'total': total,
    'win_rate': win_rate,
    'avg_win': avg_win,
    'avg_loss': avg_loss,
    'payoff': payoff,
    'ev': ev,
    'pnl_sum': pnl_sum,
    'duration_stats': duration_stats,
    'time_market': time_market,
    'avg_duration': avg_duration,
    'session': sessions,
    'vol': volumes,
    'max_concurrent': max_concurrent,
    'drawdown': drawdown,
    'eof': eof_count,
    'exit_counts': exit_reason_counts,
}, indent=2))
