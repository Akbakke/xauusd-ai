"""OOT bigger/seq-model REALITY test (#19, user audit bigger_model_reality 2026-06-17).

QUESTION: is the ~0.62 OOT-2026 direction ceiling an INFORMATION limit, or a CAPACITY limit
a fancier sequence model (TFT/PatchTST/Chronos/TCN/LSTM) would break?

We do NOT pip-install heavy frameworks. Instead we run the DECISIVE cheap proxies that test the
*capacity* hypothesis directly, all on the load-bearing dir_up target, strict-OOT:

  BASELINE   : ~16 point-in-time raw-OHLC features from the FO (body/wick/range/returns/ema/atr/rsi/roc).
               The V10 transformer ALREADY sees per-bar OHLC shape — so a redundant feature must clear THIS.
  SEQ-GBM    : BASELINE + a LAGGED raw-OHLC sequence window (W M5 bars of ret/range/body/upper-lower-wick
               and rolling stats) — i.e. give a high-capacity HGB the SAME temporal window the V10 consumes.
               If sequence info were the missing capacity, dAUC(2026) > 0 here.
  DEEP-GBM   : same SEQ features but a MUCH higher-capacity HGB (depth 8, 600 iters, leaf 50) — the
               "overfit-capacity" probe: train AUC should rocket, OOT should NOT (info-ceiling signature).
  TCN (torch): a small dilated temporal-CNN on the raw W×F sequence (GPU) — a genuine seq-DL arch,
               distinct from GBM and from the existing transformer. The cheap stand-in for PatchTST/TFT.

PASS for "fancier seq model is worth building" = SEQ/DEEP/TCN clears dAUC>0 on the 2026 holdout AND
both robustness splits (fit_early->late, fit_late->early). Standalone ~0.5 = noise; train>>test = capacity
is NOT the bottleneck (info ceiling).

LEAK-SAFE: every sequence feature at decision bar t uses ONLY M5 bars with time <= t (asof searchsorted
side='right'-1, then a backward lag window). No forward/label column enters features (asserted).
"""
import os, glob, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5 = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
      "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(20260617)
SAMPLE_CAP = 200_000
K = "K96"
W = 48          # sequence window (48 M5 bars = 4h of context; V10 uses 96 — we keep it cheap+honest)

BASE_COLS = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
             "_v1_rsi14_canon_v1", "_v1_rsi2_canon_v1", "_v1_r5_canon_v1", "_v1_r24_canon_v1",
             "ret_1_canon_v1", "ret_5_canon_v1", "ret_20_canon_v1", "roc20_canon_v1", "roc100_canon_v1",
             "_v1_ema_diff_canon_v1", "range_canon_v1", "atr_canon_v1", "atr50_canon_v1",
             "_v1_atr14_canon_v1", "_v1_close_ema_slope_3_canon_v1", "vol_ratio_canon_v1", "atr_bps"]
LAB = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "wait_", "take_now_", "_outcome", "trough", "bars_to")


def load_decisions():
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    avail = set(pd.read_parquet(files[0]).columns)
    base = [c for c in BASE_COLS if c in avail]
    need = sorted(set(base + LAB + ["decision_ts_utc"]))
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df, base


def build_seq_window(dec_ts):
    """Build a leak-safe lagged raw-OHLC sequence (W bars) ending at each decision bar.
    Per-bar primitives: log-return, range/atr, body/atr, upper-wick/atr, lower-wick/atr.
    Returns (Xseq float32 [n, W, F], colnames for flattened lag features)."""
    m5 = pd.read_parquet(M5, columns=["time", "open", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    o = m5["open"].to_numpy(np.float64); h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64); c = m5["close"].to_numpy(np.float64)
    tr = np.maximum.reduce([h - l, np.abs(h - np.concatenate([[c[0]], c[:-1]])),
                            np.abs(l - np.concatenate([[c[0]], c[:-1]]))])
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_s = np.maximum(atr, 1e-3)
    cprev = np.concatenate([[c[0]], c[:-1]])
    logret = np.log(np.maximum(c, 1e-9) / np.maximum(cprev, 1e-9))
    rng = (h - l) / atr_s
    body = (c - o) / atr_s
    up_wick = (h - np.maximum(o, c)) / atr_s
    lo_wick = (np.minimum(o, c) - l) / atr_s
    feats = np.stack([logret, rng, body, up_wick, lo_wick], axis=1).astype(np.float32)  # [n,5]
    F = feats.shape[1]

    m5ns = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    decns = dec_ts.values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(m5ns, decns, side="right") - 1   # bar at-or-before decision (leak-safe)

    n = len(idx)
    Xseq = np.zeros((n, W, F), dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    for r in range(n):
        e = idx[r]
        if e < W - 1:
            valid[r] = False
            continue
        Xseq[r] = feats[e - W + 1: e + 1]
    return Xseq, valid, F


def flatten_for_gbm(Xseq):
    """Flatten the W×F window for a GBM: raw lags (last 12 bars × F) + rolling summary stats
    (mean/std/min/max/last over the full window per channel). Keeps dim modest for the tree."""
    n, w, F = Xseq.shape
    LAGS = 12
    lag_block = Xseq[:, -LAGS:, :].reshape(n, LAGS * F)            # recent detail
    mean = Xseq.mean(1); std = Xseq.std(1); mn = Xseq.min(1); mx = Xseq.max(1)  # window stats
    cumsum = Xseq[:, :, 0].sum(1, keepdims=True)                  # window cumulative log-return (momentum)
    names = ([f"lag{j}_f{k}" for j in range(LAGS) for k in range(F)]
             + [f"mean_f{k}" for k in range(F)] + [f"std_f{k}" for k in range(F)]
             + [f"min_f{k}" for k in range(F)] + [f"max_f{k}" for k in range(F)] + ["winsum_ret"])
    X = np.concatenate([lag_block, mean, std, mn, mx, cumsum], axis=1).astype(np.float32)
    return X, names


def hgb(depth, iters, leaf):
    return HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, learning_rate=0.05,
                                          l2_regularization=1.0, min_samples_leaf=leaf,
                                          early_stopping=True, validation_fraction=0.15, random_state=42)


def fit_auc(Xtr, ytr, Xte, yte, clf, want_train=False):
    if len(Xtr) > SAMPLE_CAP:
        s = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr[s]; ytr = ytr[s]
    clf.fit(Xtr, ytr)
    oot = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) if len(np.unique(yte)) > 1 else float("nan")
    tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1]) if want_train else None
    return oot, tr


# ── tiny dilated TCN (torch, GPU) ─────────────────────────────────────────────
def train_tcn(Xtr, ytr, Xte, yte, F, epochs=25):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    class TCN(nn.Module):
        def __init__(self, fin, ch=32):
            super().__init__()
            def blk(ci, co, d):
                return nn.Sequential(nn.Conv1d(ci, co, 3, padding=d, dilation=d), nn.BatchNorm1d(co),
                                     nn.ReLU(), nn.Dropout(0.1))
            self.net = nn.Sequential(blk(fin, ch, 1), blk(ch, ch, 2), blk(ch, ch, 4), blk(ch, ch, 8))
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 1))

        def forward(self, x):                      # x: [B, W, F] -> [B, F, W]
            return self.head(self.net(x.transpose(1, 2))).squeeze(1)

    if len(Xtr) > SAMPLE_CAP:
        s = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr[s]; ytr = ytr[s]
    # standardize per-channel on train
    mu = Xtr.reshape(-1, F).mean(0); sd = Xtr.reshape(-1, F).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    # internal val split for early-stop
    nval = int(0.15 * len(Xtr)); perm = RNG.permutation(len(Xtr))
    vi, ti = perm[:nval], perm[nval:]
    Xt = torch.tensor(Xtr[ti], device=dev); yt = torch.tensor(ytr[ti].astype(np.float32), device=dev)
    Xv = torch.tensor(Xtr[vi], device=dev); yv = ytr[vi]
    Xe = torch.tensor(Xte, device=dev)
    m = TCN(F).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(yt.mean().item()); pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    bs = 8192; best_val = -1; best_state = None; patience = 4; bad = 0
    for ep in range(epochs):
        m.train(); order = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            b = order[i:i + bs]; opt.zero_grad()
            loss = lossf(m(Xt[b]), yt[b]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xv)).cpu().numpy()
        va = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if va > best_val + 1e-4:
            best_val = va; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pe = torch.sigmoid(m(Xe)).cpu().numpy()
        ptr = torch.sigmoid(m(Xt)).cpu().numpy()
    oot = roc_auc_score(yte, pe) if len(np.unique(yte)) > 1 else float("nan")
    tr = roc_auc_score(yt.cpu().numpy(), ptr)
    return oot, tr


def main():
    t0 = time.time()
    print("[1] loading decisions + baseline ...")
    df, base = load_decisions()
    print(f"  decision rows={len(df)}  baseline_cols={len(base)}")

    print(f"[2] building leak-safe M5 sequence window (W={W}) ...")
    Xseq, valid, F = build_seq_window(df["ts"])
    df = df[valid].reset_index(drop=True); Xseq = Xseq[valid]
    print(f"  valid rows={len(df)}  seq shape={Xseq.shape}")

    # target dir_up + masks
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    ok = ~(np.isnan(lp) | np.isnan(sp))
    df = df[ok].reset_index(drop=True); Xseq = Xseq[ok]; lp = lp[ok]; sp = sp[ok]
    y = (lp > sp).astype(int)
    print(f"  dir_up base_rate={y.mean():.4f}  n={len(y)}")

    # leakage guard
    for c in base:
        assert not any(tok in c for tok in FORBIDDEN), f"LEAKAGE: {c}"

    # feature matrices
    Xbase = df[base].astype(np.float32).replace([np.inf, -np.inf], np.nan).values
    Xflat, flat_names = flatten_for_gbm(Xseq)
    Xseqgbm = np.concatenate([Xbase, Xflat], axis=1)
    print(f"  Xbase={Xbase.shape} Xflat={Xflat.shape} Xseqgbm={Xseqgbm.shape}")

    # splits
    ts = df["ts"].values
    med = df["ts"].median()
    early = (df["ts"] <= med).values; late = (df["ts"] > med).values
    h2025 = (df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")).values
    oot26 = (df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")).values
    print(f"  split@{med}  early={early.sum()} late={late.sum()} | strict: fit<=2025-06 n={h2025.sum()} test>=2026 n={oot26.sum()}")

    def run_arm(name, X, model_fn, want_train=True, is_tcn=False, Xtcn=None):
        rows = {}
        for sn, tr, te in (("strict_2026", h2025, oot26), ("fit_early->late", early, late),
                           ("fit_late->early", late, early)):
            if is_tcn:
                oot, trn = train_tcn(Xtcn[tr], y[tr], Xtcn[te], y[te], F)
            else:
                oot, trn = fit_auc(X[tr], y[tr], X[te], y[te], model_fn(), want_train=want_train)
            rows[sn] = dict(oot=round(float(oot), 4), train=round(float(trn), 4) if trn is not None else None)
        return rows

    print("\n[3] BASELINE (point-in-time raw-OHLC, HGB d4) ...")
    base_res = run_arm("baseline", Xbase, lambda: hgb(4, 300, 200))
    for k, v in base_res.items():
        print(f"  [{k}] OOT={v['oot']}  train={v['train']}")

    print("\n[4] SEQ-GBM (baseline + lagged W-bar sequence, HGB d4) ...")
    seq_res = run_arm("seq_gbm", Xseqgbm, lambda: hgb(4, 300, 200))
    for k, v in seq_res.items():
        d = round(v["oot"] - base_res[k]["oot"], 4)
        print(f"  [{k}] OOT={v['oot']}  train={v['train']}  dAUC_vs_base={d:+.4f}")

    print("\n[5] DEEP-GBM (same seq feats, HIGH capacity d8/600/leaf50 — overfit probe) ...")
    deep_res = run_arm("deep_gbm", Xseqgbm, lambda: hgb(8, 600, 50))
    for k, v in deep_res.items():
        d = round(v["oot"] - base_res[k]["oot"], 4)
        print(f"  [{k}] OOT={v['oot']}  train={v['train']}  dAUC_vs_base={d:+.4f}  train-test_gap={round(v['train']-v['oot'],4)}")

    print("\n[6] TCN (dilated temporal CNN on raw W×F sequence, GPU) ...")
    tcn_res = run_arm("tcn", None, None, is_tcn=True, Xtcn=Xseq)
    for k, v in tcn_res.items():
        d = round(v["oot"] - base_res[k]["oot"], 4)
        print(f"  [{k}] OOT={v['oot']}  train={v['train']}  dAUC_vs_base={d:+.4f}")

    out = dict(W=W, F=F, n=int(len(y)), base_rate=float(y.mean()),
               baseline=base_res, seq_gbm=seq_res, deep_gbm=deep_res, tcn=tcn_res,
               elapsed_s=round(time.time() - t0, 1))
    Path("/tmp/oot_bigger_seq_model_reality.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWROTE /tmp/oot_bigger_seq_model_reality.json  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
