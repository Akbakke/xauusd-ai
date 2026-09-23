# Entry-retning: signal/støy-diagnose og instrument — 2026-09-23

Operatørspørsmål: *«Vi har masse features og en dyp AI-modell, men entry har
så og si myntkast på retningsprediksjon. Hvorfor klarer vi ikke finne
retningen?»* Operatørvedtak samme kveld: *«gjør alt dette her»* — instrument,
mål/tap-endringer, horisont, regime, dynamikk.

Kilde ved skriving: `audit/v9-premiere-20260905`, HEAD `03a4d51b` (Codex' siste
retningskontroll) pluss dette instrumentet. Ingen kandidat kjører. TEST er
forseglet og ble ikke lest av noen måling under. Evidensklasser per CLAUDE.md
regel 2d/25a: **[M]** målt på ekte deklarerte bytes, **[S]** bevist fra
kilde/algebra, **[N]** ikke undersøkt.

Bevisrot for egne målinger:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/ENTRY_DIRECTION_SNR_CLAUDE_20260923/`
(`measure_direction_snr.py`, `measure_coverage_curve.py`, JSON). Begge kjørt
under `scripts/gx1_capped_run.sh --class audit`. Walk-forward-instrumentets
evidens ligger under `.../ENTRY_DIRECTION_WALKFORWARD_20260923/`.

## 1. Markedet: kost/støy-aritmetikk per horisont [M]

Native M5-tape `XAU_M5_NATIVE_2019_20260804_V4`, deklarert TRAIN-vindu
2021-06-01 → 2026-05-31 (354 570 rader), tapen kuttet ved VAL-start før noen
statistikk. Spread: snitt 1,68 bps, median 1,57. Break-even-treff for en
alle-bar-fortegnspredikator er `0,5 + kost / (2·E|r_h|)`.

| Horisont | E\|r\| bps | Break-even, kun spread | Break-even, spread + 4 bps | p(opp) | Beste målte treff, kronologisk holdout |
|---|---:|---:|---:|---:|---:|
| 5 min | 4,0 | 71,0 % | 120,9 % | 0,503 | 50,2 % (OLS trailing) / 51,1 % (ridge 312 felt, 65k senere TRAIN-rader) |
| 25 min | 9,0 | 59,4 % | 81,7 % | 0,508 | 50,5 % |
| 60 min | 14,0 | 56,0 % | 70,3 % | 0,512 | 51,6 % |
| 95 min (knee) | ~17,5 | ~54,8 % | ~66,2 % | | (interpolert) |
| 120 min | 20,0 | 54,2 % | 64,2 % | 0,517 | 51,8 % |
| 4 t | 29,1 | 52,9 % | 59,8 % | 0,519 | 52,1 % |
| 8 t | 42,8 | 52,0 % | 56,6 % | 0,525 | 53,5 % (drift-tung) |
| 1 d | 78,5 | 51,1 % | 53,6 % | 0,537 | 55,8 % (p(opp) 53,7 % = drift) |

«Spread + 4 bps» er det arkiverte forskningsscenariet (2 bps slippage per
utførelse, bundet 23.09, ikke bekreftede meglervilkår). Lineær referanse: OLS
på trailing mid-returns (lag 1/5/12/24/96/288), fit 2021-06..2025-05, holdout
2025-06..2026-05; holdout-korrelasjon +0,001 (5 min) … +0,076 (1 d).

Autokorrelasjon i 5-min-returns er null på lag 1–12 (|ρ| < 0,006); varians-ratio
0,97–1,01 på alle horisonter. **Trend-persistens** P(fortegn på neste h-bar-retur
= fortegn på forrige h-bar-retur) er 0,488–0,500 for ALLE h fra 5 min til 1 dag
og i hvert år 2021–2026; utførbar «handle med trailing-fortegn» er negativ på
alle horisonter (−0,5 … −1,7 bps), og contrarian også (−1,7 … −2,9). En
pris-intern «umiddelbar trend»-regel finnes ikke på dette båndet.

VAL-måneden juni 2026: std dobles (2 t: 50 vs 32 bps), p(opp) 0,436 på 2 t og
0,395 på 8 t; alltid-SHORT +3,2 bps på 2 t og +16,8 på 8 t. En modell fittet på
2021–2025 (long-tiltet drift) blir straffet i denne måneden uansett kvalitet.

## 2. Q-målets struktur på V12-substratet [M]

Datasett `V12_FIVE_YEAR_ENTRY_NOTIONAL_20260922T053058Z`, 313 399 TRAIN-rader,
knee-horisont 19 barer (`diagnostic_outcome_horizon_bars`), etiketter
`y_direction_{long,short}_score_bps` (bit-identiske med
`y_{long,short}_final_pnl_at_direction_horizon_bps`, evaluatorens
research-utfall).

| Størrelse | TRAIN | VAL (juni 2026) |
|---|---:|---:|
| Myntkast / orakel / alltid-L / alltid-S | −1,66 / +16,92 / −1,34 / −1,97 | −1,54 / +29,69 / −5,87 / +2,78 |
| Tilgjengelig ferdighet (orakel − myntkast) | 18,58 | 31,24 |
| Andel rader med minst én positiv side | 0,923 | 0,960 |
| Kontrast d = L − S: snitt / std | +0,64 / 58,3 | −8,65 / 87,9 |
| Fortegns-SNR \|snitt\|/std | 0,011 | 0,098 |
| Andel av Σd² i topp 5 % / 10 % av \|d\| | 58,2 % / 71,4 % | 46,2 % / 61,7 % |
| corr(\|d\|, y_vol_fwd_K12) | 0,562 | 0,517 |
| corr(L, −S) | 1,000 | 1,000 |
| p(L bedre enn S) | 0,514 | 0,451 |

Per år (TRAIN): p(L bedre) 0,494–0,534; std_d 44–60 i 2021–2025 og 111 i 2026;
orakel 13–19 og 32,7 i 2026. Fordi corr(L, −S) = 1,000 er de to sidemålene ett
endimensjonalt mål: `L = d/2 − kost`, `S = −d/2 − kost`. MSE i rå bps på begge
sider er derfor MSE på d, der fem prosent av radene bærer 58 % av
kvadratmassen og magnituden i hovedsak er volatilitet.

## 3. Hvordan Entry-målet faktisk bygges [S]

- `gx1/contracts/entry_fitted_q_v1.py:76-112, 263-328`: LONG/SHORT-mål er den
  frosne TRAIN-Exit-lærerens n-step-verdi ved første post-fill-tilstand
  (følg lærerens unike argmax; observert reward ved første EXIT; ellers
  frossen max-Q ved siste observerte tilstand; tie bootstrapper). FLAT = 0
  eksakt, gamma 1, HOLD-reward 0. Økonomi `gross_spread_inclusive_research_only`;
  provisjon/slippage/swap/finansiering ikke bundet (`:126-160`).
- `gx1/contracts/unified_exit_fitted_q_v1.py:192-254, 292-341`: Q_exit =
  utførbar research-PnL nå; Q_hold = stop-gradient max over frossen Q ved neste
  tilstand; Double-Q av; 512 M1-tilstander per side; lærer oppdateres hvert
  `max(1, steps_per_epoch // 512)` steg.
- Tap: `entry_v10_ctx_train_v3.py:8414-8417` MSE over gyldige celler i rå bps;
  skalarisering `sum(exp(−s_i)·L_i + s_i)` per oppgave (`:359-406`, kontrakt
  `entry_model_native_joint_task_weighting_v1`). Ingen per-rad-skala.
- Q-hodet leser `cat(z_v3, z, mtf_repr, global_ctx)` (512) → LN → 128 → 3
  (`entry_v10_ctx_hybrid_transformer.py:1069-1071, 3153-3161`).
- Seleksjon: full-VAL brutto spread-inklusiv snitt-bps per Entry-rad, FLAT = 0
  (`:11202-11205`, `entry_candidate_checkpoint_policy_v1.py:56-67`).
- Evaluatorens forhåndsregistrerte PASS (`evaluate_entry_candidate_selective_edge_v1.py:685-688`):
  excess over myntkast > 2·HAC-SE OG mean > sirkulær-shift p95. Den krever
  **ikke** `mean_pnl_bps > 0`. Instrumentet under legger til `strict_pass`.

## 4. Hva dagens kontroller viste [M, fra 23.09-dokumentene]

- Lærer-optimisme: n-step-mål +5,34 bps mot realisert vindusmark −3,42 på 512
  TRAIN-rader; alle 126/126 sidemål positive på 63 rader; SHORT foretrukket
  63/63; åpne posisjoner mål −21,6 mot mark −51,7.
- Etter hver fit velger Entry SHORT 512/512 (fast lærer, flyttende lærer,
  ruting-fix, L1-warmup, BCE-warmup); Codex' joint-warmup: 31 LONG / 481 SHORT.
- Specialist-gate 99,7 % session (rang-4-familie); vol-hodet slår konstant
  baseline 3/3, forecast-hodet 1/4; Exit-lærerens gjennomsnittlige exit-indeks
  493 av 511.
- Balansert VAL-treff 47–54 % på alle horisonter i alle forsøk; intervallene
  inkluderer 50 %. Ridge på alle 312 felt: 51,06 % på 65 295 senere TRAIN-rader
  ved 5 min (justert intervall 50,35–51,60).
- Codex 23.09: 194 018 av 313 399 TRAIN-rader har både en bullish og en bearish
  horisont blant 5/25/60/120 min — retning er horisontavhengig.

## 5. Mekanismekjeden

1. **Markedet** [M]: fortegnsinformasjonen fra pris-interne features er ~1–2 pp
   over 50 % per bar på ≤ 2 t, mot 54–71 % nødvendig for å slå kostnad.
2. **Målet** [M+S]: Q-målet er vol-dominert med fortegns-SNR 0,011; lærerens
   bootstrap gjør begge sider positive; FLAT = 0 velges aldri; sidevalget blir
   lærerstøy.
3. **Tapet** [S]: MSE i rå bps ⇒ gradienten domineres av halen (58 % av
   Σd² i 5 % av radene) ⇒ representasjonen lærer volatilitet/tid-på-døgnet.
   Målt konsistent med gate-kollaps til session og vol-hodets 3/3.
4. **Instrumentet** [M]: 22 D1-barer i VAL gir ±5–10 pp; juni 2026 er en
   SHORT-måned mot et long-tiltet fit-år; ingen walk-forward, multi-seed eller
   ridge/HGB-baseline har eksistert.
5. **Dagens forsøk** [S]: 512-stegs warmups fra et kollapset sjekkpunkt, målt på
   512 VAL-rader, kan ikke skille 51 % fra 50 % (regel 2f/2g).

Koblingen 3 → gate-kollaps er en hypotese med konsistent evidens, ikke en
målt årsak. Den er testbar: fjern magnitude-dominansen og se om gaten
re-diversifiserer.

## 6. Instrument: `gx1/scripts/research_entry_direction_walkforward_v1.py`

Kontrollflate: `scripts/entry_next_edge_control.sh model-native-direction-walkforward`
(producer-cap 10G, ingen autoritet). Test: `tests/test_research_entry_direction_walkforward.py`.

- Ekspanderende kronologiske folds inne i TRAIN: holdouts 2022-06..2023-05,
  2023-06..2024-05, 2024-06..2025-05, 2025-06..2026-05; purge = maks horisont + 1
  barer på tape-indeks; holdout-rader hvis vindu krysser fold-slutt droppes.
- Mål: knee-utfallet (19 barer, teacher-fri) og utførbare close-fill-returns på
  tapen ved 12/24/48/96/288 barer; skalering `raw` og `atr` (bps / `ctx_cont.atr_bps`,
  prediksjon ganges tilbake før argmax).
- Features: `snapshot` = 241 signal + 71 ctx + one-hot session (4); `snapshot_mtf`
  = + sist-lukkede per-TF-rad (190 × M15/H1/H4/D1) hentet med eierens cutoff
  `t + 5 min − TF-varighet` og **bevist** mot eierens eksakte skalar-aliaser
  (`MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4`, null avvik kreves).
- Lærere: ridge (alpha valgt på indre kronologisk 20 %-splitt, Gram delt per
  fold/arm), HGB (iterasjoner valgt på samme indre splitt fra staged predictions,
  early stopping av, eksplisitte seeds). Én regressor per side; beslutning =
  unik argmax over (L̂, Ŝ, 0), samt kontrast-regel uten FLAT.
- Statistikk: evaluatorens `build_metric_rows` uendret (dekningsgrid 100…1 %,
  Newey-West HAC, eksakt myntkast, sirkulær-shift 512 trekk) + `strict_pass`
  (primary_pass OG mean > 0) + skiver (side-andel, treff på realisert bedre side,
  opp/ned-måneder).

### 6.1 Resultater

Fylles inn per kjøring (run1 ridge begge armer; run2 HGB snapshot; run3 HGB
snapshot_mtf; multi-seed på beste konfigurasjoner). Se `summary.md` i hver
kjøringskatalog.

## 7. Plan for mål/tap-endringer i eksisterende eiere (Phase B)

Rekkefølge etter regel 22: instrumentet avgjør om og hvilken variant som bygges.

1. **Per-rad skalert Entry-tap** (`ENTRY_ACTION_Q_LOSS_ROW_SCALE=atr_bps|none`,
   recipe-eier): `((q − y)/atr_bps)²` over gyldige celler. Ingen ny modellutgang,
   ingen mål-omskriving; Q og argmax forblir i bps. Bygges hvis instrumentets
   `atr`-arm slår `raw`-armen på strict_pass over folds.
2. **Per-prøve lært log-varians** (Kendall per rad) som neste trinn hvis 1 gir
   effekt: ny utgang `entry_action_q_log_variance` fra `entry_q_joint_hidden`,
   modell-/output-skjema v11, movement-/serve-/runtime-kontrakter oppdateres.
3. **Teacher-fri kontrollarm** for Entry-målet (research-profil, ingen bundle):
   realisert verdi under deklarert fast exit (knee-etikettene) i stedet for
   frossen lærer-bootstrap. Kontrakten `fixed_horizon_target_authority: False`
   holder: armen er diagnostikk, ikke beslutningsautoritet.
4. **Horisont**: dersom instrumentet viser edge først ved 4–8 t, deklareres
   retningssupervisjon på den skalaen; M5-klokken velger tidspunkt/abstensjon.
5. **Regime**: krav om tosidig edge (LONG- og SHORT-valgte delmengder hver for
   seg) og opp-/ned-måned-skiver er allerede i instrumentet.
6. **Dynamikk**: FLAT-absorberende seleksjon, min-epoker før patience — etter
   1–4, fordi de ikke skaper signal, bare bevarer det.

Alle native sammenligninger må evalueres med samme fold-semantikk som
instrumentet (research-evaluering over eksplisitte TRAIN-tidsvinduer), ikke
på 512 VAL-rader.

## 8. Ikke undersøkt, sagt uoppfordret

- Ingen native modell er trent med noen av variantene i §7.
- Instrumentets `snapshot_mtf`-arm bruker sist-lukkede per-TF-rader, ikke
  modellens 64/96/96/252-barers sekvenser; en sekvens-effekt kan ikke utelukkes.
- Close-fill-returns på tapen er en research-konvensjon (ikke M1 neste-åpning-fill).
- Multiple sammenligninger over horisonter × dekninger × lærere er ikke
  korrigert; en enkelt strict_pass er ikke et funn, kun gjentatte over folds.
- Kryss-asset, nyheter og mikrostruktur ble refutert på den pensjonerte kjeden;
  ikke re-testet på V12.
