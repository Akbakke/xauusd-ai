# Entry-retning: signal/støy-diagnose og instrument — 2026-09-23

> **Korreksjon 24.09.2026:** Les [ENTRY_FEATURE_FIT_CHRONOLOGY_20260924.md](ENTRY_FEATURE_FIT_CHRONOLOGY_20260924.md).
> De historiske årsavlesningene brukte features med registry-/volatilitetsparametere
> tilpasset frem til mai 2026, også når holdout startet i 2022–2025.
> Tallene er bevart, men disse avlesningene er ikke fullt kronologiske OOS-bevis.
> Lav fremtidskorrelasjon beviser verken fravær av lekkasje eller fravær av
> alle ikke-lineære signaler. Fidelity gjelder bare de undersøkte formlene.
> Generelle utsagn nedenfor om at pris-/makroinformasjon er «refutert», eller at
> inputene ikke kan være problemet, må begrenses til de konkrete gamle forsøkene.


> **Review 24.09:** Les [målereparasjonen og den fullførte før/etter-kontrollen](ENTRY_MEASUREMENT_REPAIR_20260924.md)
> sammen med denne historiske analysen. Den negative h12/juni-målingen består
> etter retting. Generelle utsagn nedenfor om umulig retningsinformasjon,
> bevist abstensjon og utelukket lekkasje går utover målegrunnlaget.
> Juni er gjenbrukt utviklings-VAL. Pooled-oppsettenes h12/24/48/96 er
> 1/2/4/8 timer med M5-barer; flere timeangivelser nedenfor er feil.


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

**Run1b, ridge, begge armer, 96 konfigurasjoner, 609 s [M]** (evidens
`ENTRY_DIRECTION_WALKFORWARD_20260923/run1b_ridge_both_arms/`; en første kjøring
ble cgroup-drept ved 10,1 GiB på siste MTF-konfigurasjon og er forkastet som
evidens, jf. regel 7). MTF-joinen er bevist eksakt: null avvik på 313 399 rader
for alle seks eier-aliaser (H1/H4/D1 × atr/ema).

- 100 % dekning: negativ snitt-bps i alle 96 konfigurasjoner (−0,1 … −2,8);
  excess over myntkast 0,1–1,6 bps. Ingen alle-bar-retning.
- 25 % dekning: strict_pass i høyst 1 av 4 folds i hver konfigurasjon.
- 5 % dekning: strict_pass 1–2 av 4; folds 2023-06..2025-05 negative eller
  null i nesten alle konfigurasjoner; 2025-06..2026-05 bærer edgen (+7,9 …
  +43,6 bps). Treff på realisert bedre side 0,49–0,55.
- 1 % dekning (≈645 rader/år, ≈2,6 handler/dag): **én celle er positiv i alle
  fire år**: h=12 (60 min), `atr`-skalering, `snapshot_mtf`: +6,83 / +4,31 /
  +1,39 / +17,77 bps (strict 2/4; HAC-SE 3,0 / 2,4 / 4,1 / 11,2; treff
  0,58 / 0,55 / 0,53 / 0,56; LONG- og SHORT-valgte delmengder begge positive i
  hvert år; p_long 0,88 i 2022-23, 0,11–0,15 senere). Ved 2 % faller folds
  2023-25 til ~0. h=48 `atr` 1 %: +9,8 / −0,5 / +14,2 / +63,2 (strict 2).
  h=24 `atr` 1 %: +6,2 / −0,6 / +4,0 / +34,7 (strict 1).
- Dagshorisont (288 barer): store fold-snitt (+37 … +299 bps ved 1 %), men
  sirkulær-shift-p95 er 31–110 bps og strict_pass 1/4: overlapp- og
  drift-artefakt, ikke seleksjon. Forkastes.
- `atr`-skalering mot `raw`: høyere fold-snitt ved 1–5 % for h=12/24/48/96 og
  mindre long-tilt (p_long 0,3–0,5 mot 0,4–0,6); min-fold fortsatt negativ i
  de fleste celler. Ikke et robust løft alene.
- FLAT-andel med argmax_flat: 1–38 % av holdout-radene (ridge lærer negative
  forventninger); ved 100 % dekning er snittet negativt uansett.

**Run2, HGB (scikit-learn HistGradientBoosting, biblioteksdefault lr 0,1 /
min_samples_leaf 20, iterasjoner valgt på indre kronologisk splitt), snapshot-arm,
48 konfigurasjoner, 3 605 s [M]** (`run2_hgb_snapshot/`).

- Den indre splitten valgte ett eneste tre i 30 av 48 fits og ≤ 10 trær i alle:
  ingen boosting-runde etter den første generaliserer. Prediksjonene blir
  nesten konstante; FLAT-andel 45–91 % av holdout-radene.
- Under dagshorisonten er HGB dårligere enn ridge i hver celle: 5 % dekning
  fold-snitt −0,8 … −17,5 bps (ridge +0,8 … +6,3), strict_pass 0–1 av 4.
- Dagshorisont (288): HGB 25 % `atr` +13,7 snitt / +1,4 min (strict 1), 5 %
  +41,7 / −4,1 (strict 2). Samme drift/overlapp-forbehold som over; ikke
  seleksjonsevidens.

Lesning etter run2: ikke-lineære interaksjoner på snapshot-flaten (Codex'
neste hypotese 23.09) gir ingenting utover lineær ridge med biblioteksdefault;
en langsommere variant (lr 0,03, min_leaf 200) og MTF-armen testes i run3.

Lesning: den lineære flaten har ingen fold-robust retningsedge i bredden.
Det finnes én smal lomme (60 min, topp 1 %, MTF-lag) som er positiv fire år på
rad med to signifikante år; den er valgt blant ~190 celler og må bekreftes av
en annen lærer (HGB, run2/run3) og av en uberørt bekreftelsesmåned (VAL juni
2026) før den kalles et funn.

**Run4, ridge, begge armer, bekreftelsesstadium på VAL juni 2026 (5 509 rader,
fit på 313 206 TRAIN-rader med purge), 120 konfigurasjoner, 734 s [M]**
(`run4_ridge_both_arms_valconfirm/`, instrument-sha og HEAD `0ea7c00c` i
rapporten). VAL ble aldri brukt av instrumentet før dette stadiet.

- **Ingen celle bekreftes.** Alle 96 (target × arm × skalering × dekning)
  har negativ snitt-bps ved 25/5/2/1 % dekning, unntatt tre støyceller
  (h96 snapshot atr 2 % / 1 %: +24 / +28 bps, n 111 / 56, langt under
  sirkulær-p95 76 / 87; h288 mtf raw 1 %: +76, p95 279).
- Lommen fra fold-stadiet (h=12, `atr`, `snapshot_mtf`, 1 %) gir **−42,5 bps**
  på 56 handler med **100 % LONG valgt** i en måned der p(opp) på 1 t er 0,461;
  5 %: −22,4 bps, 95 % LONG. Snapshot-armen: −33,8 / −10,2 bps.
- MTF-armen velger LONG på 70–100 % av de valgte radene i alle celler; treff på
  realisert bedre side 0,14–0,48. Fold-stadiene reproduserer run1b bit-eksakt
  for alle folds og targets unntatt fold 3 / h288 (192 ekstra holdout-rader ved
  TRAIN-slutt får etiketter med juni-priser når tapen strekkes til VAL-slutt;
  maks avvik 2,13 bps; rettes i instrumentet etter run3; ingen run3-target
  er berørt).

Lesning etter run4: den forhåndsregistrerte bekreftelsen refuterer alle
lineære seleksjonslommer på V12-flaten. Det de fire TRAIN-årene «lærte» var
drift og regime (2021–2026-bull pluss vol-eksplosjonen 2025–26), ikke en
betinget retning; i første måned med motsatt fortegn gikk alle armer LONG og
tapte. Dette er samme mekanisme som Q-målets drift-intercept i §2.

**run3a — HGB på MTF-armen (bibliotek-default lr 0,1 / min_leaf 20, seed 0, knee/2 t/4 t/8 t, VAL-stadium) [M]:**
ingen celle positiv i alle fire folds. Beste celle h48/atr/5 % er positiv 3/4
(−2,7 / +1,0 / +1,6 / +8,0) med strict 0/4 og VAL +6,1 uten strict. De to
VAL-cellene som passerer strict (h24/raw/contrast/1 %: +40,4; knee/raw/contrast/1 %:
+33,9) er negative i alle fire folds; å velge dem ville vært populasjonsfellen
fra 2026-06-23, ikke et funn. Modellen er nesten konstant: FLAT-andel ved full
dekning 0,36–1,00 per fold og 0,66–1,00 på VAL, så «1 %-dekning» er ofte tom
(NaN). Konklusjon: boosting med default-læringsrate tilfører ingenting utover
ridge på samme flate; run3b (lr 0,03 / min_leaf 200) kjører etter kjede 3 som
robusthetssjekk, ikke som håp.

### 6.2 Utvidelser 23.09 kveld (kilde-bevist, resultater i §6.3 når kjeden er ferdig)

Operatørens innvending — «mange leser momentum, flagg, FVG, støtte/motstand;
hvorfor gjør ikke vi det?» og «gå opp i tidsramme? sørg for at ingenting
svever ubrukt» — er gjort til tre målbare spørsmål i samme instrumentfamilie:

1. **Attribusjon (er noe ubrukt?)** — `--ablation {families,lanes,all}` i
   walk-forward-instrumentet: hver eier-mappede gruppe (spesialistfamilie via
   `classify_entry_specialist_feature`, MTF-lane per tidsramme, MTF-familie via
   `MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4`, mønster-blokk per tidsramme,
   `ctx_cat`, og unionene `patterns:all` / `mtf_lane:all`) fjernes én om gangen
   og ridge refittes på sub-Gram (én løsning per gruppe, ikke én ny Gram).
   Rapportert som Δ bps mot full arm per dekning og fold, sammen med en
   **kardinalitets-matchet null** (fidelity-registeret 21.09, F-6: «smc med 48
   felt mot session med 4» er ellers et telleartefakt): samme *antall* kolonner
   trukket tilfeldig, refittet ved full-modellens valgte alpha, samme
   seleksjonsregel; p05/p95 av null-Δ rapporteres og gruppens Δ flagges under
   p05 (bar verdi) eller over p95 (støy). Antall trekk er eksplisitt CLI-input
   (`--ablation-null-draws`); med 20 trekk er p05 i praksis minimum av 20 og
   feilen stor, så flagget er *antydende*, ikke bekreftende. En gruppe med
   Δ ≈ 0 over alle folds er *lineært* ubrukt på denne horisonten; det beviser
   ikke at den er ubrukt for sekvensmodellen (§8). Den native protokollen i
   fidelity-registeret §3 (≥ 200 trekk, mean-substitusjon under fittet
   normalisering, ≥ 5 seeds) forblir autoriteten; ingen familie pensjoneres av
   noen av dem (regel 4).
2. **Mønster-primitiver som input** — `gx1/scripts/research_entry_pattern_primitives_v1.py`
   bygger fra tapen, på M5/H1/H4/D1 sist-lukkede barer med eierens
   cutoff-regel: FVG (tre-lys-ubalanse) og order blocks (siste motsatte lys før
   displacement) som sporede soner med retest/feil-hendelser, avstand i ATR,
   alder og antall aktive; equal highs/lows-pools med sweep/brudd; flagg
   (impuls + stram konsolidering + brudd); N-bar-range-brudd; EMA-stabling
   (20/50/200), EMA200-helning og -avstand; forrige dags high/low/close,
   fullført Asia-range, ukeåpning med brudd-hendelser. Alle terskler er
   deklarerte CLI-input (regel 2a), ingen er tilpasset på data. Armene
   `snapshot_patterns` og `snapshot_mtf_patterns` legger blokken til de
   eksisterende armene, og `patterns:all`-ablasjonen gir baseline-armen på
   nøyaktig samme rader.
3. **Konfluens som eksplisitte regler** — `gx1/scripts/research_entry_pattern_setup_edge_v1.py`:
   36 faste oppsett (FVG/OB-retest i høyere-TF-trend på M5/H1/H4, sweep-fade
   av equal highs/lows ± H4-trend, flagg-brudd, range-brudd i H4-trend,
   PDH/PDL-brudd i H4-trend, Asia-range-brudd i H1-trend, momentum-konfluens
   H1+H4+D1, og «alle stabler bull/bear» som drift-referanse), scoret per
   TRAIN-år og på VAL med myntkast-null, **beste-konstant-side-null**
   (alltid-LONG / alltid-SHORT på de samme radene — drift-nøytral), HAC-SE og
   sirkulær-null. Ingen regel er fittet; evaluatoren teller.

4. **Kryss-asset (operatørvedtak 23.09, regel 1 uendret)** — armene
   `snapshot_cross` / `snapshot_mtf_cross`. Inventar samme kveld: ingen rå DXY-,
   rente- eller VIX-serie finnes; det som finnes er en avledet daglig tabell
   (`GX1_DATA/research/cross_asset_fred_20260615/macro_features.parquet`, Yahoo
   chart-JSON for DXY/TNX/VIX/TIP/IEF, log-nivåer, kalenderdag-union med ffill,
   2020-09-30..2026-06-16, uten manifest, råfilene borte, byggeskript kausalt)
   og USD_JPY H1 fra OANDA (`cross_asset_spike_20260609/USD_JPY_H1.parquet`,
   2020-11-01..2026-06-07, uten manifest). Blokken deklareres i instrumentet:
   daglig chg1d/chg5d/chg20d og z60 per instrument (VIX også som nivå), én
   kalenderdags lag (beslutning dag D bruker dag D−1); USD_JPY H1 sist-lukket
   under eierens cutoff med maks 72 t staleness: log-avkastning 1/4/24/120
   barer, realisert vol 24, relativ spread. Rader uten gyldig kryssdata
   utelates fra fit og holdout (VAL dekkes bare til 17.06 daglig / 07.06 H1).
   Provenienskl.: gjenfunnede forskningsbytes, ikke deklarerte data — et funn
   her er en grunn til å hente en manifestbundet serie, ikke et resultat.
   Gammel kjede (pensjonert substrat, M5-horisont): nivå/residual/Z refutert
   OOT med fortegnsflipp, kointegrert fair value refutert, `oot_macro_test`
   hadde én dags look-ahead. Prior: lav.

Samme kveld fikk `evaluate_frame` beste-konstant-side-nullen som felt
(`best_constant_side_mean_pnl_bps`, `excess_over_best_constant_bps`,
`beats_best_constant`), og fold-grensen ble tettet: en fold-holdout merker
aldri en rad med priser etter sin egen slutt, heller ikke siste fold når tapen
strekker seg inn i VAL for bekreftelsesstadiet (192 rader ved h288 i run4 ble
merket med juni-priser; regresjonstest `test_fold_stage_identical_with_and_without_final_holdout`).
Kontrollflaten fikk `model-native-pattern-primitives` og
`model-native-pattern-setup-edge` under audit-cap.

For §7.2b persisterer instrumentet nå, sammen med prediksjonene, en
etikettfri avstand per holdout-rad fra fit-periodens fordeling:
`ood_abs_z_mean` = gjennomsnittlig |z| over alle kolonner under fit-radenes
kolonne-middel/-std (to eksakte chunkede pass), og det samme over MTF-lanene
(`ood_abs_z_mean_mtf`) og mønsterblokken (`ood_abs_z_mean_patterns`) der de
finnes. Fit-perioden er alle rader før holdout-start (kronologisk prefiks);
horisont-purgen er irrelevant for kolonnestatistikk. Dette er en måling, ikke
en regel: ingen terskel er deklarert.

### 6.3 Kjede 3 (natt til 24.09): mønstre, attribusjon, kryss-asset, konfluens [M]

Primitivene på ekte tape (318 908 beslutningsrader, 235 kolonner, 4 tidsrammer):
FVG-hendelser på 11–12 % av M5-barer, order blocks 2,3 %, equal pools 2,6 %,
range-brudd 3,7 %, flagg-brudd 0,06 % (~300 på M5 over fem år, 56 på H1, 9 på
H4). D1: 14,5 % bullish mot 8,8 % bearish FVG — gullets oksemarked i tallene.

**run5a — `snapshot_mtf_patterns` (1 311 kolonner), ridge, VAL-stadium, 60 konfigurasjoner, 630 s:**
ingen celle er både fold-robust og VAL-bekreftet. Tre celler er positive i
alle fire folds (h48/atr/1 %: +13,4/+9,0/+10,1/+47,2, strict 2/4; h24/atr/1 %;
h288/raw/1 %), og de to første taper −54,7 og −33,9 bps på juni 2026 med
p_long 1,0 — samme drift-ekstrapolasjon som run4. Cellene med positiv VAL
(h288/raw/1 % +42,4, h96/raw/1 % +45,6, h48/raw/1 % +28,8) er juni-månedens
SHORT-drift: p_long 0,0–0,3, og ingen av dem slår beste-konstant-side-nullen
(alltid-SHORT på samme rader: +42,4 / +67,8 / +69,1) eller sirkulær p95. Δ mot
run4s `snapshot_mtf` på identiske folds er ±0–3 bps ved 5 % dekning og
fortegnsløst ved 1 % (n ≈ 56–700). Mønsterblokken flytter ikke den lineære
retningsprediksjonen; konfluens som eksplisitte regler måles i setup-evaluatoren
under.

**run5b — `snapshot_patterns` (551 kolonner, uten MTF), ridge, VAL-stadium:**
heller ingen celle som er fold-robust, VAL-strict og slår beste-konstant. Uten
MTF-lanene velger armen SHORT-tungt i juni (p_long lav), så VAL-cellene på
4 t–1 d blir positive (h96/raw/5 %: +90,9; h288/raw/1 %: +161), men det er
måneds-driften: alltid-SHORT på samme rader er like god eller bedre, og bare
én celle (h96/raw/1 %: +107,0, n ≈ 56) passerer VAL-strict — den er negativ
i fold 2 (−16,9) og strict bare 2/4 i folds, og under 36 × 2 sammenligninger
er én VAL-strict-celle ventet ved tilfeldighet. Ikke et funn; notert som
ikke undersøkt videre (TEST er forseglet, neste måned er eneste ferske
holdout).

**run6 — attribusjon på `snapshot_mtf_patterns`, 27 eier-mappede grupper +
kardinalitets-null (20 trekk), knee/2 t/8 t × raw/atr × 4 folds = 24 celler
per gruppe, 6,6 t kjøretid:** dette er den første økonomiske attribusjonen i
prosjektet (F-1 i fidelity-registeret), lineær og på TRAIN-folds. Målt:

- *Ingen gruppe bærer mer enn ~1 bps lineær retningsverdi.* Ved full dekning
  er alle Δ mellom −0,33 og +0,52 bps; ved 5 % dekning mellom −0,83 og +2,39.
- *MTF-lanene bærer det lille som finnes.* `mtf_lane:all` (760 kolonner):
  Δ −0,33 ved full dekning, under null-p05 i 13/24 celler; `mtf_lane:D1`
  −0,33 (7/24); `mtf_family:smc_liquidity` −0,23 (10/24);
  `mtf_family:vol_compression` −0,10 (9/24 med bare 20 kolonner). Ved 5 %:
  `mtf_family:trend_ema` −0,83, `mtf_lane:H1` −0,70, `mtf_family:chart_geometry`
  −0,65, `mtf_family:structure_swing` −0,61, `mtf_lane:D1` −0,57.
- *De samme lanene er det som knekker på VAL.* Å fjerne `mtf_lane:all` gir
  +7,8 bps på juni (over null-p95 i 6/6 celler), `mtf_lane:D1` +4,0,
  `mtf_family:smc_liquidity` +4,6; ved 5 % gir `mtf_lane:all` +16,2. Det er
  run4-mekanismen målt som attribusjon: regime-lanene gir in-sample verdi og
  ekstrapolerer ut av sample. Unntak: `mtf_family:trend_ema` er verdifull både
  på folds (−0,83) og på VAL (−6,2, under p05 i 4/6).
- *Mønsterblokken subtraherer lineært.* `patterns:all` (235 kolonner): Δ +0,52
  ved full dekning (over null-p95 i 11/24) og +2,39 ved 5 % (8/24);
  `patterns:D1` +0,29 / +1,76. Sparsomme hendelseskolonner er støy for ridge;
  på VAL er fortegnet motsatt (−1,2 / −10,1), som er samme drift-historie.
- *De lokale M5-familiene er nær null hver for seg* (|Δ| ≤ 0,3 ved 5 %); ingen
  er «ubrukt», ingen er avgjørende.

Regel 2f-ramme: HAC-SE per fold-celle er ~0,3–0,5 bps ved full dekning og
~1,5–2 bps ved 5 %, så bare de gjentatte mønstrene over (13/24, 11/24) er
utenfor støy; enkelttall er det ikke. Lineær ablasjon sier ingenting om
ikke-lineær eller sekvensiell bruk i transformeren (§8); den native protokollen
i fidelity-registeret §3 står. Ingen familie pensjoneres (regel 4).

**Fit-avstand (§7.2b) målt på run5a sine persisterte prediksjoner, MTF-laneavstand
(`ood_abs_z_mean_mtf`, kvartiler fra folds anvendt på VAL):**

- *Selvsikkerhet er en monoton funksjon av avstand.* Gjennomsnittlig |predikert
  kontrast| stiger med avstandskvartil i hver fold og på VAL (h48/atr: fold 0
  56 → 63 → 72 → 102 bps; VAL 35 → 52 → 80 → 98). De 5 % mest «sikre» radene
  ligger nesten utelukkende i øverste avstandskvartil (h48/atr fold 0: 2 024
  av 3 164; VAL: 245 av 276). Lav dekning velger per konstruksjon radene
  modellen har minst støtte for.
- *Men en per-rad-avstand skiller ikke juni.* Korrelasjonen mellom predikert
  og realisert kontrast er negativ i alle fire kvartiler på VAL (−0,06 til
  −0,21 for h48/atr; −0,06 til −0,17 for h96/raw), og VAL-radene fordeler seg
  nesten normalt over fold-kvartilene (25/19/22/33 %). Inversjonen er en
  hel-måneds regimehendelse, ikke en hale av avvikende rader; den globale
  gjennomsnitts-|z| over 760 kolonner fortynner den ene D1-ATR-kolonnen som
  faktisk lå +3,3σ ute.
- Konsekvens for §7.2: en abstensjon basert på global inputavstand ville ikke
  reddet juni. Det som skiller måneden er regimenivå (volatilitetsklokken),
  og en enkeltfelt-regel der er en håndskrevet port — forbudt som post-modell
  (regel 3) og tillatt bare som lært input. Avstandskolonnene beholdes som
  måling; ingen terskel deklareres.

**run7a — kryss-asset daglig makro (`snapshot_mtf_cross`, 1 093 kolonner: 17
makrokolonner fra DXY/TNX/VIX/realrente-proxy med én dags lag), ridge, VAL til
17.06 (3 280 rader), 60 konfigurasjoner, 56 min:** ingen celle er fold-robust
og VAL-bekreftet. Cellene som er positive i alle fire folds (h12/atr/1 %,
h24/atr/1 %, knee/atr/1 %) taper −64 / −56 / −65 bps på juni, som er 21–23 bps
*verre* enn samme celler uten makro (run4). Ablasjonen av makroblokken på
folds: å fjerne den *forbedrer* 5 %-dekning med +2,2 bps i snitt (over
null-p95 i 26/48 celler) og full dekning med +0,8 (17/48) — blokken er støy
in-sample-forward. På juni gir blokken −3,4 / −3,6 bps ved fjerning (under p05
i 7/12 ved full dekning), altså litt verdi i én måned. Konklusjon: daglig
makro med ærlig lag tilfører ingen robust retningsinformasjon på 2 t–1 d på
V12 heller; den gamle kjedens refutasjon står nå også på nytt substrat, med
riktig lag og VAL-bekreftelse. Proveniensen er svak (gjenfunnede Yahoo-bytes),
så dette er et resultat om *disse* seriene, ikke et bevis om makro generelt;
en manifestbundet realrente-serie (DFII10) er den eneste varianten som ikke
er testet.

**run7b — daglig makro + USD_JPY H1 (`snapshot_mtf_cross`, 1 099 kolonner), VAL
til 07.06 (2 024 rader, 1 %-celler tomme), 81 min:** samme bilde. Å fjerne
kryssblokken forbedrer folds ved 5 % med +2,3 bps (over null-p95 i 23/48);
USD_JPY-blokken alene er null (Δ +0,1 / +0,0, flaggene balanserte 7/7 og
8/13). På den ene VAL-uken bærer makroblokken verdi ved full dekning
(−9,0 ved fjerning, under p05 i 8/12) — én uke, ingen konklusjon.

**Konfluens som faste regler (`research_entry_pattern_setup_edge_v1`): 36
oppsett × 4 horisonter, per TRAIN-år + juni, 140 celler med måling:** ingen
celle er strict i tre eller flere år. Det trader-øyet ser, ser evaluatoren
også: LONG-oppsett i H4/D1-opptrend er positive i de fleste år —
`pdh_break_trend_H4` (første lukk over forrige dags high med H4-stabel
bullish) er positiv alle fem år ved 4 t og 8 t (8 t: +4,6/+4,6/+12,1/+5,1/+18,4
bps, n = 40–82 per år), `eql_sweep_fade_H1_trend_H4_bull` alle fem år ved 8 t
(min +5,4), `asia_hi_break_trend_H1` alle fem år ved 2 t. Men mot den ærlige
nullen — tilfeldige rader i samme år med samme side (sirkulær p95) — skiller
de seg bare ut i to av fem år (2023–24 og 2025–26), og drift-referansen
`trend_all_bull_any_bar` (alle barer med H1+H4+D1 bullish) passerer strict i
nøyaktig det samme sterkeste året. Med n ≈ 50 per år er HAC-SE 4–7 bps, så
en 5-bps-effekt er ikke målbar per år (regel 2f), og over 144 celler er seks
«5/5 positive» ventet av drift og tilfeldighet alene. SHORT-oppsettene er
negative i nesten alle år (oksemarked). Juni har null rader for LONG-oppsett i
opptrend (ingen PDH-brudd med H4-bull i en nedmåned), så VAL kan ikke
bekrefte noe. Klasse: *ikke demonstrert ved denne styrken*, ikke «refutert»
for `pdh_break_trend_H4`; alt annet er drift. Ingen regel er fittet; ingen
terskel-sveip er kjørt eller skal kjøres uten forhåndsregistrering.

**Regimeklokken målt (§7.2b, neste instrument): prediksjonskvalitet som funksjon
av D1-ATR-z alene (`atr_bps_14` i D1-lanen, sist-lukket, z under hvert
stadiums fit-periode), run5a-prediksjoner, h12 og h48 atr:** klokken skiller
ikke. Juni 2026 ligger på z 1,5–2,1 under full-TRAIN-statistikk (ikke +3,3σ;
det tallet var mot en annen populasjon), og begge juni-binnene inverterer
(korr −0,10 til −0,18). Men fold 3 (2025–26) hadde 44 % av radene over z 3
(snitt 6,0) og var *positiv* der (h48: korr +0,07, topp-5 % +6,4 bps), og i
fold 1–2 er korrelasjonen per bin fortegnsløs (−0,16 til +0,23 uten mønster).
Konsekvens: en regime-nivå abstensjon på volatilitetsklokken har ingen støtte i
folds; det som skiller juni er ikke vol-nivået, men at en LONG-tilbøyelig
modell fittet på et oksemarked møtte en nedmåned. Det er trendvending, og
ingen input i flaten varsler den. §7.2b nedgraderes fra «neste byggetrinn» til
«ikke støttet av målingen»; det som gjenstår der er nettomål slik at FLAT
konkurrerer (§7.2a).

**Kjede 3, samlet:** mønstre som input flytter ingenting lineært; attribusjonen
viser at ingen familie bærer over ~1 bps og at MTF-lanene er både kilden til
in-sample-verdi og til juni-inversjonen; per-rad-avstand skiller ikke
regimet; daglig makro med ærlig lag tilfører ingen robust verdi; konfluens
som regler er drift ved målbar styrke. «Retning fra pris alene på V12» og
«retning fra pris + disse makrobytene» er begge refutert ved det
forhåndsregistrerte kriteriet (fold-robust + VAL-strict). Det som står igjen
som ubevist, ikke refutert: `pdh_break_trend_H4` ved 4–8 t (for få rader),
regime-nivå abstensjon (§7.2), og en manifestbundet realrente-serie.

**run3b — HGB på MTF-armen med lr 0,03 / min_leaf 200 (3,8 t):** samme dom som run3a.
Beste celle h24/atr/contrast/1 % er positiv 3/4 folds (7,8/8,9/−6,2/30,5,
strict 3/4) og +19,8 på juni uten strict; h48/atr/argmax/1 % 3/4 og −26,8 på
juni. Ingen celle er 4/4 og VAL-strict. FLAT-andelen ved full dekning er
fortsatt 0,3–1,0. Boosting, uansett læringsrate, tilfører ingenting utover
ridge på denne flaten.

**Konfluens-oppsett, samlet femårsperiode (kjede 4, `setup_edge_pooled_v1`,
144 celler, én periode 2021-06..2026-05):** 18 celler passerer strict, alle
LONG i H4/D1-opptrend, ingen SHORT. `pdh_break_trend_H4` passerer på alle fire
horisonter (n = 299; 2 t +4,7, 4 t +7,2, 8 t +9,6, 1 d +13,3 bps; t mot
myntkast 3,5–4,2; 3,8–6,4 bps over sirkulær p95, dvs. over tilfeldige LONG-rader
med samme antall; HAC-SE 1,9–3,8). Men drift-referansen `trend_all_bull_any_bar`
passerer også strict ved 8 t og 1 d (n = 64 779; +1,8 / +5,2 bps, t 5,0 / 6,3),
og `momentum_confluence_long` og `range_break_up_M5_trend_H4` likeså. Lesning:
i den samlede perioden ga «LONG når H4/D1 peker opp» 2–7 bps mer per 4–8 t
enn tilfeldige LONG-rader, med PDH-bruddet som den sterkeste og sjeldneste
varianten. Det er trendfølging i et oksemarked målt over fem år: effekten er
konsentrert i 2023–24 og 2025–26 (§6.3, per-år strict 2/5), har ingen
SHORT-speiling (H4-bear-stabler var for sjeldne til å måles), og juni 2026 har
null rader å bekrefte på. Klasse: *målt positivt i samlet TRAIN, ikke bekreftet
ut av sample, ikke skilt fra regimedrift ved årlig oppløsning.* Det er
kampanjens eneste positive måling, og den er svak: ved n = 299 er 5 bps ≈ 1,5
SE. Den kan bare avgjøres med framtidige rader; ingen terskel-sveip skal
kjøres på den (det ville være tilpasning).

**run7c — kryss-asset på den manifestbundne FRED-tabellen (operatørvedtak 24.09
«kjør alt dette»; DFII10 realrente, DGS10, DTWEXBGS bred dollarindeks, VIXCLS;
publikasjons-skiftede tilgjengelighetsdatoer + én dags lag; 313 399 gyldige
TRAIN-rader, full juni-VAL 5 509 rader; 57 min):** samme dom som run7a, nå på
rene data. Ingen celle er 4/4 og VAL-strict; de 3/4-cellene taper −39 til −114
bps på juni. Δ mot ren MTF-arm på identiske rader er negativ på folds ved 5 %
dekning (h48/atr: −6,2 / −10,0 i fold 2–3), og ablasjonen sier det samme:
å fjerne FRED-blokken forbedrer folds ved 5 % med +4,5 bps i snitt (over
null-p95 i 30/48 celler) og ved full dekning +0,4 (19/48). På juni bærer
blokken −4,0 bps ved full dekning (under p05 i 8/12) — én måned. Konklusjon:
daglig makro med ærlig lag, fra en kilde med manifest, tilfører ingen robust
retningsinformasjon på 2 t–1 d; den skader in-sample-forward. Kryss-asset-
hypotesen er dermed refutert på V12 ved det forhåndsregistrerte kriteriet,
både på gjenfunnede og på manifestbundne bytes. Det som ikke er testet:
intradag makro (ingen kilde uten megler), posisjonering (COT, ukentlig) og
horisonter over én dag (for få uavhengige observasjoner i fem år).

### 6.4 Brukbarhetsrevisjon av hele beslutningsflaten (24.09, operatør: «dobbeltsjekk at ALLE funksjonene er brukbare») [M]

Kjørt på de ekte V12-TRAIN-bytene: 313 399 rader × 1 072 kolonner (241 signal +
71 ctx_cont + 4 MTF-laner × 190, sist-lukket under eierens cutoff). Evidens:
`GX1_RUNS/V12_EPOCH1_REVIEW_20260923/FEATURE_USABILITY_AUDIT_20260924/`
(`columns.csv` per kolonne, `fidelity.csv`, `duplicates.json`, `families.csv`,
script arkivert samme sted). Lett, lesende diagnostikk utenfor capped runner
(run3b holdt låsen); ingen beslutningsverdi.

- **Fidelitet: 48 av 49 uavhengige omregninger treffer Pearson 1,0000.** ATR14
  i bps, EMA20/50/200-avstand i ATR, EMA200-helning (k = 20), momentum 5/20,
  RSI14 sentrert og Bollinger-bredde, på M15/H1/H4/D1, regnet på nytt fra
  tapen gjennom eierne (`wilder_atr`, `wilder_rsi`, `classic_ema`) og samplet
  med cutoff-regelen; M5 `ret_1`/`ret_20`/`rvol_20`, `ctx_cont.atr_bps`,
  `close_return_5_bps`, H1/H4/D1 ATR-bps-projeksjonene: alle 1,0000.
  `hour_sin`/`hour_cos`/`dow_sin` 0,992–0,997 mot UTC-klokka (minutt-komponent),
  `D1_dist_from_ema200_atr` 0,996 (mid mot close, F-24, kjent). Den ene under
  0,9 er revisjonens egen proxy-feil (EMA20-helning målt mot EMA200-referanse).
  Sesjons-id ↔ UTC-time er kontiguøs (0: 22–06, 1: 06–11, 2: 11–15, 3: 15–20).
  **Feltene beregner det navnene sier.**
- **Liveness: 0 døde kolonner.** 34 kolonner har én verdi på ≥ 99 % av radene,
  og alle er sjeldne hendelsesflagg (EMA-kryss, `level_present`,
  `geomline_retest_fail`, D1-divergens) — sjeldne av natur, ikke døde; for en
  lineær lærer er de nesten tomme, for transformeren er de embeddings.
- **Duplikater: 71 eksakte grupper og 38 nær-par (≥ 0,999), alle etter design.**
  57 er ctx_cont-aliasene som ligger bit-likt i signalblokken (register §1,
  «71/71 alias overlap by design»), 14 er ctx-skalarprojeksjoner av MTF-felt
  (`_v1h1_ema_diff` ≡ `H1:macd_line_atr`, `_v1h1_atr_bps` ≡ `H1:atr_bps_14`,
  `*_trend_state_age_bars_v2` ≡ `TF:trend_state_age_bars`,
  `*_ema_stack_aligned_v2`, `*_rsi14_canon_v2` ≡ `TF:rsi14_centered`). Ingen
  uventet duplikat; 130 par ≥ 0,98 inkluderer registerets F-12-klasse.
- **Korrelasjonskontroll, ikke lekkasjebevis.** 19 kolonner har |ρ| > 0,05
  mot framtiden, alle mot 1-dags-avkastningen (0,052–0,075). Dette omfatter
  trege D1-trend-/nivåtilstander med varierende fortegnsstabilitet.
  Korrelasjonsstørrelse og stabilitet kan ikke utelukke lekkasje. Ingen
  terskel ved 0,1 har en slik beviskraft. Korreksjonen over dokumenterer
  faktisk fremtidig feature-tilpasning relativt til de indre/årlige kontrollene.
- **Informasjon om fortiden: ja.** Median maks-|ρ| mot forrige 1/12/48/288
  barer er 0,125; 293 kolonner over 0,3 (momentum/trend-familiene 0,4–0,6);
  142 under 0,02 (sesjon, vol-tilstander, sjeldne flagg — forventet).
- **Informasjon om framtiden: nesten ingen, og det lille er drift.** Mot
  8-timers-kontrasten: median |ρ| 0,0056, p90 0,021, maks 0,045
  (`D1:mtf_level_above_touch_count`); 127 kolonner over 0,02, 3 over 0,04.
  Mot 1 time: maks 0,028, ingen over 0,04. 26 kolonner har |ρ| > 0,02 med
  samme fortegn i ≥ 5 av 6 perioder — alle er H1/H4/D1-trend, EMA-avstand og
  nivå-tilstander, dvs. «hvilken vei har markedet gått i det siste», og
  minste års-|ρ| ligger på 0,004–0,016. iid-SE per år er 0,0044, så disse er
  reelle men små, og de er nøyaktig regime-featurene som ekstrapolerte feil
  i juni (§6.3).

**Svar på operatørens spørsmål, i tre klasser:** *målt*: alle 1 072 kolonner
er levende og bærer informasjon om fortiden. De 49 referansesjekkene gir
avgrenset formelstøtte; de dokumenterer ikke generell kausalitet eller
numerisk paritet for alle feltene; *målt*: ingen kolonne bærer
mer enn |ρ| 0,045 om de neste åtte timene, og median er 0,006; *ikke
undersøkt*: fidelitet for konsepter uten uavhengig referanse (SMC-nivåer,
geometri, divergens-sider) er bare sjekket for liveness og duplikater, ikke
for semantikk. Det er derfor ikke grunnlag for å konkludere at alle
inputs er feilfrie eller at ingen nyttig retningsinformasjon kan finnes.

## 7. Hva bekreftelsen endrer i planen: fra tap/mål-varianter til abstensjon og informasjon

Run4 er det avgjørende funnet i dette dokumentet. Fold-lommen som så robust ut
(h12/atr/mtf/1 %, positiv 4/4 år) tapte −42,5 bps på den urørte juni-måneden
med 100 % LONG i en nedmåned, og de lagrede prediksjonene viser mekanismen:
samme modell som ga kontrast +1…+4 bps på fold 3 ga +12…+27 bps på VAL, mens
korrelasjonen mot realisert utfall snudde fra +0,04…+0,08 til −0,08…−0,13.
Det er lineær ekstrapolasjon av bull-drift på regime-features som lå utenfor
fit-fordelingen (D1-ATR +3,3σ). Snapshot-armen uten MTF holdt seg nær null og
tapte mindre. Dette er samme «confident-tail inversion» som den pensjonerte
kjeden så i 2026.

Konsekvenser, i rekkefølge:

1. **Tap/mål-varianter kan ikke skape fortegnsinformasjon.** Per-rad
   ATR-skalert tap, per-prøve lært log-varians og en lærer-fri kontrollarm
   (forrige versjon av denne seksjonen) omvekter radene; de kan ikke gjøre en
   feature-flate med korrelasjon +0,05 in-sample og −0,1 out-of-sample til
   retning. De beholdes som *diagnostiske* armer i instrumentet, ikke som veien
   til edge. Ingen av dem bygges i treneren før instrumentet viser en
   fold-robust *og* VAL-bekreftet celle for dem.
2. **Abstensjon er det eneste beviste stedet å hente verdi, og den må være
   regime-nivå, ikke rad-nivå.** Målt: argmax velger aldri FLAT i dag
   (brutto-mål, lærer-optimisme +9 bps), og den mest selvsikre halen er den
   som inverterer. Målt i §6.3: selvsikkerhet er monoton i fit-avstand, men en
   per-rad-avstand skiller ikke juni — hele måneden inverterer. En Entry som
   failer closed der *regimet* er utenfor fit-fordelingen ville ha tapt 0 i
   stedet for −42,5 bps på VAL. Regel 3 tillater ingen post-modell-terskel;
   abstensjonen må derfor sitte i selve Q-målene: (a) nettomål der kost inngår
   slik at FLAT = 0 faktisk konkurrerer, og (b) regimeklokken som lært
   betingelse — men (b) er målt i §6.3 og *ikke støttet*: D1-ATR-z skiller
   ikke gode fra dårlige perioder på folds (2025–26 var +6σ og positiv; juni
   var +1,9σ og invertert). Det som gjenstår er (a), og en ærlig erkjennelse
   av at trendvending etter et oksemarked ikke varsles av noen input i flaten.
3. **Horisont.** Kostnadsaritmetikken (§1) gjør 4 t–1 d til den eneste skalaen
   der 52–54 % treff slår spread. Der er drift- og overlapp-nullene
   strengere: h288 feiler sirkulær-null, og myntkast-nullen er for snill mot
   drift. Beste-konstant-side-nullen (§6.2) er lagt inn nettopp derfor; ingen
   celle på ≥ 8 t regnes som funn uten å slå den.
4. **Ny informasjon er operatørens beslutning.** Alle fire lærere lander på
   50–52 % treff per bar på samme flate; tuning og modellbytte flytter ikke
   taket. Det som kan flytte det er informasjon flaten ikke bærer. Regel 1
   forbyr andre instrumenters markedsdata i Entry; en eventuell åpning er en
   kontraktsendring, ikke et research-valg.
5. **Dynamikk** (FLAT-absorberende seleksjon, min-epoker før patience) bevarer
   signal; de skaper det ikke. Uendret prioritet: etter 2.

Alle native sammenligninger må evalueres med samme fold-semantikk som
instrumentet (research-evaluering over eksplisitte TRAIN-tidsvinduer) og med
VAL som *bekreftelse*, aldri som seleksjon — ikke på 512 VAL-rader.

## 8. Ikke undersøkt, sagt uoppfordret

- Ingen native modell er trent med noen av variantene i §7; ingen OOD-diagnostikk (§7.2b) er bygget eller målt.
- Ablasjonen er lineær (ridge): en gruppe med Δ ≈ 0 kan fortsatt bære ikke-lineær eller sekvensiell informasjon for transformeren.
- Mønster-primitivene er én deklarert parametrisering per konsept; ingen terskel-sveip er kjørt, og det skal ikke kjøres uten forhåndsregistrering (ellers er det tilpasning på TRAIN).
- Instrumentets `snapshot_mtf`-arm bruker sist-lukkede per-TF-rader, ikke
  modellens 64/96/96/252-barers sekvenser; en sekvens-effekt kan ikke utelukkes.
- Close-fill-returns på tapen er en research-konvensjon (ikke M1 neste-åpning-fill).
- Multiple sammenligninger over horisonter × dekninger × lærere er ikke
  korrigert; en enkelt strict_pass er ikke et funn, kun gjentatte over folds.
- Kryss-asset, nyheter og mikrostruktur ble refutert på den pensjonerte kjeden;
  ikke re-testet på V12.
