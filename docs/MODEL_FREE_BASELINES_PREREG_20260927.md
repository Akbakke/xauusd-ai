# Modellfrie grunnlinjer — forhåndsregistrering 27.09.2026

Committet før noen resultatdata er lest. Formål: avgjøre med enkle, forhåndsbestemte regler om det
finnes en retningsgevinst i XAUUSD etter kost — først på scalp-horisont (operatørens førstevalg),
deretter på swing-horisont — før mer bygges (regel 22; se
[FEATURE_SURFACE_SWING_REVIEW_20260927.md](FEATURE_SURFACE_SWING_REVIEW_20260927.md)).
Instrument: `gx1/scripts/research_model_free_baselines_v1.py` (gjenbruker tape-leser og kostfunksjoner
fra `research_entry_direction_walkforward_v1.py`).

## Data og vindu

- Tape: `GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_20260927` (vedtak
  `OANDA_PAIR_PRETEST_2009_20260927`), lest til og med 2025-05-31 (TRAIN-slutt). VAL (2025-06 →
  2026-06) og TEST leses ikke; VAL er reservert for bekreftelse av et eventuelt GO.
- Evalueringsvindu: beslutninger med inngang ≥ 2011-06-01 og utgang < 2025-06-01. Historikk før
  2011-06 brukes bare til signaler.
- Fylling: long kjøper `ask_close`, selger `bid_close`; short motsatt (samme formel som instrumentet).
- Kost: bundet policy `LIFECYCLE_V2_FULL_TRAIN_20260912/PROSPECTIVE_COST_POLICY_V1/policy.json`
  (sha a48f8e56…): 2 × (slippage 2 bps + provisjon 0) per handel pluss finansiering på veggklokketid.
  Scenario A = policyen (LONG 5,4 %/år, SHORT 0). Scenario B = finansiering 0 (policyen er et
  2026-øyeblikksbilde; 2011–21 var nullrenteår). FLAT = 0 uten kost.

## Scalp-arm (M5, referanse = FLAT/0) — 26 celler

- MOM(k, h) og REV(k, h): ved M5-close t, posisjon = fortegn(mid_t / mid_{t−k} − 1) (REV: motsatt),
  hold h barer; k ∈ {1, 6, 12}, h ∈ {6, 12, 19} (30 min, 1 t, 95 min — 19 er den tidligere valgte
  knee-horisonten). Bare sammenhengende barer (ingen hull i t−k … t+h); ikke-overlappende, grådig
  kronologisk. 18 celler.
- ORB(London 07:00 UTC, New York 13:00 UTC): range = høy/lav (mid) av de 12 M5-barene fra
  sesjonsstart; inngang ved første M5-close utenfor rangen i bruddretningen; utgang ved siste bar
  før 16:00 (London) / 20:00 UTC (NY). Én handel per sesjon per dag. 2 celler.
- SESSION(vindu, side): Asia 22:00→07:00, London 07:00→13:00, NY 13:00→20:00 UTC; inngang ved close
  av første bar som starter ved/etter vindusstart, utgang ved close av siste bar som starter før
  vindusslutt (den daglige pausen gjør at baren før 22:00 mangler om sommeren); long og short.
  6 celler.
- Faste UTC-tider uten sommertid (kjent begrensning, F-25-klassen).

## Swing-arm (D1-klokke, referanse = alltid-LONG) — 36 celler

- D1-bar = handelsdøgn fra 22:00 UTC (samme konvensjon som eierne); beslutning og fylling ved siste
  M5-bar i D1-baren.
- Signaler på mid-close: TSMOM_n = fortegn(close_t / close_{t−n} − 1), n ∈ {21, 63, 126, 252} D1-barer
  (≈ 1/3/6/12 mnd); COMBO = fortegn(snittet av de fire); SMA200 = fortegn(close_t − snitt av siste
  200 D1-closer). 6 signaler.
- Varianter: LS (long ved +, short ved −) og LF (long ved +, ellers flat). Horisont H ∈ {5, 10, 20}
  D1-barer (1/2/4 uker), ikke-overlappende fra første gyldige beslutning. 6 × 2 × 3 = 36 celler.

## Statistikk og beslutningsregel (låst)

- Per celle: n, snitt netto bps per beslutning, sd, t = snitt / (sd / √n) (ikke-overlappende
  blokker). Swing: parvis differanse Δ mot alltid-LONG i samme blokker. Scalp: mot 0. Rapporteres
  også brutto (før kost) og treffrate.
- Per år (fulle år 2012–2024): andel år med positivt snitt (scalp) / positiv Δ (swing).
  Bjørnefold 2011-09-01 → 2015-12-31 rapporteres for swing.
- 62 celler totalt; Bonferroni ensidig 0,05/62 → **t ≥ 3,15**.
- **GO** (celle): t ≥ 3,15 i scenario A, og ≥ 60 % positive fulle år; for swing i tillegg Δ > 0
  i scenario B. **LOVENDE**: t ≥ 2,0 med de samme tilleggskravene. Ellers NO-GO.
- Beslutning: GO på scalp → scalp-sporet fortsetter med disse reglene som referanse for modellen.
  Ingen scalp-GO men swing-GO → høyere tidsrammer. Ingen GO → retningsgevinst i XAU-data er ikke
  påvist med enkle regler; modellarbeid må da begrunnes med noe nytt, ikke flere features.
- Robusthet (ikke del av GO): swing-snitt over alle startforskyvninger 0 … H−1.

## Styrke

Swing: ukes-sd ≈ 210 bps (målt 2011–25); med ~730 uker gir t = 3,15 en detekterbar Δ ≈ 25 bps per
uke for H = 5. Scalp: tusenvis av handler per celle; detekterbar gevinst ≈ 3,15 × sd/√n.
