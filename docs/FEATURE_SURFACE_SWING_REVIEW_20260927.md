# Gjennomgang av featureflate og grunnmur mot swing — 27.09.2026

Operatørens spørsmål: er oppsettet (alle ~250 M5-felt, kontekstlanene, fusjonen, arkitekturen)
riktig for å gå long i stigende og short i fallende marked på 1–4 ukers horisont? Fem uavhengige,
lesende revisjoner (basis/candle/lokal kontekst; nivå-/trendlinjeregistre; swing/SMC/momentum/
squeeze; MTF-laner og duplikater; arkitektur og mål) pluss egen kontroll av de tyngste påstandene.
Sammensetning lest ved å eksekvere eierne: signal v36 = 24 basis + 150 obligatoriske (10 familier)
+ 67 kandidater = 241 per M5-bar i en 96-barers sekvens; 190 felt per tidsramme for M15/H1/H4/D1;
71 ctx_cont. Evidensklasser: [S] bevist fra kilde, [M] målt på ekte data, [U] ubevist.
Målepopulasjon: V46/PRETEST_V12-artefaktene (2019-11 → 2026-06); 2011–2018 er ikke målt, så alle
epokeeffekter under er nedre grenser.

## Kort svar

Nei — en lengre horisont på dagens M5-oppsett er ikke nok. Det som er solid er fundamentet
(kausalitet, utførbar bid/ask-økonomi, lineage, fail-closed, walk-forward-instrumentet). Det som
er feil for swing er designet rundt: beslutningsklokke, mål, Exit, modellstørrelse og en stor del
av featureflaten er bygd for scalping, og flere felt blir epokeklokker på 2011–2025.

## Egen kontroll av de tyngste påstandene (27.09)

- [M] Gull 2011-06-01 → 2025-05-30: 1 535,84 → 3 289,70, log-drift **10,43 bps/uke**. Kostpolicyens
  LONG-finansiering 5,4 %/år = **10,35 bps/uke**. Med dagens kostpolicy tjener alltid-LONG ~0 netto
  over perioden; finansieringsantakelsen avgjør regnestykket. Ukes-sd 210 bps; 2011-09→2015
  −23,5 bps/uke, 2019→2025-05 +28,2 bps/uke. Kostpolicyen er et øyeblikksbilde fra 2026
  (`historical_rate_series_complete: False`); 2011–21 var nullrenteår.
- [S] Hver MTF-lane midles over hele historikken (`entry_v10_ctx_hybrid_transformer.py:2139`,
  `encoded.mean(dim=1)`): en hendelse på siste lukkede D1-bar veier 1/252 med mindre attention
  løfter den.
- [S] `gx1/scripts/augment_forward_outcome_v2.py` satte `/home/andre2/src/GX1_ENGINE` først i
  `sys.path` ved import (parbyggeren importerer den) — **rettet i samme commit** (repo-roten
  utledes fra fila; 93 tester grønne).
- [S] Kontraktinkonsistens: `entry_fitted_q_v1.py:79` «gross_spread_inclusive_research_only» og
  `:134` `fixed_512_capacity_forced_terminal_present: True`, mot `unified_exit_fitted_q_v1.py:73-86`
  netto cash og `chunk_capacity_is_terminal: False`. Én er foreldet — må rettes før neste trening.

## Arkitektur og mål (misforhold med swing)

- [S] Ryggraden er M5 × 96 barer (8 t) × 241 felt; ~119 700 inputverdier per beslutning, største
  blokk M5 der drift/sd er 0,004–0,016 og trendpersistens 0,49–0,50 [M].
- [S] Entry-målet er den frosne Exit-lærerens verdi (bootstrap); Exit-referansepolicyen holder med
  sannsynlighet 119/120 per M1-steg (forventet ~2 t), 480 M1-sekvens og 512 stitilstander. En
  ukesposisjon er ~7 200 M1-steg — max-optimismen i bootstrap-en akkumuleres (+5,34 mot −3,42 bps
  allerede ved ~2 t [M], SNR-diagnosen).
- [S] Knee-horisonten måler volatilitet, ikke retning, og taket (96) er lånt fra aux-hodene
  (`aux_targets_v3.py:94` → `entry_direction_target_policy_v1.py:39`); aux-hodene ser 1–96 M5-barer
  fram og trekker representasjonen mot intradag-volatilitet.
- [Aritmetikk] ~730 uavhengige uker i 2011-06..2025-05; modellen har 9,6 mill. parametere
  (dokumentert) ≈ 13 000 per uavhengig ukesetikett. Ridge på 1 311 felt kollapset til konstanten [M].
  Forventet OOS-R² ≈ R² − p/n tilsier ≤ ~15 felt for ukesbeslutninger.
- [Aritmetikk] VAL (juni 2026) ≈ 4 ukesbeslutninger, TEST (jul–aug) ≈ 9: bare fornuftssjekker;
  autoriteten må være forhåndsregistrerte årsfolds.

## Featureflaten — defekter etter klasse

**Epokeklokker på 2011–2025** (lar modellen lære «hvilket år» i stedet for markedstilstand):
- Spreadblokken: `spread_bps` (era-AUC 0,23), `spread_extremes_sum_bps` (0,18), `spread_bps_delta_1`,
  `quote_range_asymmetry_bps` [M]. Hører hjemme i kostmodellen, eller i ATR-/relativ form.
- Volatilitetsnivå i bps: `atr_bps` (AUC 0,67), `rvol_20` (0,63), `_v1_pk_sigma20` (0,65), innbyrdes
  ρ 0,94–0,98 [M]; `volatility.bandwidth_rel` (median ×3,3–3,5 på H4/D1 2019→2026) og
  squeeze-tilstanden (HMM på absolutt båndbredde: H4 90 % i squeeze i 2019 → 20 % i 2026, D1 100 % →
  16 %) [M]. Behold én, i relativ form (persentil eller forhold til egen 252-D1-median).
- Returer/avstander i bps: `ret_1`, `ret_20`, `close_return_3/5_bps`, akselerasjon,
  `close_distance_from_ema5_bps`, `_v1_bb10_bandwidth_change_3` (IQR ×1,35–1,46) [M].
- Rundt-tall-rutenett i faste USD 50/100 (lokalt og alle laner): D1 median |d50| 0,52 → 0,09 ATR
  2021→2026 [M]; ved 1 200 vs 4 300 USD betyr feltet noe helt annet [S].
- `level_*_recurrence_dist_atr` avhenger av historikkens lengde (nivåer beskjæres aldri): D1-median
  0,10 → 1,07 [M]. Levetidstilpasningen for nivåer skalerer med TRAIN-lengde (D1 valgt på n = 11) [S+M].
- Tick-tetthet: candle-veke-nullmasse 12,4 % → 3,2 %, `open_above/below_previous_*` 4,4 % → 1,2 % [M].

**Registre som mettes eller måler støy** (nivå/trendlinje, lokalt og H4/D1):
- Nivåbrudd uten bekreftelsesbånd: 9,6–17 % av alle barer per side, `bars_since_break` median 1–3
  barer; «retest» = neste bar som spenner nivået (69 % løses på neste bar, 33/33 hold/fail) [M].
- Trendlinjer uten gyldighetssjekk mellom ankerne: median 17 (M5) / 81 (D1) aktive linjer per side;
  brudd på 31 % av D1-barer; `retest_hold` er stort sett et ekko av bruddet [M].
- `geomchan_pos_0_1` klippet til 0/1 på 34–36 % av aktive rader (H1/H4/D1); `max_dev` sensurert ved
  båndet [M]. Nivåer er enkelt-pivoter (ingen klynger), 3-barers fraktal på alle klokker [S].
- CHoCH er bare alternasjon (50 % av M5-brudd, 42–44 % H4/D1 — tilfeldig-gang-raten) [M].

**Duplikater** (koster kapasitet og attribusjon): `ret_20` ≡ f(`mom_20_atr`, `atr_bps`);
`close_return_5_bps` ≡ f(`mom_5_atr`, `atr_bps`); `close_return_3_bps` ≡ ret_1-lagger;
akselerasjon ≡ Δret_1; `close_distance_below_high_range_fraction` ≡ veke + kropp;
`close_range_observed` (1 på 99,97 %); `smc_choch` ≡ up + down; `level_below_present` ≡
1[touch_count>0]; `m5_rsi14_canon_v2` ≡ affin kopi av lokal `rsi14_centered` i samme plan;
8 event↔styrke-par og `structure_bias`↔`swing_state` på H4/D1 eksakte; `dist_to_R1..S2` 4 felt med 3
frihetsgrader; lane-klynger ρ 0,95–0,99 (rsi14 ~ ema20_dist ~ vwap20 ~ bb_position) [M].

**Andre feil**: `_v1h1/h4_slope3/5` er differanser av en allerede ATR-normalisert serie (F-21-klassen,
bare rettet for D1; fortegnet snur på 4,5–8 % av radene) [M]; ctx-avstander til D1/H4-struktur
skalert med **M5**-ATR og udeklarerte lookback-literaler [S+M]; «VWAP» er tick-vektet snitt av
close [S]; sesjons-/timefelt uten sommertid, og døde på H4/D1-beslutningsrader [M]; `dow_sin` bruker
kalenderdøgn mens D1 bruker 22:00-UTC-handelsdøgn [M].

**Det som holder** [S/M]: kausalitet i alle familier; eksakte long/short-speil i koden (observerte
ubalanser er oksemarkedet, ingen side er strukturelt null); ATR-normaliserte felt er epokestabile
(sent/tidlig 0,91–1,13); `smc_swing_state`-inversjonen er rettet og felt går via embedding;
squeeze-absorpsjonen er rettet; én Wilder-ATR-eier.

**Horisont** [M]: alle lokale M5-felt har median alder ~1 t og p99 < 9 t — tidsinput, ikke
ukesinformasjon. Bare H4/D1-lanene bærer ukesskala (D1 brudd-alder median 9–14 dager).

## Det som mangler for en swing long/short-modell (innen XAU-only, regel 1)

- Tidsserie-momentum over 21/63/126/252 D1-barer i volatilitetsenheter (kjernehypotesen «long i
  stigende, short i fallende»).
- Posisjon i 52-ukersrange / avstand til 252-dagers høy og lav (dekker også drawdown-tilstand).
- Relativt volatilitetsregime (ATR20/ATR252, vol-of-vol) i stedet for bps-nivåer.
- Forrige dag/uke/måned høy-lav-close (deklarert i nivåregisteret, aldri bygd), eventuelt en
  siste-lukket-bar-token per H4/D1-lane i stedet for bare snitt.
- Historisk finansieringsserie til kostmodellen (ikke som input; krever navngitt kilde og manifest).
- Ikke anbefalt: mer mikrostruktur, sesongdummier (~14 observasjoner per måned), håndskrevne
  regimebøtter.

## Anbefalt rekkefølge (regel 22: mål først, minste endring først)

1. Rett kontraktinkonsistensen (gross/net, terminal ved 512) — ingen ny fil.
2. **Måling A uten modell**, forhåndsregistrert, direkte på 2009-tapen på D1-klokke (ingen
   seq513-rebuild nødvendig): alltid-LONG, alltid-FLAT, TSMOM-fortegn 1/3/6/12 mnd og kombinasjon,
   200-dagersregelen; ikke-overlappende 1/2/4 uker; årsfolds som dekker 2011–15; netto bid/ask +
   2 bps; to finansieringsscenarier (2026-øyeblikksbildet og 0). Dette avgjør om det i det hele tatt
   finnes en retningsgevinst i XAU-data utover drift.
3. Kun ved GO på A: en kompakt arm (≤ ~15 forhåndsdeklarerte felt) i det eksisterende
   walk-forward-instrumentet, målt mot TSMOM, ikke bare alltid-LONG.
4. Kun ved GO på 3: ukesmålkontrakt, operatørvedtak om Exit-klokke, feature-reparasjonene over
   (epokeformer, registre, duplikater), deretter rebuild og en liten D1-ryggrad — ikke 9,6 mill.
   parametere på M5.

Den pågående seq513-bootstrapen (C0 → par → kjede) bruker mange timer på en M5-flate som er
feiljustert mot målet; den bør vente til måling A er gjort (operatørvedtak).

## Ikke undersøkt

2011–2018-oppførselen for alle felt; faktisk edge/attribusjon per felt; om M5 gir timinggevinst;
effekten av mean-pooling; D1-historikken på ukeshorisont; Exit-lanene; parametertallet er hentet
fra `docs/RESIDUAL_NORMALIZATION_FIXED256_20260917.md`, ikke kjørt på nytt.
