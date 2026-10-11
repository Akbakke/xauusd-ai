# CURRENT HANDOVER — 11.10.2026 (kveld)

Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current. NEXT_RUN_POLICY.json er
eneste arbeidsstatus; prosess og terminal overstyrer prosa. Historikk per forsøk
ligger i Git og i policyens bindinger, ikke her.

Nå: rebuild og baseline er akseptert, HEAD-avvikene er rettet (modellfunksjon v10)
og v10-nullstegsbaselinen er målt. Swing-programmet har funnet et long-timing-
signal på 20 dagers horisont som er støttet, men ikke bekreftet; flaten er frosset
og forward-journalen er eneste bekreftelsesrute. Native neste steg er bundet som
design. Ingen dokumentert Entry-edge, ingen prediktor innrømmet, training_enabled
= false.

## Rettelser i HEAD 11.10.2026 (operatørbestilt)

Kode og tester i 56d87f49; filhasher og begrunnelse i
NEXT_RUN_POLICY.current_work.head_deviation_fixes_20261011.

1. Objective-kontrakt v11 bærer `entry_training_phase` (entry_only uten Exit-
   gradienter, joint med frossen Exit-lærer); treneren sender fasen den kjører.
2. Alle encodere avsluttes med parameterfri LayerNorm (F12).
3. Vindus-readout er middel pluss siste lukkede bar på alle fire steder (F27);
   arkitekturversjon v10, samme parametre og state_dict-nøkler, annen funksjon.
4. Vekt-EMA bias-korrigert med min(decay, t/(t+1)) (F18).
5. Checkpoint-policy v6–v8: konstant Entry-handling på VAL kan aldri velges (F15).
6. Forskningseieren merker degenerert referanse eksplisitt.

Ikke endret: hjelpehodene i Entry-only, slippage 2,0 bps, patience/min_delta.
Gamle v9-checkpoints avvises ved bundle-last. Samplerbenchmark: exit 0 på 20ab5fe5,
valgt batch 16 med målt CPU-epoke 7679 s, modelluavhengig.

## v10 nullstegsbaseline 11.10.2026: målt, ingen læring

FRESH_INITIAL_BASELINE_003 (83adab26, eget 6-timers CPU-budsjett PREPARATION_CPU_002,
lukket) målte den ferske v10-funksjonen på samme korrigerte inputs, normalisering og
TRAIN4096/CONTROL4096-kohorter som 002: 9 637 663 parametre, 513 forwards, én
TARGET-paritetsforward med alle hoder bitidentiske, null optimizersteg, 339 s i
producer20G. TRAIN argmax LONG/SHORT/FLAT 1878/733/1485 (002: 2532/717/847), 1046
av 4096 endret; CONTROL 1779/830/1487 (002: 2317/884/895), 1139 endret. Side-MSE
uendret (≈917/878 TRAIN, ≈1877/1879 CONTROL). Utrent initialtilstand; ingen
treningsautoritet.

## Swing-program 11.10.2026: målt status

Kilder (manifestbundne, hashbundne): COT deacot 2009–2026; OANDA H1 for XAU og 10
panelinstrumenter 2009-06..2026-06-30 (USD_CNH feilet lukket); Treasury par-/real-
kurver 2009–2026; Cboe VIX; GLD via Internet Archive (til 2026-02-13) og SPDR-
arbeidsbok (til 2026-10-08); FOMC-sider; Fed H.10 daglig bred dollarindeks. FRED
svarte ikke fra verten; CME og BLS er bak bot-sperrer.

Resultater, alle målt på gjenbrukt utviklingshistorikk 2011–2025 (15 analyser på
samme data) pluss ett ubrukt forward-halvår; eksakte tall og bindinger i
policyblokkene swing_d1_stage_a/b/c, swing_pooled og swing_d1_stage_c_followups:

- Porteføljefamilien (full dekning mot alltid-LONG) er NO_GO eller INKONKLUSIV i
  alle forsøk. Bulk rank-IC ≈ 0.
- Selektivt ved h20: HGB-effekten fra stage A/B overlevde ikke familiejustering
  (minste familie-p 0,23) og replikerte ikke på tvers av instrumenter.
- XAU swing_ridge h20 er treff-familiesignifikant: 008 (73 felt, kanonisk M5-rute)
  0,781 ved 5 % dekning, familie-p 0,043; 009 (to-års folds) 0,872/0,831/0,784/0,698
  ved 2/5/10/25 %, familie-p 0,022/0,0085/0,007/0,026. Fanget mål er aldri
  familiesignifikant under sirkulær-shift-nullen.
- Diagnose: valgte rader er 87–90 % LONG, og alltid-LONG på samme rader treffer
  likt. Effekten er NÅR en 20-dagers long betaler, ikke retning. Blokkmatchet
  long-null: fanget mål p 0,018/0,030 (008) og 0,003/0,004 (009) ved 5/10 %.
- Ablasjon (5 %, treff/familie-p): uten kryss-panel 0,647/0,83 (borte); uten
  drivere 0,721/0,27; uten COT 0,746/0,14; uten kalender 0,776/0,050; uten
  dollarindeks 0,831/0,011. Flaten fryses derfor til 68 felt: A, COT, panel,
  drivere.
- Forward 010 på 2026H1 (108 rader, ubrukt): rank-IC 0,44, treff 0,91 ved 10 %
  (11 rader, 5 episoder) og 0,70 ved 25 % i et halvår med 36 % positive
  20-dagersbevegelser. Forhåndsregelen ble ikke oppfylt (blokk-null-p 0,065 ved
  25 %); INKONKLUSIV med deklarert lav styrke, støttende på alle metrikker.
- Episodeøkonomi (beskrivende, 10 %/20 d, ex finansiering ≈ 30 bps per long-
  episode): 95 episoder +119 bps (008), 90 episoder +169 bps (009), 5 episoder
  +803 bps (010).

Verdikt: ubekreftet long-timing-spor. Ingen prediktor innrømmet.

## Forward-journal (vedtak 11.10 kveld)

Eier `run-swing-forward-journal` (ada3b553) på den frosne 68-feltsflaten, XAU via
H1-ruten, swing_ridge h20/h5. Policyblokk forward_journal_20261011 autoriserer
lesing av forseglet periode bare for de navngitte kildene til scored_end_utc
2026-10-11. Neste: fem kildemanifester (OANDA full refetch, COT 2026 med gjenbruk
av 2009–2025, Treasury 2026 med gjenbruk, VIX, GLD-arbeidsbok), journal-
forhåndsregistrering, henting under capped audit, scoring av 2026-07..2026-09 én
gang. Fit skjer bare på historikk før hver måned; scorede rader brukes aldri til
fit.

## Native D1-kontekst og horisont (vedtak 11.10 kveld)

Design bundet i docs/NATIVE_D1_CONTEXT_HORIZON_DECISION_20261011.md: observert
markedsmål på 1/5/20 handelsdager og en niende kontekstfamilie (4 COT + 44 panel
+ 13 drivere = 61 felt) med forskningsflatens as-of-regler. Seks steg, hvert med
egen binding og budsjett. Ingenting lansert.

## Gjort og akseptert

Featureauditen på 569d0054 reviderte alle 254 lokale signalfelt, 71 eksakte
kontekstaliaser, ett sesjonsfelt og 190 MTF-felt. Fire funksjonsfeil og misvisende
metadata er rettet. 141 fokuserte tester besto; metode, hvert felt og avgrensninger
står i [docs/FEATURE_AUDIT.md](docs/FEATURE_AUDIT.md) og feltregisteret ved siden av.

Ny generasjon: HISTORY2009W_FEATURE_REPAIR_20261010. Korrigerte MTF-/M5-/M1-felt,
berørte labels og immutable kildebindinger er publisert gjennom eksisterende eiere.
Alle 5 523 147 komplette pre-TEST M1-rader og 392 143 437 aliasverdipar er kontrollert.
Alle 652 552 TRAIN-/70 880 VAL-sekvenser og snapshots er rekonstruert mot korrigert M5.
Kanonisk readiness, spesialistruting og hele fysiske TRAIN-/VAL-indeksen består.

Base-normalisering er tilpasset én gang på hele fysiske TRAIN før sampling:
652 552 Entry-rader, 955 670 unike M5-inputrader og 3 978 505 unike M1-inputrader.
Fersk TRAIN-summary-fit og splitbindinger er publisert uten ny basefit.
VAL-/TEST-fit er null. Faktiske normaliserte NumPy-/PyTorch-input, bitlike aliaser,
kausale MTF-ruter og Entry/Exit-førstetilstand består.

FRESH_INITIAL_BASELINE_002 (25cde83c, pre-v10-funksjon) og FRESH_INITIAL_BASELINE_003
(83adab26, v10-funksjon) er begge målt på de samme 4096 TRAIN-/4096 CONTROL-
koordinatene med bitidentiske targets; 003 er gjeldende funksjonsbaseline (tall i
avsnittet om v10-baselinen ovenfor). Initialprediksjoner er ikke læring eller edge.

Samlet aksept og uavhengig kontroll av metrics, koordinater, tilstandsbytes og ti
vellykkede terminalkvitteringer:
/home/andre2/GX1_RUNS/FEATURE_REPAIR_REBUILD_20261010_001/ACCEPTANCE_REVIEW_001/RESULT.json
Alle kjørebevis og originale avvik ligger i samme run-root.

Brukerbestilt omstart etter ferdig baseline er verifisert: Windows boot
10.10.2026 21:51:28 UTC (23:51 norsk tid), ny WSL-boot og signert GPU-telemetri.
Maskinens prosjektjobber, GPU-beregninger og prosjektlåser var ledige før omstart.
Alle 160 native kildebindinger og 30 sentrale filer, inkludert initialtilstandens
bytes, er uendret. Fsync-publiserte kontrollkvitteringer er intakte. Native
treningsoppgave er deaktivert; omstarten startet eller autoriserte ingen trening.
Bevis: samme run-root, HOST_BOUNDARY_002/POST_REBOOT.json.

## Bevart avvik og grenser

Trygg Windows-/WSL-omstart ble gjennomført mellom core og komplett M1. Tre ferske
kontrollfiler uten fsync var tomme etter omstart; tomme originaler er bevart,
observasjoner gjenopprettet fra Mac og atomisk publisering brukt videre.
POSTBUILD_REVIEW_001 fikk OOM ved feil 4 GB-klasse etter beståtte komplette
sekvenskontroller. Kun uferdig spesialistaudit ble kjørt i eksisterende producer10G.
Første baselineoppstart manglet eksplisitt CPU-synlighet og stoppet før konstruksjon.
Etterfølgeren skjulte GPU før PyTorch-import; øvrige metoder og vakter var uendret.

Swing-forsøk 001 (instrumenthelligdager) og 007 (eieren krevde D1-lane) feilet ved
start og er bevart uten resultat; 013 feilet lukket i porteføljefamilien (ridge-
porteføljen gikk til null egenkapital fra 2017-04), og dets selektive tall er regnet
post hoc fra lagrede prognoser med eierens funksjon. GLD-beholdningen fra Internet
Archive slutter 2026-02-13; carry-forward holdt den flat til 2026-06-30 i forsøk
008–015 og forward-sjekken 010 (gld_log_change21D = 0 der). Arbeidsboken fra SPDR
(data til 2026-10-08) brukes av journalen. Kommandomeldingen 18856469 oppgir feil
ubetinget referanse (+29 bps); riktig er +63 bps.

Ukjente rådatagap er fortsatt ukjente; ingen imputering eller antatt markedslukking.
Deklarerte kostvilkår er ikke bevis for historisk kostsannhet. CONTROL er gjenbrukt
utviklingsdata. Ingen Exit-rollout, all-trades-/åpen-posisjonsøkonomi eller profitt
ble målt. Nye baselinekoordinater gir ingen native trenings-/sampler-/GPU-autoritet.
Tidligere benchmark- og treningsbevis tilhører sin gamle generasjon.

## Læring og neste beslutning

Forrige læringsstudie ga REJECT_ENTRY_QUALIFICATION: CONTROL4096 valgte 4095 FLAT
og én SHORT. Retnings-MSE 7357,64 mot konstantens 7360,93 ga ingen statistisk påvist
forbedring. v10-baselinen endrer ikke denne konklusjonen. Flere epoker på
95-minuttersmålet er ikke neste steg; neste native steg er D1-kontekst- og
horisontbølgen (vedtak 11.10 kveld), som starter først med egen kontrakt og budsjett.

Rebuild-/baselinemålet er ferdig; engangsplaner og budsjetter er lukket, også
PREPARATION_CPU_002. Valgt sampler (batch 16, målt CPU-epoke 7679 s) bindes i
recipe først i den native bølgen. Native launch er stengt.
Ingen automatisk trening, forlengelse eller relansering av konsumerte planer.
training_enabled=false; TEST er forseglet. Ingen live/paper, broker, ordre, spending,
promotion eller opprydding. Bevar én agent, én tung CURRENT-jobb, alle genuine
features, åtte familier, kausalitet, vakter, originalkvitteringer og checkpoints.
Ingen fast tapsgrense eller maksimal holdetid.
