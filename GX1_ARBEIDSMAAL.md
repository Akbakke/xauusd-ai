# GX1 arbeidsmål — 2026-09-16

Utvikle en modell som gir positiv kostnadsjustert netto Bps gjennom selektive,
retningsmessig gode Entries og Exit som realiserer forventet videre nettoverdi.
Robusthet og kvalitet er viktigere enn handelsantall. Intradag er ønsket stil;
M5 er verken pålagt holdetid eller eneste nyttige tidsramme.

Gjeldende arbeidsstrategi er: **Krev målbar læring før mer omfattende trening.**
Teknisk sammenkobling, lavere treningsloss på et repetert lite utvalg, en lærer
som velger FLAT, eller GPU-PASS er ikke tilstrekkelig bevis på handelsfordel.
Se docs/LEARNING_GATE_20260916.md for beslutningsgrunnlag og neste måling.

## Nåværende arbeid

Teknisk reference32/split16+16 og paret95→96-måling er ferdige. Resume består,
men læringsresultatet begrunner ikke større trening. Lokal Entry-fit bedres;
bred Entry er litt verre og allFLAT. LONG Exit flytter grunnverdien opp og blir
allHOLD. Se CURRENT_HANDOVER.md og FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json.
Ikke gjenta kontrollene. Årsaksdiagnosen i docs/VALUE_LEARNING_CAUSE_20260916.md
er ferdig: svak frossen videreverdi dominerer Entry, lærerpolicyen gir sideavhengig
effektiv Exit-targetlengde, og verdihode/representasjon flytter grunnnivåer.
Separat Exit-klipping ble målt og prøvd i én native32-kandidat, men er forkastet.
Den forbedret Exit-fit litt på faktisk trent512 og forverret separat TRAIN128
med samme femstegsmål. AllHOLD for LONG og allEXIT for SHORT er ikke kvalitet;
Entry er fortsatt allFLAT. Standardkoden tilbakeføres, alle resultater bevares.
Se docs/EXIT_PRIVATE_CLIP_LEARNING_20260916.md. Målkomponentene er nå målt i
docs/TARGET_COMPONENT_CAUSE_20260916.md. Ankerutfall er også kontrollert på
275/512 Entries; se docs/ENTRY_ANCHOR_OBSERVED_OUTCOMES_20260916.md. Neste er
én begrunnet korreksjon i målkjeden, ikke flere gjentatte diagnosekjøringer. Ingen nye treningsforsøk,
full epoch/fullVAL, target-refresh, replay eller tapsvektsøk nå.
NEXT_RUN_POLICY.json gjelder.

Dersom prognosene lærer og Entry/Exit fortsatt ikke gjør det, revurder konkret
læringssignal og verdifordeling før mer beregning. En enklere oppdeling av Entry
og Exit er en mulig senere beslutning, ikke en bestilling på nye modeller nå.
Manglende læring skal ikke møtes med blinde epocher eller mer kompleksitet.

## Bevarte krav

- Alle 200 features, åtte familier og tidsrammer beholdes. Deres samarbeid må
  etter hvert begrunnes i målte resultater; tilkobling alene viser ikke nytte.
- Entry har et selvstendig forecastsignal, men handelsverdiene bruker fortsatt
  Exit-læreren. Delvis gradientisolasjon er ikke full uavhengighet.
- «Ingen fast grense»: ingen fast tapsgrense eller maksimal holdetid.
  Exit sammenligner videre nettoverdi med gjennomførbar lukking. V4-regnskapet
  står i docs/RISK_OBJECTIVE_20260914.json. Fem beregningssteg er ikke et tidsstopp.
- Pris-/kostnadsregnskap og successor-semantikk bevares. Samlet økonomi omfatter
  realisert cash og korrekt åpen verdi for hele den valgte kohorten.
- TRAIN-utvalget for kalibrering er 2025-06-01 inklusiv til 2026-06-01 eksklusiv:
  65 295 rader av opprinnelige 313 399. Hele parenthistorikken og normaliseringen
  beholdes. Femårsvektene er initialisering; modellen har allerede sett mer enn
  ett år. Juni 2026 er gjenbrukt utviklings-VAL. TEST forblir forseglet.
- Når læring og påkrevde tekniske porter er dokumentert: avgrenset ettårsvurdering
  før større omfang. Det langsiktige målet er full femårstrening, opptil 30 epocher,
  VAL etter hver epoch og early stopping med patience 5. Dette er ikke starttillatelse.
- Kun native campaign, TRAIN16/VAL256/8CPU/3h, FP32/TF32 av og eksisterende vakter.
  Ingen live-/papirhandel, spending eller TEST-bruk.

Første gamle femårs-epoch og juni-VAL er bevart. Den gamle epoch2 stoppet på
checkpoint315, offset320, totalt19 908 steg. Gammel juni analyseres med første
epochs uforanderlige EMA: 2 227 lukket og 3 281 HOLD av 5 508 valgte handler.
Full-policy netto Bps for den gamle kjøringen er ikke tilgjengelig. Dette må
ikke forveksles med dagens rettede økonomimål eller checkpoints95/134/96/97.

Eneste kodebase: /home/andre2/src/GX1_CURRENT, work/gx1-current. Én agent og én
tung jobb. Gjenbruk verifisert arbeid og oppdater handover uten historiske
«gjeldende»-instrukser. Stående autorisasjon og alle bevaringskrav gjelder.

Oppdatering: eksisterende120-minuttersprognose har kostnadsjustert TRAIN-signal
samtidig som Entry er allFLAT. Se docs/FORECAST120_ECONOMIC_SIGNAL_20260916.md
(for filer under docs: FORECAST120_ECONOMIC_SIGNAL_20260916.md). To sensurerte
forløp gjør samlet sluttidsregnskap ufullstendig. Ingen læringsport er åpnet.
Neste arbeid er én kausalt og matematisk begrunnet rettelse av videreverdien;
frosne prognoser og alle ferdige målinger skal gjenbrukes.

Referansemålets matematikk og kompakte native datatilkobling er nå CPU-kontrollert.
26 nye dataflyttilfeller består;120 rewards krever bare én ekstra boundary-
tilstand utover dagens current/successor. Native trainer er fortsatt sperret
for denne referansen. Neste er uttrykkelig Q_mu-/recipe-/checkpointbinding og
frosne sammenlignbare mål før én avgrenset kritikerplan. To uendrede eldre
runner-tekstasserts er dokumentert separat; ingen fullsuite-PASS påstås.
Se CONTINUATION_REFERENCE_CANDIDATE_20260916.md i docs og kvitteringen
handover_snapshot/REFERENCE_POLICY_DATA_FLOW_20260916.json. Ingen lærings-PASS.
