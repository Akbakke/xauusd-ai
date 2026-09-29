# Hvorfor GX1 ikke finner edge — full repo-gjennomgang 28.–29.09.2026

Operatørens spørsmål: *«Målet er en automatisk bot som handler gull. Jeg sliter med å skaffe
edge. Hva gjør jeg galt? Gå gjennom hele repoet, vær kritisk, let aktivt etter feil og mangler.»*

Analysert kilde: `/home/andre2/src/GX1_CURRENT` @ `a3fcaf7a` (28.09). Repoet har siden gått til
`c6a7b1ab` (29.09): v37-datasettet er ferdig (TRAIN 652 552 rader 2011-06..2025-05, VAL 70 880),
readiness og lifecycle-bindinger består, kompleksitetsvurderingen er lukket, `training_enabled=false`.
Ingen av funnene under er endret av disse commitene; der det er relevant er ny status notert.

Evidensklasser per regel 2d: **[S]** bevist fra kilde/algebra, **[M]** målt på ekte deklarerte
bytes (populasjon og dato oppgitt), **[I]** velbegrunnet slutning, **[U]** mekanisme påvist,
størrelse umålt. Alt er lesende; ingen trening, ingen skriving under GX1_DATA.

## 0. Metode og hva som er verifisert

22 uavhengige, lesende revisorer (16 linser over delsystemer + 6 hull-linser fra en
fullstendighetskritiker) leverte 236 råfunn, sammenslått til 177 unike grupper (F01–F129 fra
hovedrunden, G01–G48 fra hull-runden). Hvert funn skulle angripes av to uavhengige motstandere:
én mot kildekoden (er faktapåstanden riktig?) og én mot materialitet (rotårsak eller symptom?).

- **G01–G48: fullstendig motstanderverifisert.** 46/48 faktapåstander bekreftet eller delvis
  bekreftet fra kilde; 2 tilbakevist (G33, G38). Materialitetslinsen klassifiserte 39 som
  «bekreftet fakta, men symptom for edge-spørsmålet». Detaljer i §9.
- **F01–F63 (alle CRITICAL og HIGH): fullt motstanderverifisert 29.09** — 57 kanoniske
  grupper med dobbeltdom: 0 tilbakevist på kilde, 37 materielle (PLAUSIBLE) og 20
  fakta-bekreftet, men klassifisert symptom/immaterielt. I tillegg har fem MEDIUM-grupper
  (F65, F66, F67, F68 og F70) bare én dom. Statusfilen dekker dermed 62 grupper.
  De øvrige MEDIUM/LOW-gruppene mangler full dobbeltdom; alle mottatte dommer ligger i
  `GX1_RUNS/EDGE_ROOT_CAUSE_REVIEW_20260929/verdicts_partial.json`.
  For de øvrige har jeg selv stikkprøvet 25 kjernepåstander direkte mot HEAD (A-1/A-2/A-3-kode,
  gate-normalisering, EXIT_NOW ≡ 0, horisontarv, kontrakttekst, konstantport og alpha-grid,
  kostkonstanter, tapsflagg, nullinitialisering, mean-pool, aux-fyll, tre fyllkonvensjoner,
  slope-defekt, D1-kontekstfelt, ranking-artefakt, VAL-størrelse, skjemabump-telling,
  optimizersteg i policy, trener-LOC, pytest testpaths, paritetssplit, statistikk-vokabular,
  lommegate). Alle stemte. Resten er én revisors funn med fil:linje, ikke uavhengig bekreftet;
  de er merket **[én revisor]** der de bærer en konklusjon.

Denne rapporten tar ikke stilling til om gull *kan* handles med edge. Den svarer på hvorfor
dette repoet, med disse reglene og denne prosessen, ikke kan finne den eller ikke kan oppdage
den om den finnes.

## 1. Kort svar

Prosjektet har ikke ett problem, men fem som låser hverandre:

1. **Formuleringen som faktisk bygges er målt uoppnåelig med høy styrke.** Retning i gull på
   5 min–8 t fra pris-alene med markedsordre på retail-CFD-kost krever 55–66 % treff; alle
   undersøkte metoder (ridge/HGB walk-forward, ~190 modellfrie/makro-/mekanismeceller
   2011–2025) landet på 50–52 %, trend-persistens er 0,488–0,500 på alle horisonter og år, og
   den sterkeste av 67 kandidatfeatures har |Spearman| 0,018 mot 95-min-utfallet på 994 500
   rader. Det eneste konsistente intradagsignalet (fortsettelse etter brudd, +1,0–2,6 bps mid,
   t 2,3–3,8) er mindre enn spreaden i fire av fem celler. London-bruddet har +0,28 bps
   etter spread, før slippage; alle fem er negative i det rapporterte lave
   slippage-scenarioet. Cellene er valgt etter inspeksjon og er ikke bekreftet på VAL.
   Dette er ikke bevis for at en sekvensmodell aldri kan — repoet forbeholder det korrekt — men ingen dokumentert
   hypotese sier hvorfor den skulle finne det ~200 celler ikke fant, og v37 gjenskaper samme
   M5×96×242-flate med 96-bar-taket i målkontrakten. [M]

2. **Modellen har aldri fått et mål den kan lære retning fra.** Entry-målet er den frosne
   Exit-lærerens verdi ved første tilstand etter fill: ≈ −5,85 bps likvidasjon + 0,025 bps
   fortsettelse. Med fasit i hånd velger målet selv FLAT på 914 av 1 024 rader. FLAT 256/256 er
   det korrekte svaret på spørsmålet som stilles, ikke en treningsfeil. Horisonten (95 min) er
   arvet fra hjelpehodenes vol-horisont 96 barer og Exit-referansens 1/120, ikke fra noen
   retningsmåling. [S+M]

3. **Nesten ingen læring har skjedd, og infrastrukturen er flaskehalsen.** ~26 500
   optimizersteg totalt på fire måneder (≈ 1,35 epoke-ekvivalenter av 313 399 rader), null fulle
   epoker på dagens kilde, null steg siden 19.09, `training_enabled=true` i ~18 timer totalt.
   Alle native konklusjoner kommer fra 256–512-stegs prøver på 256 rader som selv lå i
   treningsutvalget. 13 skjemabump på 71 dager har hver gang ugyldiggjort alle datasett; siste
   (v37) ble utløst av 7 av 6 019 349 rader. 47 % av commits siden juni endrer bare .md/.json.
   Treningsdynamikk-reparasjonene fra 19.–20.09 (A-1..A-4) er ikke i HEAD. [M+S]

4. **Måleinstrumentene på ukeshorisont kan ikke se en realistisk edge, og NO-GO leses som
   «lukket».** Ukesporten krever 26 %/år meravkastning over kjøp-og-hold for å bestå (power
   0,7 % ved en ekte 5 %/år-edge); HISTORY2009W 12 %/år (P(bestå) ≤ 13 %); modellfri swing
   7–9 %/år (power ≈ 10 % ved observert +9,1 bps/uke). Ridge kollapset til konstanten i 36/36
   ukesfits fordi alpha-gridet stopper ved 10⁴. Alltid-LONG-referansen betaler rundtur hver blokk
   og 5,4 % finansiering (2026-snapshot) også på nullrenteårene 2011–21, så en FLAT-beslutning
   krediteres 14–17 bps/uke den ikke skulle hatt. Intradag-nullene er derimot velpowered; det er
   ukesmålingene — der repoet selv sier retningen ligger — som ikke er det. [M+S]

5. **Reglene stenger de eneste plausible edge-kildene og alle billige eksperimenter.** Regel 1
   forbyr realrenter, USD, ETF-strømmer og posisjonering som inputs (kryss-aktiva ble bare
   refutert på 2 t–1 d, aldri på uker). Regel 3/4/17 og «aldri fjern kapabilitet» forbyr
   ablasjon, regimebetinget FLAT og forenkling. «Ingen fast horisont» gjør at det eneste
   instrumentet som kan måle retning isolert aldri får være autoritet. Ingen av reglene har
   evidensklasse eller målt begrunnelse. [S]

Det som er solid: kausalitet i alle featurefamilier og MTF-join (bevist eksakt), utførbar
BID/ASK-økonomi, fail-closed lineage, forhåndsregistreringsdisiplinen 26.–27.09 og
walk-forward-instrumentet. Fundamentet er bra. Det er bygd for feil oppgave og brukes ikke til
å måle det som teller.

## 2. Rotårsak 1 — problemformuleringen

**Påstand.** Pris-alene-retning i XAUUSD på M5-klokke med markedsordre er ikke en oppgave der
noen undersøkt metode har dekket kostnaden, og repoets egne målinger har høy styrke på dette.

**Evidens [M].**
- Break-even treff for en fortegnspredikator ved 95 min: 54,8 % (kun spread 1,68 bps) / 66,2 %
  (spread + 2 bps slippage per utførelse); strengt for fortegnshandel på Gauss ρ ≥ 0,074/0,25.
  Beste målte kronologiske treff 50,2–52,1 % på ≤ 4 t; 55,8 % på 1 d = driften
  (`docs/ENTRY_DIRECTION_SNR_DIAGNOSIS_20260923.md` §1, TRAIN 2021-06..2026-05, 354 570 M5-rader).
- Snitt/std for ikke-overlappende vinduer: 0,004 (5 min), 0,016 (95 min), 0,064 (1 d), 0,14
  (1 uke), 0,39 (1 mnd), 0,67 (kvartal). Autokorrelasjon |ρ| < 0,006 lag 1–12, variansratio
  0,97–1,01 (`docs/DIRECTION_TIMESCALE_20260926.md` §1).
- Fortegns-SNR i målet d = L − S: 0,011; corr(|d|, framtidig vol) 0,562; 5 % av radene bærer 58 %
  av Σd² (SNR §2, 313 399 rader).
- Sterkeste kandidatfeature mot 95-min sidemargin: |ρ| 0,0183 (`ctx_cont.D1_dist_from_ema200_atr`),
  deretter 0,0150/0,0150/0,0145/0,0144; ranking-artefakt 27.09, 994 500 TRAIN-rader 2011–2025.
  175 av 242 felt (basis + obligatoriske) har ingen univariat måling. Kompleksitetsvurderingen
  29.09 viser ingen eksakte duplikater på 652 552 rader — det sier ingenting om beslutningsverdi.
- Modellfrie grunnlinjer 0/62, makro 0/18, intradag 0/61, uke 0/48, tidlig kalibrert ridge/HGB
  2015–25: HGB +1,34 bps mot alltid-LONG i 5/10 år. Post hoc: fortsettelse etter brudd +1,91
  bps mid (t 3,80, 11/13 år) mot rundtur-spread 2,72 bps i bruddøyeblikk.

**Hvorfor det låser alt annet.** Ingen kapasitet, feature-reparasjon eller tapsvariant løfter
IC fra ~0,01 til 0,06–0,25 hvis informasjonen ikke finnes i prisen på den skalaen. All tid brukt
på arkitektur, normalisering, gater og lineage for denne horisonten er brukt på en oppgave som
repoets egne målinger 23.–27.09 falsifiserte for alle undersøkte metoder.

**Hva som er uavklart (ikke refutert).** D1/W1-klokken er målt NO-GO tre ganger (uke 0/48,
modellfri swing 0/36, HISTORY2009W 5/10), men under porter med power 1–13 % ved realistiske
effekter, med en referanse som krediterer FLAT 14–17 bps/uke for mye, med ridge som kollapset
til konstant av instrumentgrunner, og med kun XAU-felt. FEATURE_SURFACE_SWING_REVIEWs
anbefaling var betinget og ble fulgt: måling A ble kjørt og endte NO-GO under disse portene,
så pkt 3–4 ble korrekt ikke startet. Spørsmålet er derfor ubesvart, ikke lukket (§5, §12).

**Hva som må endres.**
- Ingen optimizersteg på v37 uten en ny, forhåndsregistrert mål-/horisontkontrakt: dagens
  objective v10 binder fortsatt 96-bar-taket og lærermålet (F08, korrigert av motstanderrunden).
- Gjør ukesspørsmålet avgjørbart før noe bygges (§12 pkt 2–3), i stedet for å velge klokke
  ved vedtak.
- Vær ærlig i GX1_ARBEIDSMAAL om a priori-nivået: et enkelt-instrument-gullsignal på uker har
  realistisk Sharpe 0,2–0,5 og kan ikke bekreftes med t ≥ 3 på 14 år (IR ≈ IC·√BR, BR = 52/år).
  Alternativene er (A) lav-SR long/flat-trend + vol-targeting med litteraturprior som port, eller
  (B) porteføljeformulering der XAU er én av flere instrumenter (krever regel 1-endring). (F37,
  [én revisor])

## 3. Rotårsak 2 — målet modellen trenes mot

**F03/F09/F28 — Entry-målet er ≈ −kost + støy [S+M, kilde stikkprøvet].** Entry LONG/SHORT-target
= første utførbare likvidasjon + (119/120)·Q_mu(HOLD) fra frossen Exit-lærer ved state 0
(`unified_exit_random_access_training_v1.py:822-836` «first_values + liquidation»,
`entry_fitted_q_v1.py:81-89`). Målt på 1 024 cachede TRAIN-Entries: likvidasjon −5,853 bps (std
4,36), lærerens fortsettelsesverdi +0,025 bps (maks +0,094); LONG-lærerens positive videreverdi
eksakt 0 på 512/512. En konstant har lavere MSE (19,05) enn modellen (19,20). Entry kan ikke ha
mer tilstandsinformasjon enn læreren, og læreren har ~0 (korr 0,03–0,08). Motstanderrunden (G01)
klassifiserer dette som *symptom* av rotårsak 1 — kost > predikerbar bevegelse på skalaen — og
påpeker at konsekvensen allerede er trukket i VALUE_LEARNING_CAUSE §1 og DIRECTION_TIMESCALE §5.
Det materielle restpunktet: objective v10 binder fortsatt dette målet for v37.

**F09 — Exit er en «forutsi fortegnet på neste 2 t»-oppgave [S+M].** EXIT_NOW-verdien er
eksakt 0 (`unified_exit_reference_policy_v1.py:141`, `random_access_model_v1.py:40`), så HOLD/EXIT
= sign(Q_HOLD) = fortegnet på forventet 120-min-drift; break-even 54–64 %, målt treff 51,6–51,8 %.
Optimal policy er side-konstant, som er det som ble målt (LONG HOLD 89,84 %, SHORT 0,00 %).
Septembers arbeid på klipping, EMA, normalisering og lærer-refresh angrep en treningsfeil som
ikke finnes.

**F22/F52 — horisonten har ingen retningsbasert opprinnelse [S+M, kilde stikkprøvet].**
`ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS = int(MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS)`
(`entry_direction_target_policy_v1.py:39-41`) der aux-taket er max(12, 48, 96) fra
vol-hjelpehodene (`aux_targets_v3.py:10`). Knee-søket valgte 19 ≈ tak/5 og måler volatilitetens
første-passasjetid (discovery 20 % ved h=1, 75 % ved h=19, 90 % ved h=96). Exit-episoden er
kappet ved 512 M1-tilstander. M1-utfallsprimitivet krever veggklokke-kontinuitet uten
stengingsautoritet (`entry_causal_m1_outcomes_v1.py:305-321`), så en ukes-etikett er ugyldig for
alle rader; fit-populasjonen for knee/hurdle er 205 002 av ~354 570 rader (57,8 %), skjevt mot
tid-på-døgnet.

**F53/F101 — MSE i rå bps på et endimensjonalt, tunghalet mål [S+M].** corr(L, −S) = 1,000, så
tre Q-verdier parametriserer én frihetsgrad + kost. Gradienten domineres av magnitude,
beslutningen trenger fortegn. Krymping mot −3,9 bps (prediction_std 2,74 vs target_std 20,4)
garanterer FLAT mot et eksakt 0-anker. Kontrakten forbyr både klassifikasjonstap og
målnormalisering (`training_objective_v1.py:28-33`, stikkprøvet).

**F20/F74 — kontrakten beskriver ikke det utførte målet [S, stikkprøvet].** `entry_fitted_q_v1.py:79`
sier `gross_spread_inclusive_research_only`, V = max, gamma 1, raw_bps; det aktive målet er netto
Q_mu diskontert med ρ = ln(1,1)/år (`unified_exit_fitted_q_v1.py:31` «discounted_net_cash_pnl_utility_bps»);
`:134 fixed_512_capacity_forced_terminal_present: True` mot Exit-kontraktens
`chunk_capacity_is_terminal: False` (:86). Påpekt 27.09, ureparert 29.09. FLAT = 0 står i
udefinert enhet; ingen test krysser kontraktene.

**F94/F95/F73 — hjelpemålene former representasjonen mot vol og etterpåklokskap [S, stikkprøvet].**
Ti oppgaver på én delt z, alle ≤ 96 barer; Entry-Q er 13 % av det samlede råtapet. Flere
hjelpemål er betinget på den i ettertid vinnende siden (`_quality_side = _direction_side.copy()`,
builder :788) — samme Jensen-struktur som det forkastede Y_wait (+13,7 bps). Aux-hodene fylles på
beslutningsbarens close (`entry_mid = close[:valid_rows]`, builder :561) mens primærmålet fyller på
neste M1-open; testen låser dette.

**F92/G21 — lærerens konstante bias bestemmer siden [M+S].** Samme måldefinisjon ga snitt
−5,39/−6,27 (16.09) og +5,34/+7,03 (23.09); sidevalget flipper med lærerens bias (SHORT 512/512 →
FLAT 256/256), ikke med markedstilstanden. Single-net max uten Double-Q; 36,6 % av målet er
bootstrap ved 120 steg ((119/120)^120).

**Hva som må endres.**
- Én kontraktsfil for et deklarert, observert Entry-mål: netto utfall ved bundet horisont på den
  klokken som skal undersøkes, med kostpolicy-sha, klokkeeier og enhet; objective/readiness/aux
  refererer den (F93). Ukesetiketter krever closure-bevisst klokke (gjenbruk
  `unified_exit_market_closure_authority_v1`) (F52).
- Exit i forskningsfasen = måleinstrument (fast horisont eller 119/120-referanse uten lært
  komponent), ikke lært policy. «Ingen fast holdetid» kan stå for handelsregelen; den skal ikke
  forby instrumentet (F39/G36).
- Parametriser Q som (nivå, kontrast): Q_L = μ̂ − ĉ, Q_S = −μ̂ − ĉ, Q_FLAT = 0, fortsatt én argmax;
  skaler målet med realisert vol i tapet (F13, F53).
- Fjern selected-side-hjelpemålene; gi aux samme fyll som primærmålet (F95, F73).
- Rett kontraktteksten og skriv kryss-kontrakt-testen i samme bølge (F20/F74).

## 4. Rotårsak 3 — læring har ikke skjedd, og infrastrukturen er flaskehalsen

**F01/F19/F55 — omfanget av faktisk trening [M].**
- Lifecycle-v2/native totalt: 19 908 steg (fullepoke 12.–14.09, kilde GX1_VAL_PAUSE_ENVELOPE_V40)
  + 4 721 (latest-year) + ~1 856 (fixed256/512) ≈ 26 500 steg ≈ 1,35 epoke-ekvivalenter. På
  CURRENTs nåværende kilde: 0 fulle epoker. Siden 19.09 17:08 UTC: 0 steg. `training_enabled=true`
  14.09 22:57 → 15.09 16:41. NEXT_RUN_POLICY har 7 blokker med optimizer_steps = 256 (stikkprøvet).
- De sju prøvene: 256 steg × batch 16 = 4 096 rader hver; alle 256 kontrollrader lå i de første
  4 096 treningsradene. SE for et 256-raders snitt med σ = 20,4 bps er 1,27 bps; å skille 52 %
  fra 50 % treff krever n ≈ 2 400. Prøvene kunne verken påvise eller avkrefte noe.
- Stegtid 2–14 s ved batch 16 → 11 t til 3,2 døgn per epoke; VAL 9–10,5 t per epoke (57,8 M
  M1-tilstandsvisninger, 467 k forwards, 60 % høyresensurert, median hold 170 t) — 4–5×
  treningen den vurderer. Én epoke krevde 11 recipe-versjoner, 11 kampanjemapper, 10 worktrees og
  118 fysiske Windows-reboots 11.–19.09.
- Hypotese→måling: 95-min-horisonten bakt inn 19.07, målt mot tidsskala 26.09 (~1 200 commits
  senere). Den forhåndsregistrerte seleksjonskurve-testen (20.08) er aldri kjørt på en trent
  native modell.

**F05/F21/F60 — skjemabump → full rebuild-sløyfe [S, stikkprøvet].** Signalskjema v2→v37: 13
bump i CURRENTs historikk (`git log -G`), HTF-cache v3→v33, signalmanifest v1→v16; 53 commits med
«rebuild» i emnet; tre rebuild-sykluser på 8 dager med null optimizersteg mellom. Kildeidentitet
bindes på filnivå over hele treet (trener 22 filer, serve-gaten hele gx1/), så hver opprydding
gjør kvitteringer røde. Hvert eksperiment krever kodeendring: `!= 256` 18×, `!= 16` 14×,
`!= 12000` 8×, eksakt dict-likhet mot NEXT_RUN_POLICY.json. [F21/F60: én revisor]

**F11/F15/F18/F54/F12/F13 — treningsdynamikken er fortsatt ødelagt [S+M, verifisert i HEAD].**
Reparasjonsbølgene 6a2be2a7/48d570ad er forfedre via `merge -s ours` («ancestry only», 809ba049)
men koden finnes ikke i HEAD:
- A-1: 0 treff på `target_refresh_interval_optimizer_steps` (17 i 48d570ad); lærer =
  `copy.deepcopy(model)` ved start (:14052), refresh bare ved epokegrense som aldri nås.
- A-2: `EARLY_STOP_MIN_DELTA = 0.0`, `MINIMUM_EPOCHS_BEFORE_STOP = 1`; all-FLAT scorer eksakt 0,0
  på monitoren (sum av tom liste), første epoke alltid «best»; FLAT er absorberende.
- A-3: `_WeightEma.update` (:6214-6226) uten bias-korreksjon, 0 treff på «Initialization-bias»;
  epoke-1-checkpoint = 36,8 % init; den eneste fulle VAL (14.09) evaluerte nettopp det.
- A-4: Kendall-vektene flyttet 0,025 på 256 steg (= lr·steg); optimum s* = ln L ≈ 6–8 → 60–80 k
  steg; effektive vekter 1:1 i råenheter (tail_risk MSE 3 208, dip 1 271 mot Entry-Q 407).
- Portkollaps: `_mk_encoder` (:1069-1082) bygger pre-norm-encodere uten avsluttende LayerNorm
  (hovedencoderen har den, :835-845); gatene over 8 familier / 4 TF / 32 tokens leser dem. Etter
  256 steg 1,000/8 familier (entropi 1,4e-8), 1,0002/4 TF; ~6 M av 9,6 M parametere døde etter én
  time, og hvilken rute som overlever varierer per kjøring. [M på 4 kjøringer 17.–19.09]
- Representasjonskollaps: Q-kontrast-std 0,09 bps mot mål-std 37,5; fuse parvis cosinus 0,999878;
  `nn.init.zeros_` på specialist_out/cross_tf_out/gates (:1181, :1305 m.fl.). FLAT 256/256 er en
  konstant-utgang, ikke seleksjon.

**F07 (nedgradert til MEDIUM av motstanderrunden) — kapasitet er ikke rotårsak, men umålt.**
9 633 055 parametere (29.09-telling): MTF-stack 41 %, lokale spesialister 21 %, Exit-only 28 %,
hovedencoder 7 %, alle hoder 0,07 %. Den observerte feilmoden er kollaps til konstant etter
≤ 1 728 steg, ikke overtilpasning; lav-kapasitetsmålinger (ridge/HGB) på samme informasjon er
allerede null. Det reelle hullet er evidensdesign: ingen læringskurve eller kapasitetsstige
finnes, og sekvenslæring ved lav kapasitet er det ene umålte trinnet. MTF-/spesialistlag er
recipe-eide; bare hovedencoderens dybde/hoder er dataclass-defaults (korrigerer F104).

**F06/F59/F31/F68/F111 — kodevolum uten målt verdi [M; F59 én revisor].** Trener 18 444 linjer
(4 336 → 18 447 på fire måneder, 293 commits, to lifecycle-stier i samme run_train, 65 CLI-flagg).
Exit: 49 511 linjer + 19 536 testlinjer, 180 commits; målt levert: null positiv OOS-økonomi.
Sizing: 8 443 linjer + 10 896 testlinjer, 0 omtaler i gjeldende dokumenter. 45 moduler /
16 516 linjer uoppnåelige fra alle inngangspunkter (AST-importgraf); 14 med null importører.
Infrastruktur ~32 500 linjer mot ~6 600 på den kjørte læringsstien. 1 580 gx1-filer har
eksistert, 261 finnes.

**F23/F65/F107 — styringsvekt [M].** September: 60 % av commits berører docs, 43 % policy-JSON,
26 % bare dokumenter. CURRENT_HANDOVER.md 163 commits siden 11.09, NEXT_RUN_POLICY.json 128
(1 999 linjer, 118 toppnøkler, 53 completed_*-blokker), RUNNING_NATIVE_CALIBRATION.json 128
(2 431 linjer). 111 docs / 13 042 linjer, 96 datert 16.–28.09, 51 foreldreløse; emne-verb
«record» 157, «bind» 126 mot «measure» 13. Reglene ble slettet 14.09 (361 → 20 linjer) og
gjenopprettet 26.09 som 25 regler.

**Hva som må endres.**
- Skill featureflate-versjon fra treningsberettigelse: tren på en frosset flate mens
  kodingsrettelser akkumuleres til én bump per målt epoke; rettelser som berører < 0,1 % av
  radene løses som maske, ikke ny skjemaidentitet (F05).
- Port de tre små rettelsene fra arkivgrenen (A-1 stegbasert refresh, A-3 `decay_t = min(decay,
  t/(t+1))`, A-2 arm patience etter k epoker) og bruk s_i := ln L_i i stedet for gradient på s
  (F11/F15/F18/F54). Gi `_mk_encoder` samme parameterfrie LayerNorm som hovedencoderen; legg
  gate-entropi og radvis Q-kontrastvarians inn som fail-closed læringsvakter (F12/F13).
- Frikoble Entry-trening fra Exit-forwarden: cache lærerens V ved første post-fill-tilstand én
  gang per refresh; Entry-epoken blir da minutter–timer (F19).
- Per-epoke Entry-monitor = netto bps ved faste horisonter på alle VAL-entries fra cachet
  M1-pris; full M1-utrulling bare på valgte checkpoints (F04).
- Én worktree (hook), én launcher, steg/batch/lr/seed som recipe-felt validert mot områder;
  reboot bare ved målt feil (F21/F55/F56/F63).
- Én statusfil ≤ 50 linjer + én append-only eksperimentledger; completed_*-blokkene ut av
  policyfilen; docs før 26.09 til docs/archive (F23/F65).
- Slett de 45 uoppnåelige modulene og frys Exit/sizing inntil Entry-retning er målbar med et
  fast instrument (F59/F31/F68/F71).

## 5. Rotårsak 4 — måling, statistikk og kostmodell

**F02 (motstanderverifisert, tall korrigert)/F10/F25/F36/F44/F79 — portene [M+S].**
- Ukesporten (n = 127 blokker, se 16,38, t ≥ 3,1): MDE 50,8 bps/uke = 26 %/år; power ved 10
  bps/uke 0,7 %. HISTORY2009W (10 årsfolds, se 10,49, t_krit 2,262): MDE 23,7 bps/blokk = 12,3 %/år;
  P(LB > 0 | 10 bps/uke) 13,5 %, P(≥ 8/10 år) 20 %; ingen styrkeavsnitt i preregen. Modellfri
  swing LF-h5-celler: parvis se_Δ 4,4–5,3 → MDE 13,8–16,7 bps/uke (7–9 %/år); tsmom_126_lf_h5
  (+9,1, t 1,87, bjørnefold +24,6) hadde ≈ 10 % sjanse for GO; LS-/h10-/h20-celler 27–136 bps.
  Fire av fem prereger oppgir MDE selv, ingen har en INKONKLUSIV-klasse; tre resultatdokumenter
  og handoveren oversetter NO-GO til «lukket»/«avsluttes»/«må begrunnes med noe nytt».
  Positive-men-underpowered celler har ingen registrert bekreftelsestest.
- Ridge valgte konstanten i 36/36 ukesfits og 9/10 HISTORY-fits; alpha på grid-maks 10⁴ i 33/36
  og 10/10, med indre-val-MSE 4–113 % dårligere enn konstanten. Konstantporten
  (`use_constant = … not best_mse < constant_mse`, walkforward :1020) er stikkprøvet. Det som ble
  evaluert var «FLAT vs alltid-LONG». [F10: én revisor + fit_reports]
- Alltid-LONG-referansen (`apply_cost_policy`, :547-553, stikkprøvet) trekker to utførelser og
  veggklokke-finansiering per blokk; kjøp-og-hold betaler én gang. En FLAT-beslutning krediteres
  ≈ 14–17 bps/uke for mye. HGBs «+1,34 bps fordel» (308/423 FLAT) er mindre enn artefakten.
- Ingen port eller sjekkpunktmonitor er risikojustert (0 filer med sharpe/sortino/drawdown i
  forskningsskriptene og evaluatoren, stikkprøvet); long/flat-trend manifesterer seg som
  vol-/drawdown-reduksjon og er usynlig for Δ snitt-bps.
- Ingen blokk-bootstrap (0 filer med bootstrap/permutation/reality_check, stikkprøvet);
  Bonferroni over speilede celler (18 av 26 scalp-celler er 9 eksakte speilpar; «0/48» er ~0/24
  distinkte utfall); statistikk i rå bps med sd 185–265 per blokk og én grådig offset.
- Retningsferdighet og netto etter markedsordre-kost blandes i ett endepunkt: et
  Bonferroni-signifikant retningssignal ($50-kryss, t 3,80 på mid) ble klassifisert «lukket».

**Merk (fra motstanderrunden, G20):** intradag-nullene er velpowered — HAC-SE 0,3–0,5 bps per
årsfold ved full dekning, MDE 0,6–1,0 bps/side under spread 1,66. Drift-SNR 0,011 måler bare
alltid-LONG-driften og er invariant under horisontskifte (√h). Kritikken av portene gjelder
ukes-/årsblokk-portene, der repoet selv sier retningen ligger.

**F48/F16/F40/F87/F70 — kostmodellen [S+M, konstanter stikkprøvet].**
- Slippage 2,0 bps per utførelse (`unified_exit_prospective_cost_policy_v1.py:178`,
  `parameter_origin: explicit_conservative_nonfitted_preregistration`, `source_status: UNKNOWN`);
  de 258 practice-fillene (20.–29.05.2026) viser halvspread median 0,617 bps. Antatt latenskost
  er 3,2× hele den målte utførelseskostnaden og 68 % av 5,86 bps RT-friksjon, og trekkes inn i
  Exit-lærer og Entry-target. Sensitivitetsscenariene er {1, 2, 4}, ikke {0, 0,5}. Motstanderrunden
  (G05): dette endrer ikke intradag-NO-GO-ene (målt også ved 0 slippage og på mid); det er
  materielt for målets FLAT-anker og for ukesreferansen.
- Finansiering `long_annual_cost_rate 0.054`, `short 0.0` (kreditt 2,82 % klippet,
  `favorable_credit_clipped_to_zero: True`, `historical_rate_series_complete: False`, :198-209):
  et practice-snapshot fra 10.09.2026 anvendt uniformt på 2011–2025 med kontinuerlig veggklokke;
  brokerbevisets `financing_days_of_week` (ons 3) brukes ikke. Alltid-LONG: drift 10,43 bps/uke
  mot finansiering 10,35 → «≈ 0 netto» ved konstruksjon.
- Tre fyllkonvensjoner: Entry på neste M1-open, Exit på samme M1-close (step provider :699-704,
  stikkprøvet), direkte-etikett på neste M1-open. Runtime godtar fill 0–90 s + intraminutt;
  målt latens 09.07: 15–19 min, fyll mot quote ±5–8 bps (n = 4).
- Ekte utførelsesdata finnes (258 fills + 4 journaliserte handler med decision_ts og quote), men
  brokerbevisets sanitizer forbyr trade_id/order_id — joinnøkkelen. Målingen ble erklært umulig.
- Kostautoriteten er runtime-bundet til fremmed worktree `GX1_EXIT_LIFECYCLE_V2` (26 absolutte
  referanser i docs/evidence, 0 til CURRENT) og hardkoder 258 fills/−0,054; ny tape krever nytt
  brokerbevis via API (F84/G18 — motstanderrunden: ingen tall endres, men bindingen er en
  gjenbruksblokkering).

**F17/F45/F41 — VAL/TEST og OOS [M+S].** For 2021–26-kjeden: VAL = juni 2026 (5 509 rader, ESS
≈ 4,5 ukeblokker, 22 D1-barer, 100 % sommertid); detekterbar netto ved t = 3 ≈ 5,8 bps/handel =
hele rundturkosten; TEST jul–aug 2026 ≈ 3,8 bps. v37 utvider VAL til 2025-06..2026-06 (70 880
rader, 32 % vinter), men TEST er fortsatt to måneder. Samme VAL velger checkpoint og avgjør PASS
med any-of over 5 dekninger uten korreksjon; NW-båndbredde 9–10 lag mot 19-bar overlapp.
Serve-paritet krever `split == "test"` (`model_native_serve_gate_v1.py:63`, stikkprøvet) mens TEST
er forseglet; ingen MODEL_NATIVE_SERVE_PARITY_*.json finnes.

**G09/G10/G11/G07 — feedens historie er ikke markedets [M, motstanderverifisert som symptom].**
Alle bundne taper er fxpractice (0 fxtrade; ordet «practice» i 0 .md). Hardt spread-gulv 0,25 USD
2013–2023 (46 % av 2022-barene pinnet); modus 0,39/0,58/0,76 USD 2024–26. Feed-generasjonsskifte
2019-02: tick-volum 433 → 31, range/|ret1| −25 %. 34 % av TRAIN-radene er i vintersesong der
den UTC-faste sesjonsklokka er 60 min ute av fase (kjent siden 13.08, urørt). Symptomer for
edge-spørsmålet, men epokeklokker i inputflaten og avgjørende for hva «spread» betyr 2013–2023.

**Hva som må endres.**
- Hver prereg får MDE ved tre plausible effektstørrelser og en «INKONKLUSIV»-klasse.
  Vol-normaliser blokk-Δ, bruk retnings-IC som primært endepunkt, pool over folds med
  blokk-bootstrap (max-t) i stedet for én årsverdi og Bonferroni (F02/F44/F79).
- Walk-forward-eieren: alpha-grid til ≥ 10⁷ eller relativt til n·tr(Gram)/p; fjern den indre
  konstantporten; rapporter andel fits som ble konstant (F10). Gjenta ukes- og HISTORY-målingene
  før noen NO-GO siteres igjen.
- Referanse = kjøp-og-hold med én utførelse per fold og tidsvarierende finansiering (navngitt
  USD-overnattserie + påslag 1,29 % utledet fra brokerbeviset; SHORT = benchmark − påslag);
  rapporter Δ på mid og på netto separat (F25/F16).
- Risikojustert port: avkastning ved lik realisert vol, blokk-bootstrap Sharpe-differanse, maks
  drawdown (F36).
- Slippage ut av target og inn som evalueringsscenario {0, 0,5, 1, 2}; én fyll-eier (neste
  M1-open) for Entry, Exit og etiketter; latensbudsjett som kontraktkonstant (F48/F87/F40).
- Én lesende produsent som joiner trade_journal med OANDA-transaksjoner på trade_id og
  publiserer en ekte latens-/slippage-populasjon (F70).
- Forhåndsregistrerte kronologiske årsfold 2015–2025 som akseptautoritet også for den native
  modellen; forsegl ≥ 2 hele år inkl. et bjørneår som TEST; paritetssplit som parameter (VAL)
  (F17/F45/F41).

## 6. Rotårsak 5 — reglene og styringen

**F24/G19 — regel 1 stenger gullets drivere [S+I].** Realrenter (DFII10), USD (DXY),
ETF-beholdninger, sentralbankkjøp, COT-posisjonering og fysisk premie er forbudt som inputs. Den
eneste kryss-aktiva-testen på gjeldende kjede (run7a/7c) brukte daglige nivåer med én
kalenderdags lag på 2 t–1 d-horisonter på M5-klokke og ble skrevet som generell refutasjon.
Ukes-testen 26.09 brukte kun XAU-felt. Instrumentet støtter `--decision-clock D1` +
`snapshot_cross`; kombinasjonen er aldri kjørt. Regelen ble skjerpet 22.07 (5adcd1ab) uten
måling. Motstanderrunden (G19): regelen forbyr *inputs i kontrakter*, ikke måling, og har egen
forskningsarm — men armen er ikke brukt på den horisonten der driverne virker.

**F35/G39/G36 — regel 3/4/17, «ingen stop», «aldri fjern» [S].** Regel 3 (19.07) forbyr
regimebetinget FLAT; SNR-diagnosen målte at det ville spart −42,5 bps på VAL og konkluderer at
regelen forbyr det. Regel 4 inverterer bevisbyrden: ablasjonen viser at ingen familie bærer
> ~1 bps, fjerning er forbudt uansett. Regel 17 blokkerer FEATURE_COMPLEXITY_REVIEWs egen
«færre hjelpeoppgaver»-hypotese uten et eksplisitt unntak. «Ingen fast tapsgrense eller maksimal
holdetid» er et operatørutsagn (14.09, RISK_OBJECTIVE `operator_statement`) uten måling, mens
treningsdataene har en implisitt holdetidsgrense (p 119/120, maks 120 steg). Ingen av de 25
reglene har evidensklasse. Motstanderrunden (G06/G39): regel 3 hindrer ikke *måling* (instrumentene
kjørte terskelseleksjon som forskning), og målingene fant ingen post-modell-verdi på gjeldende
substrat; restpunktet er dokumenthull (begrunnelsen forsvant med DECISION_LOG 04.08) og at
regelen brukes på forskningsarmer.

**F34 — målingen ble aldri et vedtak [S].** SYSTEM_MAP: «En ukeshorisont krever en eksplisitt ny
målkontrakt (VEIEN_VIDERE.md)»; VEIEN_VIDERE (også 29.09-versjonen) inneholder ikke ordene
uke/horisont/målkontrakt. Neste steg er «én bundet native forskningskjøring» med «færre
hjelpeoppgaver» som første hypotese — på uendret mål og horisont.

**F33/F72/G04/G17 — veien til en bot og lærdommen fra juli [S+M, G04/G17 motstanderverifisert].**
`ALLOWED_SCOPE_OPERATIONS` har ingen ordreløs skygge-/papiroperasjon. Den ene live-dagen
(09.07.2026, OANDA practice): 116 M5-beslutninger, 48/48 TAKE_SHORT_NOW mens gull steg 4103 →
4132, 4 fills (44 blokkert av same-side-cap), 15–19 min fra beslutningsbar til fill, ≈ −100 bps.
Ingen live-mot-backtest-sammenligning med n finnes i handover 14.07 eller DECISION_LOG 17.07;
premisset «live var fullstendig annerledes enn backtesten» er skjønn på n_eff ≈ 1, og «forrige
kjedes live ≈ backtest» er like ubevist (journalene er slettet). Det reelle funnet var short-bias
synlig offline før launch (paritetsgaten unntok beslutningsbaren) og en featurestakk som ikke
rakk M5-baren. Responsen ble en uoppnåelig TEST-lomme-gate (≤ 10 % feil i 16 lommer mot målt
treff 50–52 %, `serve_gate_v1.py:473-499`, stikkprøvet) og et 90 s-tak som gir 100 % SKIP. Ingen
post mortem finnes i repoet.

**Hva som må endres.**
- Én forhåndsregistrert kjøring med eksisterende instrument: `--decision-clock D1`, 1/2/4 uker,
  arm `snapshot_cross` med ≤ 15 deklarerte, manifestbundne felt (DFII10 Δ, DXY Δ,
  breakeven-inflasjon Δ, GLD-beholdning Δ, COT netto spek, VIX), folds 2011–2025, risikojustert
  mot kjøp-og-hold. Ved positivt resultat endres regel 1 til «navngitte, manifestbundne
  kryss-aktiva-inputs med publikasjonslag er tillatt»; forbudet mot *eksponering* i andre
  instrumenter kan stå (F24).
- Ett avsnitt i GX1_RULES: regel 3, 4 og 17 binder den aksepterte bundlen; en forhåndsregistrert
  forskningsarm kan ablatere familier/hoder, bruke regimebetinget FLAT og sveipe en deklarert
  grid under samme kostmodell (F35). Skriv evidensklasse på hver regel; fjern «aldri fjern
  kapabilitet» fra agentminnet.
- Skill handelsregel fra måleinstrument i GX1_ARBEIDSMAAL: fast horisont er autoritet for
  Entry-retning i forskning (F39).
- Fatt horisontvedtaket eksplisitt før noen native forskningskjøring bindes (F34).
- Legg til én scopet operasjon «offline_shadow_journal» (ingen ordrer) for latens-, fyll- og
  drift-populasjon (F33); skriv docs/LIVE_POSTMORTEM_20260709.md med tallene (F72); rett
  minnepremisset (G04).

## 7. Konkrete feil som må rettes (uavhengig av strategivalg)

| Funn | Feil | Eier | Minste rettelse | Verifisering |
|---|---|---|---|---|
| F11 | A-1: lærer refreshes aldri (fitted-Q iterasjon 0) | `entry_v10_ctx_train_v3.py:14052, 14707` | Stegbasert refresh som i 48d570ad; oppdater PROJECT_DEEP_REVIEW-banneret | kilde stikkprøvet |
| F15 | A-2: all-FLAT = 0,0 absorberende, MIN_DELTA 0, patience fra epoke 1 | `entry_candidate_checkpoint_policy_v1.py:21-26`, `unified_exit_entry_policy_evaluation_v1.py:181-200` | Arm patience etter k epoker; tie-break på VAL rank-IC for monitor = 0,0 | kilde stikkprøvet |
| F18 | A-3: EMA uten bias-korreksjon (36,8 % init etter én epoke) | `entry_v10_ctx_train_v3.py:6214-6226` | `decay_t = min(decay, t/(t+1))` | kilde stikkprøvet |
| F54 | A-4: Kendall-vekter immobile | `_joint_task_loss`, `joint_task_weighting_v1.py` | s_i := ln L_i (detached) eller init fra TRAIN-målvarians | STATE_AUDIT målt, én revisor |
| F12 | Softmax-gater mettes til én rute | `hybrid_transformer.py:1069-1082, 1179, 1319, 1343` | Parameterfri LayerNorm i `_mk_encoder`; gate-entropi ≥ ln 2 som vakt | kilde stikkprøvet, TRAIN_OBSERVATION målt |
| F13 | Konstant-utgang passerer som «FLAT-seleksjon» | `hybrid_transformer.py:3724-3733`, nullinit 1181/1305/1321/1345 | Radvis Q-kontrastvarians som vakt; (nivå, kontrast)-parametrisering | kilde stikkprøvet |
| F20/F74 | Kontrakttekst ≠ utført mål; 512-terminal-flagg; ingen kryss-test | `entry_fitted_q_v1.py:79,134`, `unified_exit_fitted_q_v1.py:31,86` | Bump kontrakten (modus, enhet, ρ-kilde); test at enheten avledes fra Exit-kontrakten | kilde stikkprøvet |
| F73 | Aux-hoder fylles på beslutningsbarens close | `build_entry_v10_ctx_training_dataset_v3.py:559-585` | Samme fyll som primærmålet | kilde stikkprøvet |
| F95 | Hjelpemål betinget på vinnende side | builder `:788-800, 662-672, 840-870` | Sidesymmetriske mål | kilde stikkprøvet |
| F98 | H1/H4 slope3/5 = diff av ATR-normalisert serie | `htf_features.py:3676-3697` | D1-konvensjonen: diff av rå ema-spread / ATR | kilde stikkprøvet |
| F87 | Exit fylles på samme M1-close | `unified_exit_economic_step_provider_v1.py:699-704` | Fyll på neste M1-open | kilde stikkprøvet |
| F40 | Offline fyll T+300 s vs runtime 0–90 s + intraminutt | `entry_causal_m1_outcomes_v1.py:29-37`, `runtime_evidence_v1.py:49,397-423` | Én latensbudsjett-konstant | én revisor |
| F50 | Registry-terskelfit degenerert (velges på støttegulvet, 14–18× mellom vinduer, former ingen felt) | `registry_hyperparameter_fit_v1.py:875-915`, `level_registry_v1.py:37,1226` | Pensjoner terskel-fitten; navngitt levetidskonstant | én revisor, C0-manifest målt |
| F88 | Tre eiere emitterer samme brudd (smc_bos ⊂ level_break, P = 1,0000) | `smc_v1.py:395,501-504`, `swing_structure_v1.py:37,178`, `level_registry_v1.py:1065,1233` | Én pivot-eier, én bruddhendelse per lane | én revisor, målt på cache |
| F81 | Døde bits: `close_range_observed` = 1 på 99,91 %, `open_above/below`, DST-løs sesjonsklokke | `micro_structure_v1.py:31,188`, `entry_candle_primitives_v1.py:303-304`, `session_detector.py:30-42` | Pensjoner de tre; sesjonsgrenser i America/New_York | G07 verifisert; ranking målt |
| F29 | 13 epokeklokker i bps i v37; én median/IQR på 2011–2025 | `entry_model_native_signal_v1.py:273-324`, `input_normalization_v1.py:825-907` | Relativ form; spread-blokken ut av inputflaten | én revisor (kjent klasse) |
| F30/F96 | Upoolet D1-kontekst stopper ved 5 dager; TSMOM/252-d range finnes ikke; Group-A skaleres med M5-ATR | `htf_features.py:3720-3771`, `augment_forward_outcome_v2.py:572-608,678` | TSMOM k∈{21,63,126,252}/ATR, 252-d range-posisjon, atr14/atr252; per-TF-ATR som nevner | D1-felt eksekvert |
| F27 | MTF-laner mean-pooles over 252/96 barer | `hybrid_transformer.py:2139,2163,1286-1297` | Siste lukkede rad som eget token; D1 egen proj-nøkkel | kilde stikkprøvet |
| F10 | Alpha-grid stopper ved 10⁴; indre konstantport | `research_entry_direction_walkforward_v1.py:149,1016-1023` | Grid ≥ 10⁷ eller relativ alpha; fjern konstantporten | kilde stikkprøvet |
| F44/F45 | Ingen blokk-bootstrap; Bonferroni over speilede celler; NW-båndbredde etter handler | forskningsskript, evaluator `:480` | Blokk-bootstrap, max-t; båndbredde ≥ horisont | grep stikkprøvet |
| F84 | Kostautoritet bundet til fremmed worktree | `unified_exit_prospective_cost_policy_v1.py:150-171`, policy.json | Skill quote-binding fra brokervilkår; sti til CURRENT | G18 verifisert |
| F85/F86 | 2010–13: M1-barer inne i daglig pause; 7–15 % M5-bøtter mangler M1 2009–12/2019 | `xau_tape_provenance_v1.py:593-608`, closure authority | Mål høyresensur per år før Exit-økonomi bindes | G27/G29 delvis (målt) |
| F108 | 85 filer med egen sha256-hjelper | `gx1/utils/artifact_primitives_v1.py` | Én mekanisk import-PR | én revisor |
| F62 | Fullsuite rød «som normalt»; `testpaths` peker på `gx1/tests` som ikke finnes | `pytest.ini`, pre-commit | Tiers; null-toleranse for rød | stikkprøvet |

## 8. Prosessendringer (kortest vei fra hypotese til måling)

1. **Stopp-regel for infrastruktur.** Ingen nye schema-versjoner, kvitteringstyper,
   launcher-varianter eller vakter før N fulle epoker er kjørt; nye vakter krever en observert
   hendelse (regel 22). Budsjett «infra-commits per optimizersteg» i handover.
2. **Trapp som obligatorisk rekkefølge** for hver ny mål-/horisonthypotese, bundet i
   `gx1_handover.sh --check`: modellfri → ridge/HGB på eksisterende matriser → frossen
   representasjon + lineær avlesning → først da rebuild/native. 26.–27.09 viste at prereg →
   resultat tar < 1 dag når instrumentene brukes.
3. **Eksperimentledger** (JSONL, én rad per kjøring: hash, n, MDE, resultat, evidensklasse) i
   stedet for completed_*-blokker i policyfilen; én statusfil ≤ 50 linjer.
4. **Forskningssandkasse**: `gx1/research/<dato>_<hypotese>.py` + én ledger-linje; ingen
   kontrakt/test/doc før GO.
5. **Tester**: rød→grønn-bevis i commit for hver feilretting; de ti manglende testene i F32
   (all-FLAT-absorpsjon, EMA-bias, kryss-kontrakt-enhet, aux-fyll, grov-klokke-lekkasje,
   ekte-bytes paritet på én hash-bundet dag, label-sanity, skala-/translasjonsinvarians for
   epokeklokker, instrument-styrke ved realistisk edge, sesjonsspread i kostmodell).
6. **Én worktree**, én launcher, `gx1_capped_run.sh` som eneste inngang for korte kjøringer.
7. **Seeds**: ≥ 3 seeds før noen positiv læringspåstand; seed som recipe-felt (G03/G12/G13 —
   symptomer, men billige).

## 9. Verifiseringsstatus, korreksjoner og tilbakeviste funn

**Hull-runden G01–G48 (fullstendig).** 7 PLAUSIBLE, 39 «fakta bekreftet, symptom for
edge-spørsmålet», 2 tilbakevist på kilde: G33 (kostpolicyen priser per bar bid/ask, ikke døgnsnitt;
sesjonseffekten er ~0,06 bps) og G38 (ingen split-brain i aktiv lineage). Nedgraderinger jeg tar
til følge: G20 (drift-SNR er ikke betinget lærbarhet; intradag-nullene er velpowered); G01/G02
(FLAT-fikspunktet er symptom av rotårsak 1); G03/G12/G13/G45 (én seed svekker bare positive
påstander); G05/G09/G23 (kost/feed/finansiering endrer ikke intradag-NO-GO, men ukesreferansen);
G07/G08/G24/G42 (klokker: reell defekt, ≤ 0,3 bps effekt på håndregler); G14 (paper-evidensen ble
slettet gjennom retention-eieren 29.07; originalen 07.07 før eieren fantes).

**Hovedrunden F01–F129.** Alle CRITICAL/HIGH-grupper (F01–F63, duplikater dekket via sine
kanonikker) fikk full dobbeltdom 29.09: ingen faktapåstand tilbakevist, 37 materielle, 20
symptom (bl.a. F05 skjemabump, F07 kapasitet, F11 A-1, F15 A-2 — rettelsene er reelle, men
motstanderne klassifiserer dem som lås nr. 2–3, ikke primær rotårsak). F01-dommen presiserte at
fullepoken 12.–14.09 gikk på forgjengerkilden, og strøk «kjør N fulle epoker på dagens mål» —
konsistent med F08-dommen: ingen optimizersteg på v37 uten ny mål-/horisontkontrakt. Tre
tidlige dommer med korreksjoner:
- F08 (formulering): faktapåstandene bekreftet; «bevist uløsbar» nedgradert til «målt uoppnåelig
  for alle undersøkte metoder; ikke bevis mot sekvensmodell»; den foreslåtte fiksen
  «D1/W1-klokke» er allerede målt NO-GO tre ganger — innarbeidet i §2 som uavklart, ikke løsning.
  Minste rettelse: ingen optimizersteg på v37 uten ny mål-/horisontkontrakt.
- F02 (porter): kjernen bekreftet; modellfri-swing-MDE korrigert fra 24,3 til 13,8–16,7 bps/uke
  (parvis se_Δ 4,4–5,3, ikke alltid-LONGs se 7,7); power ved +9,1 ≈ 10 %, ikke 2,5 %. Fire av fem
  prereger oppgir MDE. Innarbeidet i §5.
- F07 (kapasitet): nedgradert CRITICAL → MEDIUM; symptom, ikke rotårsak; feilmoden er kollaps til
  konstant, ikke overtilpasning; MTF-/spesialistlag er recipe-eide. Innarbeidet i §4.
F64–F129 (MEDIUM/LOW) står fortsatt som én revisors funn med fil:linje; 25 kjernepåstander er
stikkprøvet mot HEAD (§0) og stemmer. Restverifisering kan kjøres når som helst fra
verdicts_partial.json (bare manglende grupper).

**Korrigert fra utkast:** «anbefalingen 26.–27.09 står ureffektuert» er feil — måling A ble kjørt
(NO-GO), pkt 3–4 var betinget av GO. F104 («kapasitet er ikke recipe-verdi») gjelder bare
hovedencoderens dybde/hoder.

## 10. Det som er solid (og ikke skal røres)

- Kausalitet: MTF-join bevist eksakt (lukket-bar-regel uavhengig av origin, null avvik på
  313 399 rader); pivot-refleksjon først fra j+lookback i alle registre; fits bruker
  framtidsutfall bare innen deklarert kalibreringsvindu; ingen forward-outcome-lekkasje inn i
  features.
- Økonomi: side-korrekt utførbar bid/ask fra hash-bundet M1-tape for entry og exit;
  kostkomponenter lagres separat per steg; geometri- og glitch-vakter; helgemaske v2 er
  DST-riktig; TRAIN/VAL/TEST overlapper ikke fysisk.
- Lineage: fitted-Q-byggerne er stop-gradient og fail-closed; hindsight-feilen max(HOLD, 0) er
  rettet (17.09); Y_wait korrekt parkert; normalisering TRAIN-only; input-normaliseringstilstand
  immutabel og verifisert før allokering.
- Instrumenter: walk-forward-eieren har D1-klokke, cross-arm, kardinalitetsmatchet ablasjon,
  HAC-SE, sirkulær-shift og beste-konstant-null; fold-grensefeilen fra 27.09 er rettet med
  regresjonstest; forhåndsregistrering praktiseres konsekvent og avvik rapporteres ærlig.
- Prosess: retention-eieren er begrunnet i en reell hendelse; handover-sjekken er lesende;
  dokumentene skiller inputbevis fra læring/edge og lover ikke mer enn målt.

## 11. Ikke undersøkt

- 2011–2018-fordelingene for alle felt (epokeeffekter er nedre grenser målt på 2019–26).
- Faktisk edge/attribusjon per feature under trening (ingen ablasjon eller læringskurve finnes).
- Om et nettomål eller (nivå, kontrast)-parametrisering endrer representasjonen ved full trening.
- OANDA fxtrade-feed vs fxpractice (ingen live-prøve finnes).
- Passive/limit-fyllinger, kostgrid under 2 bps, horisonter > 4 uker, porteføljeaggregat,
  historisk finansieringsserie, VAL-bekreftelse av de fem fortsettelsescellene på mid.
- Sekvensmodellen på ukeshorisont (bare ridge/HGB på øyeblikksbilder er målt).
- Resterende MEDIUM/LOW-grupper uten full dobbeltdom (§9).

## Publiseringspresisering 29.09.2026

Originalrapporten er bevart uendret under
/home/andre2/GX1_RUNS/EDGE_ROOT_CAUSE_REVIEW_20260929/EDGE_ROOT_CAUSE_REVIEW_20260929.md.
Publiseringskontrollen og kildehashene ligger i
/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/DOCUMENTATION_VERIFICATION.json.
Kontrollen korrigerer dommetelling og spreadpåstanden ovenfor; den er ikke en ny
fullstendig gjennomgang av alle 177 funngrupper eller en ny markedskjøring.

Rapporten beskriver regelverket før forskningsvedtaket 29.09. Gjeldende regel 1
står i GX1_RULES.md, og godkjent videre arbeid står i
[forskningsplanen](TA_RESEARCH_PLAN_20260929.md). Lav marginal korrelasjon beviser
ikke fravær av betinget lærbarhet. Historiske NO-GO-resultater gjelder de faktisk
undersøkte oppsettene; instrumentrettelser skal måles før videre konklusjoner.

## 12. Anbefalt rekkefølge (billigst først)

1. **Vedtak (operatør):** ingen optimizersteg på v37 før mål-/horisontkontrakt er
   forhåndsregistrert; Exit som måleinstrument i forskning; forskningsarm-unntak fra regel
   3/4/17; kryss-aktiva som navngitte, manifestbundne forskningsinputs; ikke bind «færre
   hjelpeoppgaver»-kjøringen på dagens mål.
2. **Reparer instrumentene (1–2 dager, CPU):** alpha-grid/konstantport (F10), blokk-bootstrap og
   max-t (F44), vol-normalisert Δ (F79), kjøp-og-hold-referanse med tidsvarierende finansiering
   og scenario B som ko-primær (F25/F16), risikojustert endepunkt (F36), MDE og
   INKONKLUSIV-klasse i prereg (F02). Gjenta uke- og HISTORY-målingene på 2011–2025.
3. **Måling A (dager):** ≤ 15 skalafrie D1-felt (TSMOM 21/63/126/252 i ATR-enheter, 252-d
   range-posisjon, ATR14/ATR252, EMA200-avstand) på D1-klokke, 2011–2025 årsfolds, ridge/HGB med
   og uten vol-regime-interaksjon, mot kjøp-og-hold (F100). **Måling B:** samme med
   `snapshot_cross` ≤ 15 makrofelt (F24). **Måling C:** VAL-bekreftelse av composite av de fem
   fortsettelsescellene på mid, med kostgrid (F43). Utfallet avgjør om ukes-/makrospor har noe
   å lære — spørsmålet er i dag ubesvart, ikke lukket.
4. **Bare ved GO på A/B:** ny målkontrakt (§3), closure-bevisst utfallsklokke (F52), de manglende
   D1-feltene (F30), siste-bar-token i MTF (F27), pensjonering av epokeklokker og døde bits
   (F29/F81), én bruddhendelse per lane (F88). Én rebuild, deretter frys.
5. **Parallelt, uavhengig av strategi:** A-1..A-4-portering, gate-LayerNorm og kontrastvakt
   (F11–F18, F12, F13, F54); kontrakttekst og kryss-test (F20/F74); fyllkonvensjon og
   latensbudsjett (F87/F40); slett 45 uoppnåelige moduler og frys Exit/sizing (F59/F31/F68);
   ledger + statusfil (F23); tests-tiers (F62).
6. **Første native jobb etterpå:** ikke rebuild, men walk-forward + seleksjonskurve på
   epoch-1-sjekkpunktet fra 14.09 (VAL-prediksjonene finnes) — første OOS-avlesning på en trent
   modell (F26/F57). Deretter kapasitetsstige (1-lags GRU/attention → full) på samme cachede
   mål (F07), med Entry-epoke på minutter (F19).
7. **Skyggejournal uten ordrer** (F33) når en kandidat finnes, for latens- og fyllpopulasjon.
