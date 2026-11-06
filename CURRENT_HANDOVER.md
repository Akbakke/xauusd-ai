# CURRENT HANDOVER — GX1 M5 Entry

Eneste arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt ved første overtakelse og kjør
`bash scripts/gx1_handover.sh --check`. `NEXT_RUN_POLICY.json` er eneste
arbeidsstatus. Terminalkvitteringer overstyrer denne teksten. Tidligere
statusnotater er bevart i Git og originale kvitteringer.

## Gjort og målt

- Begge armers pre-oktober TRAIN-fit er avsluttet på 29 849 foreldre og
  1 866 optimizersteg hver. De egne oktober-scorekjøringene er konsumert:
  6 323 senere foreldre, 396 forwards per arm, null nye optimizersteg og
  signert vakt PASS. Oppstartsoppgaven er deaktivert. Ingen TEST ble brukt.
- [Kontrollens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_SCORE_003/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `d37c387736b8578fc3ff228aff7eba09d75b8380dcd102624061c6f4d4e9d932`;
  [kandidatens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_SCORE_002/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `02043546ea5da35437843164724d9107cdc0a994042e99b8b07a332b0e71c47d`.
- [Paret oktoberutviklingsrapport](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PAIRED_DESCRIPTIVE_20261106_001/REPORT.json)
  SHA-256 `82988bb594c31c5d77ac1108b79befb6497dbaac65277b3e6babc8770f17e258`.
  M5 +5 har 3 359 observerte av 6 323 foreldre. Brier for kandidat /
  kontroll / fryst TRAIN-prior / Markov er 0,614672 / 0,615091 / 0,612714 /
  0,615325. Kandidat-minus-kontroll har radvektet 95 % UTC-dag-intervall
  [−0,000861, +0,000032]. OPP/NED-AUC er 0,4898 for kandidat og 0,4897
  for kontroll. M5 fullgrid Brier er 0,686345 mot kontroll 0,685498 og
  prior 0,624951. De andre fire tidsrammene er sekundære i rapporten.
- Av 2 964 ukjente M5 +5-fasiter har 2 950 årsakskode 24:
  intraminutt-rekkefølge ukjent og retning uavklart under M1-OHLC. Ukjent
  er ikke FLAT. Dette er målkontraktens konservative kildegrense, ingen
  påvist kodefeil.

## Hindring og neste minste steg

Aldersinput kvalifiserte ikke M5-retningsverdi. Oktober er utviklingsdata,
og den opprinnelige planen manglet en konkret beslutningsregel bundet før
scoring. Rapporten er derfor deskriptiv og gir ingen retrospektiv PASS.
Ingen uavhengig retningsedge eller økonomisk edge er påvist.

[Entry-path TRAIN-kvitteringen](/home/andre2/GX1_RUNS/NATIVE_M5_ACTUAL_NATIVE_BINDINGS_20261019_057/PHYSICAL_TARGET_RECEIPT.json) SHA-256
`dd95d9ecb43f309635e61ba268ec4db6518a3ee23550aa0c94330ff303a5063d` og [oktoberaksen](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/OCTOBER_AXIS.json) SHA-256
`2aa5bad2804d6635211ceac21eb9935639198f147d7d1a821188d3a07d6c17c5` har samme 118 249-raders M5-barstartklokkehash
`e5d528ff8fcec2b23f0099834b1c5b42a12489dadfada0a6fc4aeab3b6ff4c5d`. Oktober er radene [29 849, 36 172), 6 323
foreldre. Beslutningstid er barstart pluss lukket M5-bar i den eksisterende
eieren. Dette kvalifiserer rad-/klokkematch, ikke observerbarheten av
framtidige executable endepunkter eller MFE/MAE-stier; tette framtidsbaner
er ikke lagret. [Dekningsrapporten](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_MASK_COVERAGE_20261106_001/REPORT.json) SHA-256
`17943a62ef36020120878c3e72a124c99304fe2f60ea7d7c638926b5175a0b37` er fullført og engangsplanen konsumert. Den målte alle
tette 1..480 minutters horisonter innen oktobercutoff med 0 modellkall,
0 oppdateringer, 121,6 sekunder og maksimalt 2,94 GiB RSS under 4 GiB.
Ved +5 var 6 322 rader foldmodne (én høyresensur); 6 297 hadde executable
terminal-open og observerbar terminal-open bane på begge sider, og 6 322
hadde observerbar closed-prefix-bane. Dette er andre mål enn tilstandsfasiten
på 3 359 observerte rader; bedre dekning er ikke lært retningsedge.
Ved +480 hadde 5 607 av 6 240 foldmodne executable terminal og 4 079
observerbar terminal-open bane. Den eksisterende kildeleseren åpnet hele
pre-2026 TRAIN-kilden ved oppsett, men rapporterte bare spørringer før
oktobercutoff; ingen TEST. Den forhåndsbundne, modellfrie M5 +5-joinen er nå fullført én gang.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/CLAIM.json) SHA-256
`03dbcfb1e5836502ed02bc129cbc2b4cc6e98251eb38369642090fca3bf12c34` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/REPORT.json) SHA-256
`078f9ea44a096a64d5001799773b9ae8d6e508756c332868fa74dce236101ad0`.
Av 6 323 foreldre var 6 322 innen cutoff og 6 297 hadde executable
terminalutfall på begge sider. Ingen eksakte LONG/SHORT-ties. Kandidatens
radvektede AUC var 0,490519 mot kontrollens 0,490249. Paret forskjell
hadde 95 % UTC-dag-bootstrapintervall [−0,001306, +0,001950]. Ved lik
vekt per UTC-utfallsdag var AUC 0,477911 mot 0,477474. Kandidat mot
0,5 hadde lik-dag-intervall [−0,043802, −0,004016]; radvektet
mot 0,5 omfattet null. 27 utfallsdager. 0 modellkall, 0 oppdateringer,
0 TEST, 0 seleksjon; 116,0 sekunder, 2,72 GiB maksimal RSS under 4 GiB.
Erklærte historiske kostnader er ikke sannhetskvalifiserte. Dette er bare
utviklingsdiagnostikk og gir ingen Entry-gate, uavhengig eller økonomisk edge.
Planen er konsumert og kan ikke kjøres på nytt.

Stans +5-tilstandsscoren som surrogat for Entry-sidegate. Det løpende
OPP/NED/FLATT-prognosemålet på fem TF består som egen fase. Kontraktene viser
at tilstand ved beslutning+5 gjelder lokal M5-retning fra lukkede M1-midtbars,
mens Entry-sideutfallet er netto executable BID/ASK-endring fra beslutningen
til terminal-open. De er ikke samme target. En lesende gjennomgang fant at
direkte native 191/192 valgte FLAT på alle 256 TRAIN + 256 gjenbrukte CONTROL;
177/178 gjorde også bare FLAT, mens 151/152 valgte 39 LONG og 71 SHORT av
512 uten kvalifisert valgt kvalitet. Ingen bred kronologisk direkte Entry-OOF
er lagret i aktivt scope. Gamle rene retningsreferanser 247/249/252/259/215
er konsumert eller avvist; ikke gjenta dem.

Den fysiske Entry-kvitteringen dekker 118 249 TRAIN- og 17 284 VAL-foreldre,
men `dense_future_paths_materialized=false`; den lagrer M1-radpekere og
første masker, ikke en bred LONG/SHORT/FLAT-fasit. Eksisterende
`DynamicM5TargetProvider.targets_for_rows` beregner terminal og path
on-demand. Oktoberrapporten måler bare dekning; 512-raders native panel er
ikke bred kronologisk Entry-OOF. En bred modellfri målfordeling ville bare
telle kjente framtidsmuligheter, ikke kausal forutsigbarhet. Den endrer ikke
Entry-beslutningen og er ikke neste kjøring; en slik diagnose får bare egen
plan ved et nytt, konkret spørsmål som krever den. Skjult maksimum av
framtidsfasit over horisonter er forbudt som Entry-Q.

[Kostautoriteten](/home/andre2/GX1_RUNS/NATIVE_M5_CURRENT_COST_TERMS_20261019_059/COST_POLICY/parameter_authority.json)
SHA-256 `e93ed40451a24344f94716e1f5ebe4cd348e6c7c3120e7509d6d55827bf40522`
er en komplett prospektiv parameterpolicy, men `historical_cost_truth_qualified=false`.
[Brokerevidensen](/home/andre2/GX1_RUNS/NATIVE_M5_CURRENT_COST_TERMS_20261019_059/BROKER_EVIDENCE.json)
SHA-256 `d0dd8b19a4aa8aed82aaa3c20cadc0dae211d14f6edcc797db0e0c2084ee951c`
har 258 fills, men mangler full historisk finansieringsrenteserie,
beslutningsquote-til-fyll-klokke og eksplisitt historisk prisplan. M1
BID/ASK-candles ved barstart kan ikke gjenopprette disse faktaene; 2 bps
slippage per utførelse er en prospektiv antakelse med 1/2/4-sensitivitet.
Rapporter deklarert kost og sensitiviteter, aldri historisk kostsannhet
eller realisert økonomi fra disse bytene.

Mål-mot-mål-kontrollen er fullført én gang etter [engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001.json) SHA-256
`90915641b504d18603fd76562a97461fa2872c399283f36014bb0fd1f84e3d39`.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001/CLAIM.json) SHA-256
`b6ee022fc6d75860abe89f862cdf6e30f3a4c760510888a5e39b17fafd20fee0` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001/REPORT.json) SHA-256
`3377e7d092abcccc90705e064031d6c545ec63ca7329978e4b43682c0566f090`.
Felles observerte oktober M5+5-rader: 3 344. DOWN hadde 262 LONG-bedre og
1 067 SHORT-bedre; UP 1 212 LONG-bedre og 317 SHORT-bedre; FLAT 263/223.
P(LONG-bedre | UP) = 0,7927 mot 0,1971 ved DOWN; forskjell 0,5955 med
beskrivende 95 % UTC-dag-intervall [0,5662, 0,6252] over 27 dager.
Ytterligere 2 953 executable rader hadde ukjent tilstand; 15 hadde kjent
tilstand, men ukjent Entry-side; 11 hadde begge ukjent. Tilstandsmålet
samsvarer dermed sterkt med sideutfall på den observerte delmengden,
men fullpopulasjons-samsvar og forutsigbarhet fra kausale inputs er ubevist.
Samtidig rangerte de lagrede modellprognosene M5+5 nær tilfeldig på oktober.
0 modellkall, 0 optimizersteg, 0 TEST, 0 action-seleksjon; 110,5 sekunder,
2,75 GiB maksimal RSS under 4 GiB. Kostsannhet er fortsatt ukvalifisert.
Ingen Entry-gate eller økonomisk edge følger av denne utviklingsrapporten.
Planen er konsumert og kan ikke kjøres igjen.

Lagret sluttmåling på 16 798 observerte TRAIN M5+5-rader har kandidat-Brier
0,604215 mot fryst prior 0,609071. Oktober har 0,614672 mot prior 0,612714.
På hele M5-horisontgridet var kandidaten verre enn prior også i TRAIN:
0,680811 mot 0,618665. Det native +5-endepunktet har eksplisitt tapsvekt
1,0 og faktisk gradient; ren 5×1440-tapsfortynning er ingen påvist forklaring.
Kandidat-only TRAIN-rangmålingen er fullført én gang etter [engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001.json) SHA-256
`1da8201cc59a5e69308b0b6c8c3bef6a311a9f52711d779df2254f050b29e553`.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001/CLAIM.json) SHA-256
`f75d5382c858b4a693e971ef9af10f1b652de487e73b665c6a4f8be8d676a94b` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001/REPORT.json) SHA-256
`a1518c6ae33705ce1229ee9eb77e87c9079f227bb0b48f3dd089fe2f5462ec3b`.
På 16 798 observerte TRAIN M5+5-rader var 14 485 OPP/NED. Kandidatens
in-sample OPP/NED-AUC var 0,521784; beskrivende 95 % UTC-dag-intervall
for AUC minus 0,5 var [+0,012716, +0,030792] over 131 utfallsdager.
Lagret Brier ble gjenskapt eksakt innen avrunding (0,6042147567).
Senere oktoberutvikling hadde AUC 0,489817 på 3 359 observerte rader.
Dette viser svak in-sample rangering uten målt overføring til oktober;
periodene er ikke en paret eller uavhengig test. Den lagrede 2,58 GB-filen
ble SHA-verifisert, M5+5 ble minnekartlagt, og kjøringen brukte 0 modellkall,
0 optimizersteg, 0 TEST, 7,35 sekunder og 3,04 GiB maksimal RSS under 4 GiB.
Ingen Entry-handling, uavhengig edge eller økonomisk edge er påvist.
Planen er konsumert og kan ikke kjøres igjen.

Lesende feature-/maskerevisjon fant ingen konkret input- eller
normaliseringsfeil. Kandidatens fit- og scoreoppskrifter binder samme
feature-view, prefix-normalisering, kausale input og målskilder; faktiske
kjøringer gjorde 1 866 optimizersteg og 396 score-forwards. Den fire-raders
[batchkvitteringen](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_BATCH_BINDING_20261106_002/RESULT.json) SHA-256
`57608c0cab08605fd6ac4095090f80ee56ac90037e89887503210a1230a80aa0`
viser de 17 forventede inputnøklene. Koden ruter fem TF og åtte spesialister,
med 61 D1/makrofelt til kontekst. Teknisk rute og liten batchparitet beviser
ikke at alle felt er informative eller at modellen lærte bidragene deres.
M5+5-observasjonsdekning var 56,28 % i TRAIN og 53,12 % i oktober;
DOWN/FLAT/UP-fordelingen var henholdsvis 41,29/13,77/44,94 % og
39,71/14,59/45,70 %. Kandidatens Brier forverret seg 0,01046 mot priorens
0,00364. Mask-/klasseforskyvning alene forklarer ikke målt fall, og ingen
kausal feilårsak er påvist.
Lagret [kandidatfit](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_FIT_002/RUNTIME/prefix-fit/window-0001/RESULT.json) SHA-256
`a42712db365c1bb4f53669c4564c09db7cf3e56c9feb2f6ff6257d43f53f11a5` viser
1 866 fysiske AdamW-steg. Siste batch hadde ikke-null gradient i alle 135
rapporterte grupper, inkludert åtte spesialistprojeksjoner, MTF/GC,
`entry_q_joint` og tilstandshodene. Første batch hadde null i de åtte
spesialistprojeksjonene; det vedvarte ikke. Kvitteringen gjelder samlet tap
og måler ikke isolert endpoint-gradient eller vektendring per parameter.
Ingen vedvarende gradientbrudd er påvist; dette innrømmer ingen kodeendring
eller ny fit.

Ny feilhypotese fra brukerens EMA/ATR/nivåinnvending: oktoberkandidaten
predikerer M1-midbasert markedstilstand, ikke executable Entry-Q. I den
aktive dynamiske M5-Entry-fasen er `entry_action_q_loss` inaktivt;
Entry-kvalitet er predikert terminal-netto minus predikert tidlig MAE fra
to separat trente komponenter. Dette er en mål-/beslutningsmismatch,
ikke en bevist featurekodefeil. Den eldre samme-rad-kontrollen i
`docs/DYNAMIC_MTF_STATE_NO_EDGE_DIAGNOSIS_20261103.md` fant motsatt
M5-candlefortegn AUC 0,5185 mot modell 0,4979 på 7 812 senere
utviklingsrader, men ingen Entry- eller økonomisk edge. EMA20/ATR14 og rå
støtte-/motstandsavstand har varierende TRAIN-normaliseringsstatistikk på
alle fem TF; faktisk rå tensorparitet på beslutningsrader er ikke bevist.

Én [rå inputaudit](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001.json) SHA-256
`b5a6785ab4666d03663ce4ad3878f6c84a839821d4c3f2598cc6d2f2e01ef0d9`
er innrømmet som lokalt 0 USD engangskall: alle 36 172 as-of-valg på fem TF
og åtte faste rader med rå EMA/ATR/nivå cache→gather-paritet. Ingen
modellforward, trening, TEST eller broker. Resultatet kvalifiserer ikke
post-normalisering, per-felt modellbidrag eller nivåobservasjon. Først etter
denne kontrollen vurderes hypotesen om direkte supervisjon av dagens
`terminal-netto − MAE`-kvalitet på felles observasjonsmaske; ingen fit er
innrømmet. FLAT på 191s 256 CONTROL-rader kan være riktig, siden observert
sidekvalitet der var negativ i snitt.

En siste metadataavklaring fant ingen lagret, foreldrejustert 2D-matrise
med hele den native inputflaten i gjeldende bundne kandidat-artefakter. [CAUSAL_INPUTS](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_FEATURE_DELIVERY_20261106_001/CAUSAL_INPUTS/CAUSAL_INPUTS.json) SHA-256
`2fdea20ebeb90ec8a027f067726f1dd488139e4a1662ee61a75920b1ab764ae6`
har bare justert nåtilstand/alder/masker. [M5-cachemanifestet](/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003/MULTI_TF_V4_CACHE/manifest.json) SHA-256
`3f4c3b4b54b08931850b716e0b7cf4b5de0b54c42b0fee813cde6f1324371da4`
har en kausal M5-matrise på 1 140 982 × 253 på baraksen, men den må først
kobles as-of til beslutningene og mangler native signal-/kontekst-/GC-flaten.
Eksisterende eier `gx1/features/htf_features.py:slice_multi_tf_v4_window`
velger siste lagrede M5-barstart ≤ `issue−5 min`, men likhet med hver av de 36 172
foreldrebarene er ennå ikke fysisk kontrollert. Denne M5-only-baselinen er
ikke innrømmet: den kan ikke avgjøre full native input- eller Entry-verdi.

Ingen ny native eller enkel modellfit er innrømmet. Neste eksperiment krever
én særskilt hypotese med forhåndsbundet kronologi, målmaske, referanse og
akseptregel; oktober er undersøkt utvikling, TEST forseglet. Inntil da står
modellen uten påvist senere retnings- eller Entry-beslutningsverdi.
Fem-TF-prognosemålet og Entry-kravene består.

Eksisterende data er nok til å undersøke læring videre: fysisk native XAU/GC
TRAIN har 118 249 foreldre fra 2024-05-01 til 2026-01-01, og VAL har
17 284 fra 2026-01-01 til 2026-04-01. GC MBP-1, genuine indikatorer, fem TF,
åtte spesialister og D1/makro er ikke en manglende datakilde. Ny historisk
kronologisk TRAIN-OOF kan planlegges med foldspesifikk fit på tidligere,
målmodne rader uten nye data; dette åpner ikke dagens `training_enabled=false`
eller en ny kjøring uten særskilt hypotese og eksakt policyadmisjon. Alle
26 månedene er allerede undersøkt utvikling, så intern OOF er ikke en urørt
sluttbekreftelse. Nye uavhengige data er ikke en forutsetning for å undersøke
læring nå; de trengs først for en endelig uavhengig edgepåstand.

En fryst lineær avlesning av kandidatens siste hidden er ikke en gyldig
kronologisk OOF-test av representasjonen: backbonen ble fit på alle 29 849
pre-oktober TRAIN-foreldre. Lagrede [TRAIN-arrays](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_FIT_002/RUNTIME/prefix-fit/window-0002/FINAL_TRAIN_PREDICTIONS/ARRAYS.json)
SHA-256 `77abe66d40db906fdb8266edfdb532b2dbffa161db83994ef644107f1f973b38`
og [oktober-arrays](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_SCORE_002/RUNTIME/prefix-fit/window-0001/OOF_PREDICTIONS/ARRAYS.json)
SHA-256 `d8bd67c9c471f4ef93978ca8ab8367f50197d9a82694c1d0eee5dd9b6f557bcc`
inneholder bare rad/tid, sannsynligheter og sekvenstap, ikke hidden. Et nytt
uttrekk ville kreve ca. 2 261 full-native batcher over 36 172 foreldre.
Direkte +5-join måler bare relativ LONG/SHORT terminal-markout, ikke FLAT,
MAE før første MFE, valgt kvalitet eller økonomi. Den lineære proben er
høyst komponentdiagnostikk og er ikke innrømmet som Entry-test.

Zero-refit-skåring av en senere utviklingsmåned er fysisk mulig med bevart
checkpoint, men dagens scorer og mål/akser/tape er bundet eksakt til oktober.
En ny måned krever ny avgrenset plan og kvalifisert full-native input; to
oktoberlignende scorearmer tok samlet omtrent 35 minutter og 1,1 GB output.
Dette ville fortsatt måle tilstand, ikke full Entry-beslutning, og alle
26 historiske måneder er utvikling. Forward-journalen i `GX1_RULES.md`
gjelder D1-swing, ikke M5. Ingen slik kjøring er innrømmet. Neste kvalifiserende
hypotese må gjelde direkte LONG/SHORT/FLAT Entry med forhåndsbundet
kronologi, dekning, path/kost/økonomi og en genuint uavhengig fremtidig
bekreftelsesrute; fem-TF-prognosemålet bevares separat.

M5 Entry er hovedmål med kausal lukket M15/H1/H4/D1-kontekst, åtte
spesialister og niende D1/makro. `training_enabled=false`. TEST er forseglet;
bare forward-journalen scorer sin periode på forhåndsbundet måte. Exit,
broker, ordre, live/paper, spending og promotion er stengt. Ingen konsumert
engangsplan relanseres. Stabil langkjøring kontrolleres ca. hver 15.–30.
minutt, gjerne sjeldnere ved stabil drift.

## Bevarte eldre artefaktankere

Disse stistrengene sto bare i tidligere overlevering og beholdes for
retention-eierens navngitte sti-vern. De gir ingen arbeids- eller kjørestatus.

/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json.
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19.
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_PREP_20261105_003/TEST_TERMINAL.json
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_PRIVATE_001/CAMPAIGN/CAMPAIGN_PLAN.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_INPUT_LINEAGE_PARITY_REVIEW.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_LOSS_COMPOSITION_REVIEW.json
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_BATCH_GEOMETRY_ROOT_REVIEW_20261028_002.json.
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_NATIVE_B16_SUPPORTED_QUALIFICATION_20261028_003/ACTUAL_ROOT_REVIEW_001.json.
/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_003/CAPTURE/INPUT_TIMING.json
