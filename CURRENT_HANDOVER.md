# CURRENT HANDOVER — GX1 markedstilstand før Entry

Eneste arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt ved første overtakelse og kjør
`bash scripts/gx1_handover.sh --check`. `NEXT_RUN_POLICY.json` er eneste
arbeidsstatus. Terminalkvitteringer overstyrer denne teksten. Tidligere
statusnotater er bevart i Git og originale kvitteringer.

## Aktiv fase etter brukerpresisering 06.11

Definer og lær dynamisk OPP/NED/FLATT-markedstilstand på M5, M15, H1, H4 og
D1 først. M5 er prognosetakt; Entry-plassering, direkte Entry-Q, Exit og
handelsbeslutninger kommer senere. Ingen ny fit er innrømmet.

Én [målsensitivitetsplan](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001.json)
SHA-256 `9aa37c8531de784968c9821960b43cd40ee7722a018827c00561ab1319543e2a`
er fullført og konsumert. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001/RESULT.json) SHA-256
`b31f022ec3398a1a6400e202f513407830229d97e0e86d73063bf70bc1655ab2`; [TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001/TERMINAL.json) SHA-256
`db33cf6136e1d020027f4bf4cfad9022f63d5eb6d613531148e900add87326ae`. Den gjenskapte alle 13 lagrede
0,5-arrays eksakt og sammenlignet 1,0 på 29 849 mai–september-foreldre
uten modellkall, TEST eller publisering av 1,0-serie. Exit 0, 10,3 s og
481,1 MB prosessmaks under 4 GiB-vakten; ingen ny kjøring er innrømmet.

| TF / native lead | 0,5 observert | 1,0 observert | FLATT av observerte, 0,5 → 1,0 |
| --- | ---: | ---: | ---: |
| M5 / +5 | 16 798 / 29 849 | 27 936 / 29 849 | 13,8 % → 63,9 % |
| M15 / +15 | 25 679 / 29 849 | 29 062 / 29 849 | 13,9 % → 61,2 % |
| H1 / +60 | 27 770 / 29 849 | 28 278 / 29 849 | 13,4 % → 60,5 % |
| H4 / +240 | 26 829 / 29 849 | 26 632 / 29 849 | 8,4 % → 54,7 % |
| D1 / +1440 | 18 799 / 29 849 | 16 640 / 29 849 | 3,4 % → 52,7 % |

Større grense løser mye av M5-observasjonsmangelen, men gjør FLATT til
majoriteten av de observerte etikettene og reduserer D1-dekningen. Dette
er en mål- og observerbarhetsavveining, ingen modellkvalifisering eller
automatisk beslutning om ny terskel. Gjeldende 0,5-kandidat er fortsatt
semantisk ulåst. Aktiv neste oppgave er å binde observert aktivitet versus
ukjent retning, framtidige gyldighetstidspunkter, output, tap og måling
uten å gjøre ukjent til FLATT. Deretter følger indikatorenes normaliserte
konsum og den åpne numeriske kvalifiseringen.

Kildegjennomgang viser en konkret delvis observert fasit som dagens tap ikke
bruker direkte ved senere native ledespørsmål: full lokal range kan bevise
ikke-FLATT selv når OPP/NED er ukjent. [Lagret dekningskontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_LABEL_COVERAGE_NATIVE_LEADS.json)
SHA-256 `963115cac4f3b7859a47f8155985c4305d361c9763a867b3db49b9575a862331`
fant 21 252 slike M5+5-rader i den eldre 47 602-raders TRAIN-populasjonen og
8 305 i 17 288 senere gjenbrukte utviklingsrader. Det aktive binære
tillleggstapet gjelder bare +1; native +5-tapet krever kjent treklassefasit.
Sekvensloss kan fortsatt gi indirekte gradient via nabotilstander.
Eksisterende målspørring kan hente binær aktivitet ved vilkårlig native lead,
og desember-scoreren gjør det allerede på alle fem TF. Ingen ny fasitmotor
eller scorer trengs nå. [Lagret binær skår](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_SCORE_NATIVE_LEADS.json)
SHA-256 `c8bb833a3f0777f362108252a9b2c9dbcbb98aaa7cde8fea3df405ec2a451bb4`
viser at den eldre ettpassmodellen ved M5+5 hadde binær Brier 0,063594
mot Markov 0,071860 på gjenbrukt utvikling, mens D1 var verre.
Dermed er et ekstra binært tap en kontrollert hypotese, ikke en klar første
rettelse for OPP/NED. Bind eventuell delvis-observert native loss, maske,
enhet og vekt før fersk baseline; ikke tren den automatisk.

Målkontrakten er ennå ikke låst som meningsfull markedsdefinisjon. Faktoren
0,5 × ATR(14) per TF er en brukerakseptert kandidat, ikke endelig kalibrert.
02.11-gjennomgangen bestod teknisk regelkontroll på 61 TRAIN-spor og merket
bare dette avgrensede reviewet `target_rule_locked=true`; gjeldende
04.11-målvurdering og 05.11-serie har `false`. Ingen korrupt artefakt eller
mål-kodefeil er påvist av spriket. Før ny fit må bevegelse/FLATT, de
forhåndsvalgte framtidige gyldighetstidene, ukjent-maske, output, tap og
måling bindes sammen på hver TF. M1-OHLC gir observerbar M5-tilstand på
126 395 av 238 142 unike TRAIN-tidspunkter (53,08 %); oktober M5+5 har
3 359 av 6 323 (53,12 %). Ukjent er aldri FLATT. Den aktive
[bundne OANDA M1-kilden](/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source/manifest.json)
har ett OHLC-tidsstempel per minutt og kan ikke ordne intraminutt høy/lav.
En spredt, umanifestert Dukascopy-tickcache dekker ikke aktiv 2024-TRAIN;
GC-futures er heller ikke fasit for XAU. Den avgrensede 0,5/1,0-ATR-
sensitiviteten over er fullført og konsumert; den valgte ingen ny terskel.
Neste målavgjørelse gjelder semantikken i delvis observerte tilstander,
uten modellscore, VAL eller TEST som terskelvelger.

Fullprofilen ruter 253 felt per TF til modellen. ATR14, EMA20-avstand og rå
støtte-/motstandsavstand har kontrollert formel, variasjon, TRAIN-skala og
kausalt valgt lukket bar i målt scope. 95 tidligere parkerte felt per TF
inngår nå i fullprofilen uten selvstendig numerisk kvalifisering. Råaudit
beviser ikke verdier etter normalisering, kvaliteten på de observerte nivåene
eller prediktiv bruk. Én [avgrenset postnorm-plan](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_001.json)
SHA-256 `5e4f5f5cc96db4e684fcb30dba3ba1e824217d9e445562917681c103e898d62c` er
innrømmet, ennå ikke claimet: åtte faste cache-rader på fem TF, alle 253 felt
per TF mot modellens normalisering, uten forward/fit/TEST. Den binder eksakt
kjent kildeavvik i en tapsfunksjon; avviket gir ikke fit-/forward-adgang.
Etter terminalkontroll gjenstår numerisk kvalifisering av 95 felt per TF og
bidrag på kronologisk utviklings-OOF.

Vi har ikke kjørt omfattende trening gjennom flere år. Tidligere fem-TF-fit
var ett pass på 47 602 TRAIN-foreldre (2 976 steg); oktoberparet var ett pass
per arm på 29 849 foreldre (1 866 steg hver), med senere score uten refit.
26 måneder beskriver datadekning, ikke treningsmengde. Den lagrede
M5-retningsrangeringen på senere utviklingsrader falt fra AUC ca. 0,845
ved +1 minutt til ca. 0,498 ved +5; på foreldre med begge observerte
retningsfasiter skiftet 48,09 % klasse mellom disse tidene. Det avgrenser
hvilket mål og tidspunkt som faktisk ble lært, ikke alt mulig signal.
Først etter låst mål og kvalifiserte inputs kan én forhåndsbundet
tilstandsfit sammenlignes med enkle referanser på senere utviklingsrader.
TEST og Entry-fasen forblir stengt.

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

## Bevarte utviklingsmålinger og senere Entry-spor

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
historiske dynamiske M5-Entry-implementasjonen er `entry_action_q_loss` inaktivt;
Entry-kvalitet er predikert terminal-netto minus predikert tidlig MAE fra
to separat trente komponenter. Dette er en mål-/beslutningsmismatch,
ikke en bevist featurekodefeil. Den eldre samme-rad-kontrollen i
`docs/DYNAMIC_MTF_STATE_NO_EDGE_DIAGNOSIS_20261103.md` fant motsatt
M5-candlefortegn AUC 0,5185 mot modell 0,4979 på 7 812 senere
utviklingsrader, men ingen Entry- eller økonomisk edge. EMA20/ATR14 og rå
støtte-/motstandsavstand har varierende TRAIN-normaliseringsstatistikk på
alle fem TF; faktisk rå tensorparitet ble kontrollert nedenfor.

Én [rå inputaudit](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001.json) SHA-256
`b5a6785ab4666d03663ce4ad3878f6c84a839821d4c3f2598cc6d2f2e01ef0d9`
er fullført én gang. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001/RESULT.json) SHA-256
`62f37fc0ae50d967b2b8a449a54d72ed66b2fc5a08ad11ae0a381f68f92388e7`
og [TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001/TERMINAL.json) SHA-256
`3f9abd4bf5c822cf62e119abd17fa1dab7004a53436a39c460428c794e70cec9`.
Alle 36 172 foreldrene valgte siste lukkede cachebar på fem TF; M5-valget
hadde eksakt samme barstart som forelderen. På åtte forhåndsbundne rader
var rå ATR14, EMA20-avstand og støtte-/motstandsavstand bitlike mellom
cache og modellens gather-tensor. Alle fire felt var finite og ikke-konstante
på alle fem TF over foreldrene. 0 modellforwards, 0 optimizersteg, 0 TEST;
15,92 s, `ru_maxrss=4 388 908 KiB` og verifisert `memory.max=4G`.
RSS-tallet overstiger 4G-grensen; faktisk cgroup-topp ble ikke bevart, så
regnskapene må ikke omtales som samme mål. Kjøringen avsluttet 0 uten OOM.
Kontrollen beviser ikke post-normalisering, per-felt modellbidrag, faktisk
nivåobservasjon eller Entry-verdi. Planen er konsumert og kan ikke kjøres igjen.

Parkert hypotese for en senere Entry-fase: ett direkte tap på
`terminal-netto − MAE`-kvalitet. Den er ikke gjeldende neste steg og gir
ingen fit-adgang. Tidligere 191s 256 CONTROL-rader hadde negativ observert
sidekvalitet i snitt; det kvalifiserer heller ikke Entry.

En siste metadataavklaring fant ingen lagret, foreldrejustert 2D-matrise
med hele den native inputflaten i gjeldende bundne kandidat-artefakter. [CAUSAL_INPUTS](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_FEATURE_DELIVERY_20261106_001/CAUSAL_INPUTS/CAUSAL_INPUTS.json) SHA-256
`2fdea20ebeb90ec8a027f067726f1dd488139e4a1662ee61a75920b1ab764ae6`
har bare justert nåtilstand/alder/masker. [M5-cachemanifestet](/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003/MULTI_TF_V4_CACHE/manifest.json) SHA-256
`3f4c3b4b54b08931850b716e0b7cf4b5de0b54c42b0fee813cde6f1324371da4`
har en kausal M5-matrise på 1 140 982 × 253 på baraksen, men den må først
kobles as-of til beslutningene og mangler native signal-/kontekst-/GC-flaten.
Eksisterende eier `gx1/features/htf_features.py:slice_multi_tf_v4_window`
velger siste lagrede M5-barstart ≤ `issue−5 min`. Ny rå inputaudit
har bevist likhet med alle de 36 172 foreldrebarene i dette bundne
pre-november-scope. Denne M5-only-baselinen er
ikke innrømmet: den kan ikke avgjøre full native input- eller Entry-verdi.

Ingen ny native eller enkel modellfit er innrømmet. Gjeldende neste steg
er tilstandsmålets semantikk og observerbarhet, deretter indikatorenes
normaliserte konsum og kvalifisering. En senere tilstandsfit krever én
forhåndsbundet kronologisk hypotese, målmaske, referanse og akseptregel.
Oktober er undersøkt utvikling; TEST er forseglet. Senere retningsverdi
er ikke påvist.

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
gjelder D1-swing, ikke M5. Ingen slik kjøring er innrømmet. Neste
kvalifiserende hypotese gjelder fem-TF-markedstilstand med forhåndsbundet
kronologi, dekning, mål og enkle referanser. Direkte Entry,
path, kost og økonomi hører til den senere fasen.

Aktivt mål er fem-TF-markedstilstand med M5-prognosetakt, kausal lukket
M15/H1/H4/D1-kontekst, åtte spesialister og niende D1/makro.
`training_enabled=false`. TEST er forseglet; bare forward-journalen scorer
sin periode på forhåndsbundet måte. Exit,
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
