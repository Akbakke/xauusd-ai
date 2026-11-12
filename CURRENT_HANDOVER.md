# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_PAIRED_FULL_BOUNDARY_OBSERVATION_ADMITTED.
Målet11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. Korte runder er bevisst feilsøking, ikke kapasitetsdom.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.
Alle tidligere engangsplaner er konsumert; nyeste observasjon er COMPLETE/CONSUMED. Training,
native_training/native_launch/full_epoch er false. GX1TFActivityShortLearning001,
GX1NonM5ShortLearning001 og begge c171-oppgavene er Disabled.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion stengt; GC på pause.
Én ny lesende paired boundary-observasjon er innrømmet; ingen trening. Ingen operasjonell blokkering.

## Indikatorer og sammensetting

Alle8genuine TAfamilier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning, varierende TRAIN-støtter og oppdatering i11encodere/228tensorer
beviser mekanikk, ikke prediktiv familienytte. Mean+last kansellerer ikke generelt;
fast MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler entydig
samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features/TF og kausalitetsvakter er bevart. Ingen generell normaliserings-
eller konstantinputfeil er vist. [Samlet oversikt](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).

Tidligere felles nonM5-sideforskyvning ble målt. Separate sidehoder endret læringen,
men ga ingen ekstra M5gevinst og forverret aktivitetstapet. Retained-gradientkontroll
bekreftet motstridende TF-bidrag til den delte aktivitetsraden; ingen tapsformelfeil.
Sparse batchvekter blåser ikke opp D1s optimale konstante aktivitetssannsynlighet.
Se [sideforsøket](docs/TA_NON_M5_SHORT_LEARNING_REVIEW.json) og
[aktivitetskontrollen](docs/TA_ACTIVITY_PRESSURE_REVIEW.json).

## Nyeste faktiske læringsmåling

[Resultat, alle10celler og klasse-/kvartalsregnskap](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json).
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001.
Kilde7dcddb37586cdcf297ffac2e8b1f643912f4c50a uendret under kjøring; boot642.
RECIPE4ae0082f, PLAN63b7702e, faktisk vindua4d7d93e, RESULTaca65dc4,
campaignterminala2f80817, cursor2f7b7593. COMPLETE/CONSUMED, alle exit0.

Samme4096kronologiske TRAIN-rader/256updates som originalshort002,1.–21.mai2024.
Originale startvekter og INITIAL-/FINAL-trenings-RNG, fersk tom AdamW, uendrede
hyperparametre/features/targets/tapsvekter. Bare ti separate aktivitetshoder på
M5/M15/H1/H4/D1 forS/F:20klonede tensorer/1290parametre. Separate nonM5-sidehoder av.
Hele4096initial før første update: akser/targets/alle sidekanaler byteeksakte;
aktivitetsavvik maksS9,5367e-7/F2,3842e-7, tapsavvik2,3842e-7 innen faste toleranser.
Hele4096målt igjen final.256initial+256train+256final=768forwards,0skips.
Worker948,98s; faktisk fit350,19s. Uendret20GiB/512MiB,CPU0–18/pids128,8tråder,
250Wlimit/310Wdraw/85Ccore/80Cmemory/GPU12288MiB;1800smodell/3600svindu/3900sytre.
792telemetripunkter, topper55C/64C/133,59W/2172MiB. Ingen vaktbrudd.

| Samme brukte TRAIN | Initial | Original final | Sidehoder final | Aktivitetshoder final |
| --- | ---: | ---: | ---: | ---: |
| Opprinnelig totaltap | 1,878003 | 1,892871 | 1,884922 | 1,884324 |
| S/M5 side-AUC | 0,546185 | 0,551466 | 0,551231 | 0,551352 |
| F/M5 side-AUC | 0,546125 | 0,553079 | 0,553037 | 0,553147 |
| S/M5 side-Brier | 0,247999 | 0,247924 | 0,248301 | 0,248027 |
| F/M5 side-Brier | 0,248788 | 0,248193 | 0,248318 | 0,248218 |

AUC er rangering, ikke treffprosent eller økonomi. Alle tre totaltap forverres fra
initial. Aktivitetsvarianten gir mindre forverring enn original med0,00854651 og
sidevarianten med0,00059749. Ingen entydig samlet M5gevinst mot originalfinal:
S/M5AUC litt lavere, F/M5AUC litt høyere; beggeM5Brier/log litt dårligere.
Aktivitets-Brier bedre enn originalfinal i alle10celler, aktivitets-log i9av10.

Δoriginalt tap for aktivitetsvarianten=+0,006321593: aktivitet+0,000967350 og
side+0,005354243. Netto aktivitetsregresjon er87,23% mindre enn originalens,
men ingen87% forbedring i prediksjon eller edge. Sidevarianten hadde i stedet
aktivitet+0,013960765 ogside−0,007041688. Separate armer beviser ikke additiv gevinst
fra en kombinert modell. Første to kronologiske kvartaler forverres, siste to bedres.

## Gjenværende målt feil og neste kontroll

S/D1aktivitet:2011AKTIV-/215FLAT-rader. AKTIV-bidraget bedres−0,020664163,
FLAT forverres+0,066652219 før/5; netto+0,009197611 i totaltapet.
På faktiskFLAT øker P(aktivitet)88,49→93,35%. Q2 rommer207av215FLAT-rader:
observert aktivandel68,87%, mens gjennomsnittlig P(aktivitet)89,82→94,00%.
Q2forverres, de tre andre delene bedres. F/M15-H1-side trekkes også opp i alle
kvartaler; Q1–Q3ender72–77%UP mot51–57%observertUP, med verre Brier/log.

S/D1s isolerte eval-biasderivert med original/5 er+0,007071016initial og
+0,011873531final: gjennomsnittstapet ønsker lavere logit. Samtidig har175/174batcher
negativ derivert,49/50positiv og32tom. Færre positive bidrag veier mer samlet.
Dette er eval-endepunkter, ikke de historiske treningsgradientene eller AdamW-stegene.

Retained faktisk treningsgradient/Adam er nå kontrollert:1TRACE-NPZ/2CPUcheckpoints,
0modeller/forwards/optimizerkonstruksjoner. Alle2560aktivitets-biasderiverte er
sjekket med sentraldifferanse; maksavvik1,14e-11. M5fangede gradienter/momenter
kvalifiserer aritmetikken. D1sluttbias matcher rekonstruksjonen innen8,23e-8.
178negative/46positive/32tomme treningsbatcher; gjennomsnittlig derivert er positiv
både før og etter klipping. Adam-historikk gir likevel netto positiv biasendring.
203opp/53ned og bevegelse i32tomme batcher er rekonstruert, ikke direkte fanget.
FP64-decaybidraget er nominelt; det er ikke målt faktiskFP32-decay.

FaktiskD1bias øker0,00264197, mens gjennomsnittlig logit på2226informative rader
øker0,52717903. Biasdelen er bare0,50115%; resterende0,52453706 blander vektendring
og endret representasjon. Dette er additivt endepunktregnskap, ikke kausal prosent.
Bias alene forklarer ikke problemet og gir ikke grunnlag for en biasrettelse.
Kvittering SD1_TRAINING_OPTIMIZER_REVIEW.json7fc4a63a i samme læringsROOT.

Ny lesende INITIAL-avlesning er COMPLETE/CONSUMED:1eksisterende checkpoint,
1modell/256B16forwards/4096rader,10hooks/2560inputobservasjoner og0oppdateringer.
Alle12baselinefelt og256originale batchtap er byteeksakte. Vekter/buffere/RNG,
modi/flagg/identiteter/hooks er uendret;0optimizer/backward/RNGrestore/vektskriving.
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_INITIAL_INPUTS_20261112_001.
PLANe4f988f4, RESULT6aac9103, TERMINAL84433f8c; kildef5d2abe5 uendret.
Arbeider327,54s; controller329,94s.263telemetripunkter, topper46Ccore/50Cmemory/
132,16W/554MiB; alle vakter bevart, exit0. Opprinnelig metadataparsers formatbom
for heartbeat/peak er bevart og korrigert i GUARD_TERMINAL_REVIEW.json; nativekvittering uendret.

Etterfølgende analytisk vektregnskap fb695cf6 brukte3NPZ/2CPUcheckpoints,
0modeller/forwards. Alle20kanaler/4kvartaler; opprinnelige B16-nevnere,/5,Fvekt1
og M5q/N bevart. Maks initialaffin-replay4,88e-7; taps-/logitkonservering4,44e-16.

| Originalt totaltap, samme4096TRAIN | Verdi |
| --- | ---: |
| INITIAL | 1,878002535 |
| FINALhoder på fast INITIALrepresentasjon, analytisk | 1,871377344 |
| Faktisk FINAL | 1,884324128 |

Hodeendring på fast h0 bedrer totaltapet−0,006625191; representasjons-/avrundingsrest
forverrer+0,012946784. D1aktivitetslogit øker0,527179: hodevekter0,103682,
bias0,002642 og resten0,420855. På faktiskFLAT går Paktivitet88,49→89,30→93,35%.
F/M15 PUP går53,25→55,18→69,94% mot52,08%observertUP; F/H1 går53,51→55,45→70,88%
mot53,97%observert. Mellompunktet bruker FINALhoder på fast h0, ikke en nytrent modell.
Representasjonsresten bedrer noen kanaler og siste to kvartaler, men forverrer andre
og første to kvartaler. Dette er ordnet endepunktregnskap, ikke unik kausal attribusjon.

Retained48pilot er nå PASS444089b4:3NPZ/2CPUcheckpoints,0modeller/forwards/trening.
Originale batcher0/123/246;10akse-/targetfelt, alle sideoutputs og beggeM5h0 byteeksakte.
Maks initialjoint-hidden-replay1,583e-6, allTFh0-replay7,63e-7, raw3,58e-7,
alignment4,32e-7 og opprinnelig batchtap2,41e-7. Gammel FINAL-fase er ikke brukt.
S/D1har kun14informative rader, alleACTIVE og0FLAT; oppforskyvningen bedrer tapet
på disse. Piloten kan derfor ikke forklare de215FLAT-tilfellene i fullprefixet.
F/M15-H1har bare15/11kjente retninger: queryprojection, joint og tidligere restledd
øker både middel-logit og sidetap i valgt rekkefølge. Ingen enkelt årsak er isolert.

Neste fulldekkende avlesning er kildekvalifisert og ADMITTED etter141fokuserte tester
og metadata-preflight med0modeller/arrays/checkpoints/claims. NyROOT:
/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_BOUNDARIES_20261112_001.
PLAN6f0bc72e, PREFLIGHT3d964d2d.1invocation/0claim/0resultat; ingen gammelplan relanseres.
Én modell,2eksisterende checkpoints, strict INITIAL→FINALvektovergang uten RNG-/
optimizerrestore;512B16forwards/8192observerte rader på samme4096kronologiskeTRAIN.
Fang begge endpointenes512Djoint-source,128Djoint-hidden og alle10TFhodeinnganger.
Alle12originalfelt/256batchtap per endpoint og tidligereINITIALh0 må være byteeksakte.
20headregistreringer/5120headkall,512jointsource/512jointhidden/1024queryinputkall;
0backward/optimizer/autograd/RNGrestore/vektskriving.64MiBNPZ/2MiBJSON.
Sammeproducer12GiB/512MiB,CPU0–7/pids64,8tråder;250Wlimit/170Wdraw/65Ccore/80Cmemory,
GPU12288MiB,900sarbeider/1200svakt. Kilden fryses gjennom hele observasjonen.
Etterpå kan beholdteverdier gi begge ordnede hode/query/joint/source-regnskap og4
allerede kvalifiserte semantiske sourceblokker. Ingen trening eller modell-/tapsendring.

## Kvalifisering og ubeviste grenser

302fokuserte tester og obligatoriske commit-vakter bestod før launch.92kildeeiere
var uendret etter trygg fysisk boot641→642. Preflight og faktisk immutablevindu
hadde likt innhold, ulik JSON-formatering; faktiske bytes fikk egen cap-/oppstartskontroll
før claim. Ingen endring av plan/kilde. Originale feilkvitteringer er bevart.

Uavhengig læringskontrollf79b828f:5NPZ/60akse-targetfelt/600scoreceller/6tapsvektorer,
0modeller/checkpoints/forwards/updates. Maks replayavvik total3,90e-7/komponent4,64e-7.
S/D1klassekontroll7ccf551f:2NPZ/0modeller, originalB16-nevner/tommebatcher og/5;
konserveringsavvik2,22e-16. RNG/checkpoint-identitet bygger på bundne nativekvitteringer.
Windowsoppgaven er lesende bekreftet Disabled/LastTaskResult0 etter vellykket disable;
kvitteringsformattering av tom NextRunTime feilet først, uten å endre treningsresultat.

Brukt avhengig TRAIN og ett seed/256updates. Ingen signifikans, senere generalisering,
kapasitetsdom, økonomisk edge eller promotion er etablert. Familienytte og trendlinje-
identitet er fortsatt åpne spørsmål; fullhistorikk/30–40epoker krever senere kvalifisering.
