# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Prosessobservasjon og eksakte terminalkvitteringer overstyrer prosa.

## Nå: fullført avgrenset ordrestudie, D1-årsak fortsatt delvis åpen

Årsaksmålet i chat01a20035-2836-7f62-bb88-9e24603a51a5 er IKKE fullført.
Alle fem modellporter erfalse,003/004execution_authorizedfalse/remaining0/
relaunchfalse. PID41507 er borte. Ingen ny plan/fit/hovedepoke er innrømmet.
TEST, M1-retning, GC, Entry/Exit, broker, live/paper, handel og promotion er stengt.
Kandidaten tasikkeibruk: fireTF bedres, D1forverres, generell stabilisering/edge er ubevist.

004 er COMPLETE/exit0,400,896s, faktisk ferdig22.11kl20:54:33UTC/21:54:33Oslo.
Controller lukket allefemporter straks etter wrapperexit og bekreftet fraværendePID.
Kjøringskildeeb42c4de73b1a027cd28123d7646ecd2e009bf98,77eiere/source/input/plan/
policy/Windows-/WSLbootbookends og M1firewallPASS. Policyadmittedsnapshot
b59f94980e597ca1c6822b9db3f06321c259f3cef5cf1e032b14ef55aeebde60 består.
Senere metadata-/handovercommit endrer ikke den historiske kjøringsidentiteten.

Eksakt plan `configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004.json`,
SHA6c061afd7963b787f714740813f2499e7e763e4a48ba0f13123b78222e194ba2.
Ny owner `gx1/audit/continue_native_future_state_bias_suffix_v1.py` bevarer003
og MAIN. Bare lokal strengV3-restore, separat004lineage og navngittguardroute;
72gamle eiere byteidentiske,2guard-eiere med eksakt avgrenset AST-/shellbro.
Ingen modell/input/target/loss-matematikkendring eller falskMAINtrajectory.

Én ferskmodell/2AdamW/2CPloads/2restores/2technicalfits,319FW/238BW/updates.
Først ekte003omordnetCP128→192: alle64stegrapporter og8statekomponenter bitlike.
Så streng kaldrestore av ekte192 og bare174gjenstående batcher fra samme
forhåndsvalgte366-permutasjon. Ni usikrede003updates193–201 regnes somtapt,
aldri lagt til på nytt utenfor sammeordre. EndeligRNG bitlik kvalifisertkontroll.
Tre nye durableCP ved total256/320/366 med full AdamW/RNG/step/orderprefix,
atomisk publisering og diskreadback.81eval/20727fasteTRAINrader,1capture;
0normaliserings-/priorfits/CONTROL/VAL/TEST. Ingen delcheckpoint-/seedutvalg.

## Målt effekt på samme faste TRAIN-kohort

| TF | Fasit OPP | Kontroll P(OPP) | Permutert P(OPP) | SideBrier kontroll → permutert |
|---|---:|---:|---:|---:|
| M5 |50,883%|53,635%|52,325%|0,250322→0,249566|
| M15 |52,062%|55,785%|52,351%|0,250390→0,248858|
| H1 |54,460%|58,297%|56,876%|0,249370→0,248119|
| H4 |55,488%|61,567%|61,057%|0,251117→0,250360|
| D1 |54,321%|64,097%|66,625%|0,258023→0,263711|

Allefire ikke-D1sideBrier/logloss bedres i allefemTRAINårsblokker; D1erblandet
påårsblokker ogverre samlet. BareM5/M15 slår BEGGE konstanteTRAINpriorer på
sideBrier/logloss. H1ermarginaltverre,H4/D1tydeligverre. D1AUC0,51954→0,51082;
activityresultatene erblandet. TRAIN er gjenbrukt utvikling, ikke generalisering.

Kausal følsomhet for denne faste rekkefølgen er bevist, betinget på ekteCP3328,
verifisert runtime og sammeRNGstrøm; endret batch→dropoutmaske er delav
intervensjonen. Dette isolerer ikke hele årsaken tilD1skjevhet eller indikatornytte.
Kontrollmetrics har originalfrozenFINAL somproducer, legitimert ved003s rawlogit-
paritetsbevis. Ingen fabrikkert003RESULT eller ommerking avgammelproducer.

## Konkret gjenstående mekanisme og minste videre vei

Begge armer har identisk sluttdel3328:3694. Intervallene ved3584og3694 gir:

| TF | Hele TRAIN OPP | Samme366batcher OPP | Batchmedsidestøtte |
|---|---:|---:|---:|
| M5 |51,118%|52,893%|366|
| M15 |51,854%|54,604%|366|
| H1 |53,635%|57,296%|365|
| H4 |55,220%|62,839%|333|
| D1 |54,342%|76,471%|67|

D1:3588OPP/4692kjentefasitceller,299av366batcher utenD1sidestøtte.
Ingen ukjentunikD1barantall er oppfunnet for denne suffixen. HeleTRAIN har
162distinkte kjenteD1målbarer bak44566fasitceller;626av3694støttede batcher,
alle626ensidige. Gjentatte fasitceller er ikke uavhengige observasjoner.

Objektivets analytiske konstant-D1prior i suffixen har konservative grenser
75,9586–76,7651%, motheleTRAIN53,4573–55,1239%. Derfor trekker samlet
suffixBCEgradient en konstantglobalpriorOPP. Dette er algebra på lagrede
support/intervalltall, IKKE faktisk ikke-lineær AdamW-gradient eller nypriorfit.
Åstokke innen sammeUP-tunge suffix endrer ikke grovregime-/klasseeksponeringen.
16frosne checkpoints er sammenstilt med ALLE originale intervalltall i reviewen;
korrelasjon der er ikke en ny isolert komposisjonsintervensjon.

EksaktBrieridentitet påD1: sannhetsvarians0,2481329066 +kvadrertmiddelbias
0,0151389462 +nettofeatureavhengigvarians/kovarians0,0004395517 =0,2637114045.
Middelbiasleddet er97,18% av positivtoverskudd overeksaktkonstantreferanse;
kontrollens tilsvarendeandel96,63%. Dette avgrenser kalibreringsproblemet,
menbeviserikke læringsårsaken ogskaperingenAUC/edge eller gyldigkalibreringsfit.

Neste begrunnede årsaksspørsmål er grov nyligregimeeksponering og sparsom
D1-gradientstøtte, med samme genuine funksjon. En fremtidig isolert kandidat
må beskrive nøyaktig hva input-/targetmiks endrer og holde kilde/RNG/recipe/
ressurser bundet. Ingen klassekvoter/FLATommerking/TFfjerning, tilfeldig
prøveløkke eller automatisk nyhovedepoke. Kvalifiser én begrunnet kandidat,
ikke flere alternativer samtidig. MAINfit/generaliseringsscoring krever eget
avklart omfang; TEST forblirforseglet.

## Ressurser og verifikasjon

Faktisk004topp: GPU64°Ckjerne/70°Cminne/210,15W/9798MiB; CPU43°C/35,718W.
Minste fysiskWindowsledig55,103GB og commitreserve50,103GB. Ingen målt
ressursgrense ble brutt under400smodellkjøring; dette er ikke universellgaranti.
Vakter: GPU250Wdraw/limit,75/80°C,12288MiB; CPU80°C/125W;8GiB fysisk/
commitreserve og prelaunchhelecgroup+reserve;20G/512M,8tråder/CPU0–18/tasks128.
Eksisterende signedWindowsv2bro måler ferske10kjernetemperaturer/packagewatt,
fysiskRAM/commit ogvedlikeholds/rebootstatus. Cacheverdier nulles. Manglende/
ugyldige målinger stopper. Ingen global sikkerhets-/driverrollback/powerendring.

54nye004modul+74guard+1gammelguardsmokePASS; mandatoryprecommit3kontraktfiler/
64casesPASS. Faktiske27feature/30target/15cohortNPYheadere ognorm/provider/clock-
identitet PASS førCLAIM,0array-/CPpayloads. Fulltensoridentitet førførsteFWPASS.
Første CLI hadde integratorens ekstra timeoutwrapper ogbleavvist førscope.
Første canonicalpreflight manglettelemetri; samme2squery(.51s) og uendret
canonicalpreflight bestod. Initialtelemetriårsakukjent; ingen tidsvakt ble svekket.

003ble tidligere stoppet av konservativmaintenanceproxy20:13:02UTC, ikke
påvistoverlast eller dokumentertplanlagtrestart. Vaktenkanutløses av idleworker;
ORtermen vedbreach erukjent. Senere snapshot20:22:36 fantingenworkers/reboot/
setup/events; COMIsBusyUNKNOWN. Ingen vedlikeholdsvakt er fjernet.
003s366kontroll gjenskapte alleoriginalsteps/8FINALstates og20727radersbegge
rawlogitarrays bitlikt pådriver610.88 motoriginal591.86. Ektev3CP128/192 ble
brukt av004;003gjenstartesaldri. Tidligere001ble avbrutt av beviseligWindowsUpdate-
restart09:30(User32ID1074); ikke bland dens sterkere OSbevis med003proxy.

## Artefakter og ubeviste grenser

004rot `/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004`.
RESULTSHA2ddf1df677b381dbfbc595d91735823edcf6ca9af6ffb0e31c51f72f5543887e,
TERMINALSHA22638ed8505952cc9c182c507adfb04b24696074a3d0db6769a5c5e3ac3ce763.
Bevisrot `/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_BIAS_CAUSE_20261122_001`.
GjeldendeSUFFIX004_RESULT_REVIEW001.json
SHA068de28cc55effea1bef2bdd402753ec8d28fbbf4249158f28012b1fc48f02e4.
Denne reviewen binder all5TF/allårsblokker, konstante referanser,16CP-tabell,
suffixpriorgrenser, Brieridentitet, kilde/counters og faktiskeubevistegrenser.

ALLE3nyeCP+capture/result/terminal/admittedpolicy/review er Macdisk-sikret:
`/Users/andrekildalbakke/GX1_BACKUP/CURRENT_BIAS_SUFFIX004_COMPLETE_20261122_001/MANIFEST.json`,
SHA2b7e08b1812b70d01d065d3f066eaeee53778d50eeeb04be2522a7953424918a,
23artefakter297436263bytes. Atomisknoreplace/kildeSHA-statbookends/fsync/
F_FULLFSYNC/faktiskMacdisk-SHAreadbackPASS;0modelldeserialisering.
Preflightbackup22artefakter5188194bytes, manifest42d18997...403c56.
Tidligere003niCPstopbackupmanifest e3e4e0c1...0d6d3e0 og alle16MAINCP
(CURRENT_FUTURE_STATE_ONE_EPOCH_20261122_001,manifest0d8b270c...98b30)
samtfrozencapture(CURRENT_FUTURE_STATE_BIAS_FROZEN_20261122_001,faa02cbb...bc21)
beholdes. IngenDATA/RUNSrydding; bruk retentioneieren.

Den originaleén-epoken er COMPLETE/konsumert:354550TRAINrader2021–2025,
M5_primary/B96/E256/3694updates/5079FW, åttegenuinefamilier+61makrofelt og
M5/M15/H1/H4/D1. Én epoke forbedret mestaktivitet; sideBrier/logloss varverre
ennpriorpåalleTF. OPPargmax alene erikkekollaps: wholeTRAINprior eralleredeOPP
påalleTF. MAINresult/frozen/permutert måles påsine egneeksaktestøtter;
sammenlign ikke heleTRAINscore direkte med den faste20727kohorten.

Makrofirst-vintage/revisjonsledger, komplett historiskhelligdagskalender og
universellfeilfrihet erfortsattubevist. Ingen lærtverdi frahverindikatorfamilie,
uavhengiggeneralisering, bedrehandelsbeslutning eller realisertøkonomi erpåvist.
