# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

## Aktivt årsaksmål 22.11.2026

Brukeren bestiller nytt mål for å finne årsaken til OPP-skjevheten.
Målchat01a20035-2836-7f62-bb88-9e24603a51a5; eiernative_future_state_bias_cause_20261122_001.
Frossen16checkpointdiagnose klargjøres på samme20727TRAINkohort, E256,
med1296modell/backboneforwards og324ekstrareadoutforwards,0optimizer/autograd/fits.
CP256/FINAL2x2readout/hiddenbytter skal skille funksjonelle bidrag; de alene
beviser ikke treningsrekkefølgens årsaksrolle. Velg senere én isolert intervensjon
ut fra målt onset; ingen automatisk hovedepoke eller nyTEST/CONTROL/VAL-scoring.
Alle fem treningsporter er false. Eksakt plan/runtime/admisjon og faktiskCUDA gjenstår.
Minste nødvendige kildeutvidelse er én separat readonlyvakt/diagnoseeier fordi
fortidige entrypoints ikke støtter dette. Ingen numeriskmodell/input/target/lossrecipe er endret.
17fokuserte diagnosecaser og42vaktcaser samt faktisk bundet norm/input/prior/CP-
kontraktmetadata består. Begge kildebroer bevarer gamle vakter; peerreview av
konsument/restore/masker/tellere/feilrapportering er utført og konkrete gap rettet.
Bevisrot: `/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_BIAS_CAUSE_20261122_001`.

## Undersøkt retningsskjevhet 22.11.2026

Brukeren bestilte undersøkelse av OPP-skjevhet og stabilitetsalternativer.
Én ny capped CPU-audit er COMPLETE/exit0/konsumert,21,971s,audit4GiB/512MiB/1tråd.
Bevisrot `/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_DIRECTION_BIAS_20261122_001`:
`RESULT001.json`, `INTERPRETATION001.json` og `PEER_REVIEW001.json` er gjeldende.
12nativeTRAINtruthfelt og16originale checkpoints er lest; allearray-/filhasher og
komplette opprinnelige checkpointproofs stemmer.0modeller/forwards/optimizerupdates/fits.
Kanoniske masker/targeteier er gjenbrukt; ingen modell-/mål-/taps-/recipekode er endret.
Alle62kildehashene stemmer også i etterkontrollen. Første metadata-preflight fant
én feilimport i audithjelperen; denne ble rettet før CLAIM/payloadlesing. Originallogg beholdes.
Alle fem treningsporter er fortsatt false; ingen ny epoke er innrømmet.

OPPargmax må tolkes riktig: den konstante wholeTRAINprioren gir allerede100%OPP
på alle5TF. MangeOPPvalg alene beviser derfor ingen ny kollaps. Det robuste
negative resultatet er dårligere sideBrier/logloss enn prior på alleTF.
H4/D1s laveste finale P(OPP|aktiv) er56,49/56,03%, over faktisk55,22/54,34%.

| TF | Hele TRAIN OPP gitt kjent side | Siste256batcher OPP |
|---|---:|---:|
| M5 |51,12%|53,68%|
| M15 |51,85%|56,33%|
| H1 |53,63%|59,91%|
| H4 |55,22%|67,21%|
| D1 |54,34%|76,92%|

Slutten av det kronologiske passet har beviselig annen labelfordeling.
Dette styrker rekkefølge-/sluttpåvirkning som hypotese; det beviser ikke årsaken
til sluttprognosen. D1s siste256batcher har bare13unike kjente målbarer (10OPP/3NED).
D1harfasit i626av3694batcher; alle626erensidige.3068manglerD1sidefasit,
lengste opphold82batcher. Gjentatte mål gir ikke tilsvarende uavhengig støtte.
WholeTRAINOPPandel perunikmålbar er nesten lik prognosecelleandelen; repetisjon
forklarer derfor ikke alene den samlede basefrekvensen.

Direkte sidegradient er null når side er ukjent. AdamWfortsetter likevel med
nullgradienttensor/momentum ogglobalstep; D1headAdamstep3694mot626støttede
batcher er verifisert i checkpoint. SistekjenteD1batch varNED og siste9utenstøtte;
finalt momentum trekker biasNED. Dette er ingen bevist årsak tilOPPskjevhet.
BiasparameterdeltaH4/D1bare0,00587/0,00721logit forklarer ikke heleprognosen;
vekt·hidden er også positiv på allekjentesiderader. Representasjonsnytte/årsaken
krever tidsdelte frosne prognosemålinger. Ingen aktiv UP/DOWNsign-/maskefeil er påvist.
M5_primary-backbone styres avM5aktivitet+side; øvrigehoder følger bevegelighidden.
Globalnormklipping varaktiv2199/3694steg ogkobler allehoder i skaleringsfaktoren;
fullMAINgradientfordeling perhode er ikke lagret. Ingen gjettede tapsvekter innføres.

Anbefalt videre: kvalifiser én avgrenset lesende kalibrerings-/tidsblokkscoring av
lagredecheckpoints først. Førsteisolertefitkandidat er fastomrekkefølge avhele
TRAINbatcher, medallekausalevinduer/rader/masker/klassefrekvenser/tap/recipebevart.
Lavere/avtakendeLR er et eget alternativ ved måltoverskyting; støttefriheadfreeze
trenger egen perheadAdamstep-/checkpoint-/kaldreplaykvalifisering og gjelder særligH4/D1.
Kalibrering kanbedre sannsynligheter, ikke skape rangering/edge. Ingen klassekvoter,
FLATommerking, TFfjerning, automatiskflereepoker eller nyTESTadgang.

## Gjort og vurdert 22.11.2026

Brukerens bestilling er gjennomført: nøyaktig én epoke, timevis oppfølging,
sikrede checkpoints og resultatvurdering før eventuell fortsettelse.
Målchat `01a20035-2836-7f62-bb88-9e24603a51a5`.
Eksakt plan `configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json`,
SHA c3e769fc5468739b576144f2cb2e5bf572d6b662a3c490ee56d8641fde1dd561,
er COMPLETE/exit0, konsumert og aldri relanserbar. Alle fem treningsporter er false.

354550 TRAIN-rader2021–2025, M5_primary, TRAINB96/EVALB256, én pass,
3694optimizerupdates/5079forwards,0skips/normaliseringsfits/priorfits/CONTROL/VAL/TEST.
Fersk modell/AdamW; ingen studievektwarmstart. Åtte genuine familier,61makrofelt,
M5/M15/H1/H4/D1 og alle eksisterende masker/kausalitets-/ressursvakter er bevart.
M30 er fortsatt separat utvidelse ved siden avM15; M1 brukes aldri til retning.
Kjøretid5664,220s (94min24s), faktisk avsluttet22.11 kl00:01:47UTC/01:01:47Oslo.
Original kildeHEADf96d89f88/runtime006/Windows670 og frosne policybytes var
uendret gjennom jobben; terminalens kilde-/input-/plan-/bootbookends består.
Første frakoblede oppstart mistet WSLWindows-relay før CLAIM/modell; ingen
admisjon var brukt. Åpen SSH/WSL-sesjon claimet gjenværende tillatelse én gang.
Ingen brukt plan ble relansert. Alle originalkvitteringer/feillogger beholdes.

16 checkpoints er lagret med atomisk publisering, full diskreadback og JSON-receipt:
INITIAL, hver256optimizerupdate til3584, og FINAL ved3694. Kontroller omtrent
hvert60minutt bekreftet framdrift, uendret kilde/policy og til slutt stoppet prosess.
Guardtopp:66°Ckjerne/74°Cminne/220,45W/9770MiBGPU; faktiskeffektgrense250W.
Vakter var85/80°C,310Wdraw/300Wlimit/12288MiB. CPUpackagewatt/temperatur
måles ikke av den eksisterende signerte broen. Cgroup20GiB/512MiBswap/128tasks,
CPUaffinity0–18 og8numerikktråder er bevart.

Runrot: `/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001`.
Oppfølging/admisjon/vurdering:
`/home/andre2/GX1_RUNS/GX1_FUTURE_STATE_ONE_EPOCH_FOLLOWUP_20261121_001`.
`ONE_EPOCH_ASSESSMENT001.json` og `ONE_EPOCH_PEER_REVIEW001.json` er gjeldende
resultatvurdering. RESULTSHA dcc87287713ba1fb93095cb19026ad4d2cfc08b71229de63d46152bcf79446ef.

## Målt resultat og begrensning

INITIALs wholeTRAINreferanse er eksakt samme konstante prior som FINAL sammenlignes med.
Samlet partialNLL0,653420→0,622872 (4,675%lavere), hovedsakelig aktivitet.
AktivitetsBrier/logloss bedres påM5/M15/H1, forverres påH4/D1.
Betinget OPP/NED-Brier OGlogloss er dårligere enn prior på ALLE femTF.

| TF | SideAUC TRAIN | SideBrier prior → FINAL | OPPargmax på kjenttreklassestøtte |
|---|---:|---:|---:|
| M5 |0,530017|0,249875→0,250031|85,98%|
| M15 |0,531694|0,249656→0,250639|97,62%|
| H1 |0,524835|0,248679→0,250628|99,82%|
| H4 |0,505825|0,247275→0,251822|100%|
| D1 |0,523872|0,248115→0,257607|100%|

OPPargmax er en prognose, aldri en handel. FaktiskOPP på samme treklassestøtte
var44,20/41,64/40,86/47,50/52,71%. Svak rangering og bedre aktivitet er ikke
kvalifisert retningssannsynlighet eller edge. Årsaken til OPP-skjevhet er ikke bevist;
ingen konkret aggregate-/prior-/scorefeil ble funnet i den avgrensede peerreviewen.
TRAIN er gjenbrukt utvikling, ikke senere generalisering eller realisert økonomi.

D1side har bare162distinkte kjente målbarer bak44566gjentatte prognoseceller.
Gjeldende high–low/endpoint-target er konservativ og maskebevisst; ufullstendige
små rangevinduer blir ikke FLAT og ukjent side gjettes aldri. Bare manifestbundet
ukestenging unntas. Historisk daglig/helligdagskalender og komplett makrofirst-
vintage-/revisjonsledger er fortsatt ukjent. COT21daterte klokkeunntak er rettet,
ny normalisering/baseline/mekanikk/kaldreplay/fysiskrecovery er kvalifisert og gjenbrukt.
Ingen påstand om nytte fra hvert enkelt indikatorfelt/familie eller universelt feilfri kode.

## Checkpoints og neste steg

Separat fysisk Mac-backup av ALLE16checkpoints med kvitteringer og run-/admisjonsbevis
ligger i `/Users/andrekildalbakke/GX1_BACKUP/CURRENT_FUTURE_STATE_ONE_EPOCH_20261122_001`.
ManifestSHA0d8b270c1def71f658e3c75d6edc3ce2f9c0165d8f2977f598f3aa9221098b30.
Atomisk no-replace, kildeSHA/størrelse/statbookends, fsync/F_FULLFSYNC og faktisk
MacdiskSHAreadback består;0modelldeserialisering. Tidligere49+60+21+29pakker
beholder avhengighetene. Dette er den kvalifiserte kjeden, ikke hele DATA/RUNSrestore.

Ingen ny epoke er innrømmet. Biasdiagnosen ovenfor er fullført. Radvise finallogits
er ikke lagret; én avgrenset lesende tidsdelt prognoseskåring krever egen eksakt
kvalifikasjon/plan uten optimizerupdates. Ingen automatisk taps-/klassevektendring.
Bruk uendrede beståtte kontroller; ingen jobb bare for å fylle ventetid.
TEST/CONTROL/VAL-scoring, M1-retning, GC, Entry/Exit, broker, live/paper, handel,
spending og promotion er stengt. DATA/RUNS ryddes bare via retentioneieren.
