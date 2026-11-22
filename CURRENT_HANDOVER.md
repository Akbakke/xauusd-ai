# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

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

Ingen ny epoke er innrømmet. Anbefalt neste bevis er å diagnostisere lagret
retningsskjevhet/kalibrering og kronologi før en ny fit. Radvise finallogits er ikke
lagret; detaljert oppdeling eller senere frossen evaluering må få egen eksakt
kvalifikasjon/plan uten optimizerupdates. Ingen automatisk taps-/klassevektendring.
Bruk uendrede beståtte kontroller; ingen jobb bare for å fylle ventetid.
TEST/CONTROL/VAL-scoring, M1-retning, GC, Entry/Exit, broker, live/paper, handel,
spending og promotion er stengt. DATA/RUNS ryddes bare via retentioneieren.
