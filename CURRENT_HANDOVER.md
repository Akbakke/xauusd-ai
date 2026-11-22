# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Prosessobservasjon og eksakte terminalkvitteringer overstyrer prosa.

## Aktiv oppgave: full dekning og varsling før bevegelse

Brukeren slettet det gamle årsaksmålet og ba22.11 om et nytt mål:
100% av kvalifisert M5-klokke, alle aktive indikatorer/tidsrammer og test av
selektive OPP/NED-varsler FØR tydelig bevegelse. Nytt mål er ACTIVE i
chat01a20035-2836-7f62-bb88-9e24603a51a5. Ingen tokenbudsjett er angitt.
Gamle årsaksresultater er bevart; D1-årsak er ikke feilaktig merket løst.

Alle fem globale modell-/treningsporter er false. Ingen ny plan/CLAIM/modell
for denne oppgaven er ennå publisert eller innrømmet. Ingen ny trening/epoke,
TEST, M1-retning, GC, Entry/Exit, broker eller handel. Klasse `trainer` skal
bare brukes som eksisterende GPU-ressursinnkapsling for en lesende kjøring.

## Dekning som er kontrollert før kjøring

MAIN brukte alle354550 kvalifiserte TRAIN-rader2021–2025. Bare de senere
lagrede detaljprognosene var et utvalg20727rader (5,85%, opptil16/24h-bøtte).
Ny capture skal lagre hver eneste TRAIN-prognose, ikke et nytt utvalg.
Native2021–25 har fem parquetdeler med til sammen354550rader; faktisk metadata
og targetklokke/headers er kontrollert uten å laste arraydata.

FEATURE_CONSUMPTION_CROSSWALK001.json binder faktiske originale konsumentkilder,
registry, provider, normalisering og modellruting. Ingen aktive felt er uten
eier:318signal,132ctx_cont,1kategori,256MTF perTF. Alle åtte familier og
M5/M15/H1/H4/D1 beholdes. 132aliaser er tilsiktet;186felt går til generell
snapshot-projeksjon. Alle61makrofelt rutes gjennom session_regime_encoder og
inngår i familiesamspillet; det finnes ikke en separat niende encoder.
Makrobindingspredikatet er komplett på389579/389579retinerte snapshotrader.
Dette beviser ikke historisk revisjons-/first-vintage-fullstendighet.

Dekningen gjelder gjeldende aktive native TA/MTF-profil. GC/orderbok er utsatt,
M1-retning er forbudt og M30 er ikke implementert i dette checkpointet.
Ingen skjult påstand om at alle mulige markedskilder allerede brukes.
Normaliseringsfilen har ikke numeriske perfelt-observedcounts; finite-counts
kan også telle nullkodet ukjent. Collector gjenbruker derfor de eksisterende
provider.readiness_report/support_report én gang hver på hele TRAIN-aksen,
med ekte current-field-missingness skilt fra finite-support og ukjent nytte.

## Låst test og kvalifisert kilde

Forhåndsvalgt modell er original MAIN FINAL, modellstatec30b16b4...47ed61,
autentisert gjennom003chronological/FINAL.pt på driver610.88. Ikke004kandidaten.
003/004 er konsumert og relanseres aldri. Kontrollens hele V3-tree og alle åtte
originale FINAL-komponenter valideres før parameterkopi. Bare modellvekter lastes;
ingen optimizer. Faktiske input-/norm-/priorbuffere må være bitlike.

Ny owner `gx1/audit/capture_native_future_state_full_train_v1.py` skal kjøre
1385E256-forwards på354550rader (sistebatch246), én modell/én CP-load,
0optimizer/backward/fit/normaliseringsfit/priorfit/VAL/TEST og0CP-writes.
Hver batchs rålogit-hasher sammenlignes straks med opprinnelig MAIN/PASS1;
første avvik stopper. Fulloutput-SHA og alle proper scores må også være bitlike.
Én fullcapture med issue/source/position-akse publiseres atomisk og diskleses.
Sourcebro beholder75gamle eiere bitlike og tillater bare to guard-eieres
navngitte innsettinger; ingen modell-/feature-/target-/loss-matematikk endres.

Ny CPU-owner `gx1/audit/audit_native_selective_moves_v1.py` skal deretter måle
alle354550issues, alle5TF og terskler0,50/0,60/0,70/0,80/0,90 separat.
Score=a*max(s,1-s), retning fra s; ingen TF-stemmer, kvote eller vinnerutvalg.
Hver native targetbar-gruppe får et klokkevalgt første issue-anker. Fra kjent
M5-close og kausal TF-ATR måles første ±1ATR-kryss fram til eksisterende
validity_ns. Dette er en observasjonstest, ikke ny treningsfasit, tapsgrense,
holdetid eller Entry-/Exit-regel. Første varsel beholdes selv om et senere er bedre.

Manglende forventede åpne M5-barer gir UNKNOWN; bare manifestbundet ukestenging
unntas. Begge barrierer i samme første M5-bar har ukjent rekkefølge. Treff før
selve targetbarens start rapporteres separat. Alle signaler/ukjente utfall og
oversette kjente bevegelser blir med. Unike targetvinduer og fysiske hitbarer
rapporteres uten å påstå uavhengige handelsepisoder.

Reversal/continuation-strata bruker bare prisretning i et komplett kjent
fortidsvindu med samme kanoniske TF-varighet ved ankeret. Varsel ved ankeret
før usett prisbane skilles fra senere varsel før1ATR-treff. Det siste er ikke
automatisk før en latent markedsvending. Progresjon/headroom synliggjør sene
varsler. Samme varslingstidspunkter brukes for alltid-OPP/NED og frossen prior.
Midpris-markout er ikke fill/PnL; kostnader og MAE/MFE er ikke målt i piloten.
TRAIN-gjenbruk beviser ikke senere generalisering eller økonomisk edge.

190 unike fokuserte cases PASS:76guard+68collector+46CPU-audit. Først45negative
guardcases ventet gammelt feilnavn og én immutabel-publiseringscase feil exception;
bare testforventninger ble rettet og bare feilcaser omkjørt. Produksjonsvern ble
ikke lempet. En statisk oppdaget CPU-vindusfeil ble rettet før første test:
anchor<=target_start<end, med to regressjoner. Faktiske input-/header-/MAIN-
referanser, native TRAIN-footere og fullplan-view/capped CPU-lås er kontrollert.
Ingen modell, checkpoint-payload eller dataarray er lastet i kvalifiseringen.

## Neste steg og vakter

Bind en eksakt ny lesende captureplan fra fryst kilde. Røykprøv den kanoniske
absolutte Python -m-kommandoen med --preflight uten CLAIM/OUTPUT/modell.
Behold alle fem globale treningsporter false også ved admisjon; åpne bare
én eksakt lesegate. Kjør så capture og deretter egen CPU-auditplan.

Én tung CURRENT-jobb, eksisterende capped/signed Windows-vakter. GPU250Wdraw/
limit,75°Ckjerne/80°Cminne/12288MiB. CPU80°C/125W og8GiB fysisk/commitreserve;
prelaunch RAM til cgroup+reserve. GPU20G/512M/8tråder/CPU0–18/tasks128,
3600souter/3570sintern. CPU-audit4G/512M/1tråd/600souter/570sintern.
Ingen endring av effektgrense eller Windows-sikkerhet. Stabile kjøringer
kontrolleres omtrent hvert15.–30.minutt; ingen minuttvise modellrunder.
Ingen kilde-/policyendring under kjøring. Alle originale checkpoints beholdes;
ny capture/resultater skal sikres og diskverifiseres på separat Mac-disk.

Bevisrot `/home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001`.
SELECTIVE_MOVE_PREREGISTRATION001.json SHA94e93090...52825c9;
FULL_COVERAGE_REVERSAL_SCOPE001.json SHA005422b0...488d50;
FEATURE_CONSUMPTION_CROSSWALK001.json SHAbc88ae03...85f876;
SOURCE_QUALIFICATION001.json binder alle testlogger og faktiske metadata-prober.

Tidligere fullførte004resultat og tre checkpoints er Mac-sikret i
CURRENT_BIAS_SUFFIX004_COMPLETE_20261122_001, manifest2b7e08b1...24918a.
003s ni durableCP finnes i CURRENT_BIAS_SUFFIX003_HOST_MAINTENANCE_STOP_20261122_001,
manifest e3e4e0c1...0d6d3e0. Alle16MAINCP og frossen16CP-capture er også bevart.
Ingen DATA/RUNS-rydding. Øvrig historikk ligger i Git og immutable kvitteringer.
