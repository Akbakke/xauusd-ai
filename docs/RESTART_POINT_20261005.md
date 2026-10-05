# Eksakt restartpunkt - 05.10.2026

Dette er den korte menneskelesbare inngangen til gjeldende arbeid.
CURRENT_RESTART_POINT.json er den maskinlesbare tvillingen.
Historiske dokumenter og COMPLETED_RUN.json er bevis, aldri startordre.

Operatørens nye vedtak 05.10: gjennomfør punkt 1–6 i
docs/NATIVE_V38_EXECUTION_20261005.md. SAMPLER_BENCHMARK_001 er
forhåndsregistrert; følg den bundne planen/operatoren i NEXT_RUN_POLICY.json.
Deretter fryste koordinater, fersk initialbaseline og én256-stegs prøve.
Et større endelig budsjett er bare autorisert betinget av bestått læringsport.

## Authority og første kontroll

Bruk bare /home/andre2/src/GX1_CURRENT på work/gx1-current.
Mac-mappen er en overleveringskopi. Ved denne kontrollen var HEAD før
dokumentoppdateringen 90ca4ac4e3a83174d35500c921e82c48c2bbe657, arbeidsstreet rent, ingen native
GX1-prosess kjørte og ingen GPU-prosess var registrert.

~~~bash
cd /home/andre2/src/GX1_CURRENT
git branch --show-current
git log -5 --oneline
git status --short
bash scripts/gx1_handover.sh --check
~~~

Forvent work/gx1-current. Stopp hvis prosess, lås, kildeidentitet eller
kvittering avviker fra CURRENT_RESTART_POINT.json.

## Målet

Målet er en fullstendig automatisk XAUUSD-bot basert på én komplett kausal
før-TEST-M1-kilde. M5 er Entry-klokke, M1 er Exit-klokke og M15/H1/H4/D1 er
lukket MTF-kontekst. Modellen skal bevare alle 254 features, åtte familier
og kausalitet.

Før boten kan kalles kvalifisert må den vise selektive handelsbeslutninger,
senere kronologisk generalisering, positiv kostnadsjustert økonomi for alle
valgte og åpne posisjoner, funksjonell train/serve-paritet, sikker restart
og brokeravstemming. Forseglet TEST brukes først etter fryst modellvalg.
Live/paper, ordre og spending er stengt.

## Verifisert fremdrift

- Komplett før-TEST-M1 er kvalifisert. Alle 1 215 514 M5-barer er
  rekonstruert eksakt fra M1 i de 13 markedsfeltene.
- Native v38 sweep/AVWAP er implementert i eksisterende featurekjede og
  kvalifisert på full inputdekning. Dette er inputbevis, ikke edge.
- TRAIN/VAL-klargjøring, normalisering, prospektiv kostnadspolicy og
  økonomisk random-access-indeks er publisert og kontrollert.
- OANDAs observerte finansieringsvilkår er bundet. Godkjent to-GET-lesing
  er brukt opp.
- Offline lagring av handelstilstand er rettet og feilinjeksjonstestet.
  Full ordrekoordinering og brokeravstemming gjenstår.
- Makrokjernen med DFII10, DTWEXBGS og T10YIE er ferdig målt.
  Utfallet var INKONKLUSIVT; makro er ikke promotert.
- Full B med seks kilder er separat og blokkert på GLD/COT-versjonshistorikk
  og VIX-tilgjengelighetsklokke.

## Rettet dokumentmismatch

Statusfilene pekte feilaktig på INDEX_FEATURE_SOURCE_REVIEW_001 som neste
jobb. Runtime viser at jobben ble fullført 02.10 med exit 0, uendret kilde,
null TEST, null modellforwards, null optimizersteg og null normaliseringsfit.

TRAIN har 652 552 Entry-rader og 4 884 638 M1 child-rader.
VAL har 70 880 Entry-rader og 382 744 M1 child-rader.
Begge har 254 felt og eksakt feature-/foreldreklokke. Ingen sampler ble valgt.

Resultat: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/EVENTS/RESULT_20261002T062117624344Z.json
SHA256: f7fbe63ad88230c6a7da65d76384f60276c17ad12895bd3ce00c5b4388cbf23b

Terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/EVENTS/TERMINAL_20261002T062117642659Z.json
SHA256: 2ffd811ee77d821dda0324ccca1889e4086bc004195663d75a10aa8803758fac

Planen er konsumert og skal ikke relanseres.

## Eksakt neste arbeid

Les benchmark_unified_exit_random_access_train_v1.py og den eksisterende
SAMPLER_BENCHMARK_CANDIDATES.json. Forhåndsregistrer deretter nøyaktig én
workload-matchet samplerbenchmark mot den faktiske indeksbundne 254-felts
TRAIN-kilden.

Gjenbruk:
- gx1/scripts/benchmark_unified_exit_random_access_train_v1.py
- /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/FINAL_BINDINGS_V1/SAMPLER_BENCHMARK_CANDIDATES.json
- /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/RANDOM_ACCESS_INDEX_V1/ROOT.json
- featurekildekvitteringen over

Kjør bare gjennom capped producer etter kontroll av planhash, kildeidentitet,
ledig prosjektlås, ledig maskin og ren Git-kilde. Ingen refit, optimizer,
TEST, broker, live/paper eller spending inngår.

## Rekkefølgen etter benchmarken

1. Velg sampler bare fra det forhåndsregistrerte resultatet.
2. Publiser immutable epoch0-, first4096-, TRAIN256- og CONTROL256-koordinater.
3. Kjør fersk nullstegs initialmåling for aktuell ONLINE/TARGET-funksjon.
4. Åpne høyst én avgrenset native v38-sammenligning hvis læringsporten tillater det.
5. Mål senere kronologisk generalisering og all kostnadsjustert økonomi.
6. Etabler ny funksjonell train/serve-paritet.
7. Kvalifiser ordreintensjon, idempotent restart, avstemming og recovery offline.
8. Frys modellvalg før eventuell tilgang til forseglet TEST.

## Status for edge

Datagrunnlaget, kostnadsbindingene og driftskontrollene er vesentlig bedre.
Generaliserende edge, positiv nettoøkonomi og ferdig automatisk bot er fortsatt
ubevist. Benchmarken reduserer kjørefeil og ressursrisiko; senere matchet læring
og kronologisk generalisering avgjør om vi faktisk har edge.
