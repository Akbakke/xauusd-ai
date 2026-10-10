# CURRENT HANDOVER — 10.10.2026

Ny bestilling: bygg de berørte dataene på nytt, kontroller dem og mål en fersk
initialbaseline. Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; prosess og terminal vinner over prosa.

## Gjort

Featureaudit og forrige firestegsstudie er avsluttet på 569d0054. Ingen kvalifisert
Entry-edge: CONTROL4096 ga 4095 FLAT og én SHORT; forbedringen mot konstanten var
ikke påvist. Alle 254 lokale felt, 71 aliaser, ett sesjonsfelt og 190 MTF-felt er
individuelt revidert. Fire funksjonsfeil og misvisende metadata er rettet;
141 fokuserte tester og Git-kontrakttestene består. Detaljer finnes i
[docs/FEATURE_AUDIT.md](docs/FEATURE_AUDIT.md) og feltregisteret ved siden av.

Brukerens nye bestilling åpner en egen bygge-/baselineplan:
[configs/research/FEATURE_REPAIR_REBUILD_20261010.json](configs/research/FEATURE_REPAIR_REBUILD_20261010.json).
Ny dataset_run_id: HISTORY2009W_FEATURE_REPAIR_20261010.
Kjørebevis: /home/andre2/GX1_RUNS/FEATURE_REPAIR_REBUILD_20261010_001.
Dette er ikke relansering av en konsumert plan. Gamle bevis beholdes.

## Aktiv grense og neste steg

Kjernebygget fullførte 17:16 UTC med exit 0 etter 22130 sekunder på uendret
550b4116. M5/MTF/M1, datasett og labels er publisert; dette er ennå ikke komplett
inputaksept. Trygg fysisk omstart er bekreftet ved ny Windows-/WSL-boot.
Tre ferske kontrollfiler manglet fsync og var tomme etter omstart; tomme originaler
er bevart, observasjonene er gjenopprettet fra Mac og publisert atomisk.
Core-kvitteringene og kildeidentiteten er separat verifisert.

COMPLETE_M1_001 fullførte 19:34 UTC med exit 0 på uendret 1b31c164:
5 523 147 komplette pre-TEST M1-rader. Klokke- og bytekontrollen besto.
Det korrigerte datasettet har 652 552 TRAIN-rader og 70 880 VAL-rader.
INPUT_VERIFICATION_001 fullførte 19:50 UTC med exit 0 på uendret 73484ee5.
Alle 5 523 147 M1-rader og 392 143 437 aliasverdipar er kontrollert; ingen manglende
TRAIN-/VAL-quote-/fill-rader. Ny populasjonsdiagnose og canonical readiness består.
Uendrede råresponsbevis er gjenbrukt etter byte- og parserkontroll; nye
featurefordelinger er målt. Ukjente kildegap er ikke forklart som markedslukking.
217 fokuserte overleveringstester består etter retting av foreldede statuskrav.

INPUT_VIEWS_001 fullførte 20:07 UTC med exit 0 på uendret 64964ed5.
Fysiske TRAIN-/CONTROL-visninger dekker henholdsvis 4 884 638 og 382 744 M1-rader;
652 552/70 880 Entry-rader er bevart. Nye egne klokkebundne gapautoriteter er
publisert. POSTBUILD_REVIEW_001 verifiserte alle 652 552/70 880 lagrede
seq/snap-rader mot korrigert M5 med PASS. Deretter ble spesialistkontrollen drept
ved 4 GB cgroup-grensen; kernelbevis bekrefter OOM, exit 137. Kilden var uendret,
alle originaler og begge beståtte sekvenskvitteringer er bevart.
POSTBUILD_SPECIALIST_001 fullførte 20:40 UTC med exit 0 på uendret 1196222a;
spesialistaudit PASS gjennom eksisterende producer10G/512M-eier. Ferdige nye
sekvensbevis er gjenbrukt. Ingen feature-/modellendring eller budsjettfornyelse.

BASE_NORMALIZATION_001 binder eksisterende producer20G/512M-eier for én fersk
normalisering på hele fysiske TRAIN, alle 652 552 Entry-rader før sampling.
VAL-/TEST-fit er null. Deretter følger TRAIN-summary-normalisering, faktisk
normalisert inputparitet og fersk nullstegs ONLINE/TARGET. Kvitteringene avgjør
framdriften; ingen ny normalisering eller baseline er akseptert før de består.

Gamle featuredata, normalisering og vekter er ikke korrigert av kildeendringene.
Ingen nye inputs eller ny baseline er akseptert ennå. Rådata, perioder og de
uendrede kalibreringene gjenbrukes der de eksisterende eierne tillater det.
Gamle ukjente kildegap forblir ukjente; ingen imputering eller periodeflytting.

## Bevar

Én agent og én tung CURRENT-jobb gjennom eksisterende capped-eier og vakter.
Ingen kildeendring under kjøring. Alle genuine features, åtte familier og native
klokker består. Ingen fast tapsgrense eller maksimal holdetid.
training_enabled=false; null optimizersteg i dette målet. TEST er forseglet;
kun eksisterende konstruksjons-/forseglingsansvarlig kan opprette det nye splitet.
Ingen TEST-analyse, live/paper, broker, ordre, spending, promotion eller opprydding.
Stabile langkjøringer observeres omtrent hver time; ingen minuttvis modellpolling.
