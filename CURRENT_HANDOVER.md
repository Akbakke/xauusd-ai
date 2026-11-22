# CURRENT_HANDOVER — GX1

Arbeidssted `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md helt. NEXT_RUN_POLICY.json er eneste arbeidsstatus.
Prosessobservasjon og eksakte terminalkvitteringer overstyrer prosa.

## Aktiv oppgave: full dekning og varsling før bevegelse

Brukeren ba22.11 om nytt mål: full kvalifisert M5-klokke, samtlige aktive
indikatorer/tidsrammer og selektive OPP/NED-varsler før tydelig bevegelse.
Målet er ACTIVE i chat01a20035-2836-7f62-bb88-9e24603a51a5. Ingen tokenbudsjett.
Alle fem globale treningsporter er false. Ingen refit/ny epoke/VAL/TEST,
M1-retning, GC, Entry/Exit, broker, handel eller spending er autorisert her.

## Full prognosedekning er faktisk gjennomført

NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001 er COMPLETE og konsumert.
354550/354550 TRAIN-tidspunkter2021–25,5TF,1385E256-forwards (siste246),
én modell/én CP-load; null backward/optimizer/fit/CP-writes. Alle batchhashes,
fulloutput og proper scores er bitlike opprinnelig MAIN/PASS1. Faktiske
normaliserings-/input-/priorbuffere identiske. Ikke004kandidat: original MAIN
FINAL modelstatec30b16b4...47ed61 via autentisert003chronological/FINAL.pt.
Plan638c32e0...54717f; fryst kildeccaa7db2;80eiere/source/input/policy/boot-
bookends passerte. RESULTce287ed6...68e89a; TERMINALb47b4a90...21115f6;
CAPTURE.npz5349c1d6...9bc98b. ClaimPID60233 er borte; wrapperexit0 og lesegate
lukket med0gjenstående. Kjøringen tok839,64s. Ingen relansering av brukt plan.

Capture, kvitteringer, logger og controller er separat Mac-diskverifisert:
CURRENT_FULL_TRAIN_CAPTURE_COMPLETE_20261122_001/MANIFEST.json
SHA812be0eb...ccb7b94 (12artefakter/26383392bytes). Forhåndskontrollens25filer
finnes i CURRENT_FULL_TRAIN_CAPTURE_PREFLIGHT_20261122_001,
manifest176c2f60...fb6d4f6. Ingen nye vekter ble laget. Alle opprinnelige
MAIN/003/004-checkpoints og tidligere capture er bevart i eksisterende backups.

Providerens eksisterende readiness_report/support_report ble kalt én gang hver
på alle354550rader: alle input_ready,0unmaterialized,0feil i erklært historie/
MTF/readiness, current_field_missingness tom. Support-observedcounts er bare
np.isfinite og er ikke en total faktisk observasjons- eller nyttegaranti.

## Hva indikator- og tidsrammedekningen betyr

FEATURE_CONSUMPTION_CROSSWALK001 binder faktiske originale konsumentkilder,
registry, provider og normalisering. Ingen aktive felt uten eier eller med
utilsiktet flere eiere:318signal,132ctx_cont,1kategori,256MTF perTF.
Alle8familier og M5/M15/H1/H4/D1 beholdes. 132signal/ctx-aliaser er tilsiktet;
186felt går til generell snapshot-projeksjon. Alle61makrofelt rutes gjennom
session_regime_encoder; ingen separat niende encoder. Makrobindningen er
komplett på389579/389579retinerte snapshotrader, men dette beviser ikke
historisk first-vintage/revisjonsfullstendighet eller lært nytte per felt.
GC/orderbok er utsatt, M1-retning forbudt, M30 ikke implementert i checkpointet.
100% gjelder aktiv kvalifisert profil, aldri alle mulige kilder i verden.
MAIN hadde allerede100%TRAIN-fit og aggregatscore; bare senere detaljcapture
var20727rader/5,85%. Det observerte loggingshullet er nå lukket, uten ny fit.

## CPU-prisforløpstesten gjenstår

Forhåndslåst SELECTIVE_MOVE_PREREGISTRATION001, FULL_COVERAGE_REVERSAL_SCOPE001
og UP_DOWN_REPORT_PARTITION001 finnes i bevisroten nedenfor. Før ny CPU-plan
innrømmes skal minimal report-only OPP/NED-partisjon føyes til etter full GPU-
stopp; ingen endring av hendelser, score, terskler, signalvalg eller modell.

CPU-owner audit_native_selective_moves_v1 bruker fullcapture og bare fem native
TRAIN-prisår. Hver av5TF/terskler0,50/0,60/0,70/0,80/0,90 rapporteres uten
vinnerutvalg. Score=a*max(s,1-s), side fra s. Klokkevalgt første issue per
native targetgruppe er anker. Første ±1TFATR14-kryss fra kjent M5-close fram
til kanonisk validity_ns er observasjon, aldri ny treningsfasit/stop/holdregel.
Første kvalifiserende varsel beholdes selv om et senere varsel er bedre.

Manglende forventede åpne M5-barer gir UNKNOWN; bare manifestbundet ukestenging
unntas. Begge barrierer i samme M5-bar har ukjent rekkefølge. Alle grupper,
ukjente, falske/sene varsler og oversette bevegelser telles. Alltid-OPP/NED og
frossen prior sammenlignes ved samme varslingstidspunkter. OPP/NED og begge
vendingsretninger splittes eksplisitt. Reversal mot fortidsretning måles ved
anker. Varsel ved anker skilles fra senere varsel før1ATR-treff; ingen av dem
beviser før latent fysisk vendingsøyeblikk. Headroom/progresjon synliggjør dette.
TRAIN-vekter/norm/prior brukte hele2021–25: dette er en historisk TRAIN-diagnose,
ikke et ærlig out-of-sample2021-modellforløp eller bevist fremtidig edge.
M5 første treff er et femminutters intervall. Overlappende targetvinduer er
ikke uavhengige episoder. Midpris-markout er ikke fills/PnL; kostnader og
MAE/MFE er ikke målt. Opprinnelig sidehode er ikke kalibrert1ATR-eventmodell.

## Kvalifisering og neste handling

190unike fokuserte cases og obligatoriske3Entry-kontraktfiler/64cases PASS før
GPU-planbinding. Testforventninger ble rettet for45negative guardcases og én
immutabel publisherexception; bare feilcases omkjørt, ingen vern ble svekket.
Minimal CPU-rapportpartisjon skal verifiseres med egne nye fokuserte cases.
Commit fryst CPU-kilde, bind så egen nøyaktig CPU-plan og CLI-preflight uten
CLAIM/output/arraydecode, åpne kun én lesegate, kjør og lukk etter wrapperexit.

Én tung CURRENT-jobb via capped. GPU20G/512M/8tråder/CPU0–18/tasks128 og
3600souter/3570sintern; GPU250Wdraw/limit,75°Ckjerne/80°Cminne/12288MiB.
CPU80°C/125W;8GiB fysisk/commitreserve og prelaunch cgroup+reserve. Signed
Windows-vakter må være tilgjengelige. CPU-audit4G/512M/1tråd/600souter/570sintern.
Ingen kilde/policyendring under fryst kjøring. Stabilt tilsyn15–30min, ingen
minuttpolling/fyllarbeid. Ingen maskinvaregrenser/Windows-sikkerhet endres.
Bevisrot /home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001.
Ingen DATA/RUNS-rydding. Historikken er i Git og immutable kvitteringer.
