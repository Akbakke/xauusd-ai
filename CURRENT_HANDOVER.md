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

## CPU-prisforløpstesten er fullført; prediksjonsmålet er ikke nådd

NATIVE_SELECTIVE_MOVE_AUDIT_20261122_001 er COMPLETE på22,10s, kilde9f4f04b4a.
Eksakt CPU-plan3c4509d5...3b8144; RESULT1b1e30e1...bff6a6;
TERMINAL5db61feb...2d758. Fullcapture-, target- og native-prisklokke er bitlike
på354550rader. Alle5TF/25forhåndslåste terskelceller er rapportert, også nullvarsler,
falske/sene varsler, oversette bevegelser, UNKNOWN og begge vendingsretninger.
Nullmodell/forward/fit/checkpointload/TEST. Wrapperexit0; ingen kanoniske
leafprosesser igjen; CPU-lesegaten er lukket/konsumert med0gjenstående.

Modellen velger OPP på86,269%M5,98,357%M15 og100%H1/H4/D1. Ved terskel0,50:
M5 fanger25084/42527kjente OPP-vinduer, men bare52/43747NED;
M15 fanger16197/23160OPP og1/23546NED; H1/H4/D1 fanger0NED.
Dette er targetvinduer, ikke uavhengige markedsepisoder eller handler.
Ingen av25celler slår alltid-OPP ved identiske modellvarslingstidspunkter:
24like (inkl13uten varsler), M5/0,50 er16timelytreff svakere (25136mot25152).
Høyere scoreterskel løser ikke ensidig retning; intet vinnerutvalg er adoptert.
Sammenligningen avgrenser retningsvalg ved egne varslingstidspunkter; den
avviser ikke all aktivitets-/tidsinformasjon. M5s kjente bevegelsesandel er
24,33%allegrupper,32,17%valgt0,50 og43,18%valgt0,60. Kun beskrivende TRAIN,
ikke terskelutvalg eller edge. Event-/reversal-direction presisjonsfelt er
betinget på faktisk fremtidig retning; bruk dem som recall/support, aldri
prospektiv varselpresisjon. DIRECTION_SELECTION_INTERPRETATION001 binder dette.
CPU_SELECTIVE_REVIEW001 (7e0188b2...20ca) kontrollerer125modell/baseline-summaries.

D1 har897/1290UNKNOWN-hendelsesvinduer (69,53%) fordi forventede åpne barer
mangler før treff eller i uavsluttet no-hit-vindu. De er ikke fylt/droppet.
D1-bevisene er derfor begrenset; D1-kontekst beholdes. Det er ikke dokumentert
nytte for hvert indikatorfelt eller at lang context nødvendigvis har lært edge.

Testen målte første ±1TFATR14-kryss fra klokkevalgt anker/M5-close til eksisterende
validity_ns, kausal ATR, score=a*max(s,1-s), første varsel per targetgruppe.
Alle5TF og terskler0,50/0,60/0,70/0,80/0,90 ble låst før nye prognoser/prisutfall.
Opp-/nedreversal måles mot kjent fortidsretning ved anker. Varsel ved anker
skilles fra senere varsel før1ATR-treff; ingen av dem beviser før latent fysisk
vendingsstart. Lead0minutter deler den ukjente femminutters treffbaren.
Bare manifestbundet ukestenging unntas; same-bar toveistreff har ukjent rekkefølge.
Overlappende vinduer og TF skal ikke summeres til uavhengige episoder.

Vekter/normalisering/prior brukte heleTRAIN2021–25. Dette er in-sample historisk
diagnose, ingen ærlig out-of-sample2021modell eller bevist fremtidig edge.
Opprinnelig endpoint-side er ikke en kalibrert first-cross/reversal-sannsynlighet.
Målmismatch er en konkret designrisiko, ikke bevist eneste årsak til bias.
Midpris-markout er ikke fills/PnL; kostnader og MAE/MFE er ikke målt.

FULL_COVERAGE_SELECTIVE_ASSESSMENT001.json SHA62e42c8b...952de5 binder alle25celler,
klokke-/inputdekning, baselinegap og ærlige grenser. Deknings-/testmålet kan
lukkes med negativt prediksjonsresultat; prosjektets hovedmål om før-varsling og
økonomisk edge er fortsatt IKKE oppnådd. Ingen ny epoke/automatisk fortsettelse.
Neste designprioritet er et eksplisitt kausalt mål for selektiv opp/nedbevegelse/
vending, inklusive muligheten til å avstå, og senere kronologisk utviklings-
validering. Behold indikatorene/TF; ikke tving50/50 eller velg vinnende terskel.

## Kvalifisering og neste handling

190unike fokuserte cases og obligatoriske3Entry-kontraktfiler/64cases PASS før
GPU-planbinding. Kun nødvendige testforventninger ble rettet og bare feilcases
omkjørt; ingen vern svekket. Den forhåndslåste minimale CPU-rapportpartisjonen
fikk3nye fokuserte PASS-cases; tidligere46ble ikke omkjørt. AST/diff--check PASS.

GPU-guard:535målinger, topper65°Ckjerne/72°Cminne/215,3W/1924MiB;
CPU44°C/27,52W. Minste Windows fysisk ledig55397228544B, commitreserve58051346432B.
Ingen guardstopp, alle registrerte maintenance/CBS/WU false. Eksakt metadatareview
FULL_CAPTURE_HOST_REVIEW001.json SHA71289b52...056e20. Den egne guardkvitteringen
lagrer ikke boot-ID; modellrun-kildens separate boot-bookend passerte før RESULT.

CPU-resultat, alle vurderinger/reviews, plan, snapshots, kode og logger er
Mac-diskverifisert: CURRENT_FULL_COVERAGE_SELECTIVE_TEST_COMPLETE_20261122_001,
manifest686a18fe...a52d74 (21artefakter/27243365bytes). Capture er separat sikret.
Sluttsyntaks/scoped stale-path/diffkontroll PASS; handover-observasjon før
metadatacommit viste kun forventet dirty metadata (policy/overlevering/CPU-plan).
Avsluttende ren kildeidentitet og push bookendes etter denne metadatacommitten.
Ingen videre modellkjøring er innrømmet; deknings-/testarbeidet er ferdig.

Én tung CURRENT-jobb via capped. GPU20G/512M/8tråder/CPU0–18/tasks128 og
3600souter/3570sintern; GPU250Wdraw/limit,75°Ckjerne/80°Cminne/12288MiB.
CPU80°C/125W;8GiB fysisk/commitreserve og prelaunch cgroup+reserve. Signed
Windows-vakter må være tilgjengelige. CPU-audit4G/512M/1tråd/600souter/570sintern.
Ingen kilde/policyendring under fryst kjøring. Stabilt tilsyn15–30min, ingen
minuttpolling/fyllarbeid. Ingen maskinvaregrenser/Windows-sikkerhet endres.
Bevisrot /home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001.
Ingen DATA/RUNS-rydding. Historikken er i Git og immutable kvitteringer.
