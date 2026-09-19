# Samlet gradient og Adam-retning — én avgrenset diagnose

Entry fuse256 forbedret verdiestimater, men alle Entry/Exit-handlinger er uendret.
Saved-state-audit bekrefter726 optimizerparametertensorer på steg256, Entry/Exit-
tapsvekter0,9753/0,9759 og endring i alle fuse-/Entry-hodeelementer. Ingen frosset
Entry eller avslått tapsvekt er dokumentert. Parameterbevegelse beviser ikke
nyttig retningslæring. Historisk routingmåler omfattet bare fire gateparametre.

Eksisterende native diagnose er utvidet med final_joint_update. Ingen modell,
treningsmatematikk eller handelsregel er endret. Samme verifiserte TRAIN16-cache,
originale targets/masker og frosne lærer. Fem forwards: ONLINE Entry inferens og
gradientmodus, original lærer Entry, native ONLINE Exit og lærer Exit. Diagnose
bruker eksisterende Exit-backward og nøyaktige Entry-/hjelpetap. Ingen optimizersteg.

Alle akkumulerte parametergradienter kontrolleres mot separat Entry/Exit/hjelpe-
beregning. Rapporten skiller Entry fellesverdi, LONG−SHORT-kontrast og FLAT,
fuse, Entry-hode, Exit-parametre og alle prediksjonsparametre. Lagret Adam-
historikk, opprinnelig klipping og weight decay brukes til en FP64 beregning av
mulig neste oppdatering. Beregnet tapseffekt er første orden, ikke utført trening.

En fast kontrast utelater bare nåværende hjelpegradient. Historiske momenter
beholdes, så dette isolerer ikke hele tidligere hjelpepåvirkning. Målingen er
på én gjenbrukt eval-batch uten dropout; den gjenskaper ikke historiske steg.
Ingen målvarians antas predikerbar, og intet resultat er profitt/generaliseringsbevis.

Seks fokuserte tester bestod, inkludert sammenligning mot virkelig native AdamW
med historikk og klipping, delt Exit-gradient, uendret modell og opprydding ved
feil. To rapporteringstester bestod etter rettelse av en konkret gammel peker:
handover viste hovedencoder som siste fullførte kjøring etter fuse256.
Inputaudit bekreftet eksakt batch/rad/target/mask-paritet og ny ONLINE-funksjon.

Plan: NATIVE_JOINT_UPDATE_DIAGNOSTIC_20260919/PLAN.json under vanlig BASE.
Én invokasjon, null optimizer, ingen CONTROL/VAL/TEST eller automatisk utvidelse.
Kilde fryses under kjøring. Gjenbruk tester og audits; vurder resultatet før kodeendring.

## Bekreftet feil og minste rettelse

Første kjøring feilet2026-09-19T14:08:16Z i cuDNN GRU-backward fordi hele
modellen var eval. Ingen diagnose/resultat; opprinnelig checkpoint/pointer uendret.
Task deaktivert med last_result1. Se handover_snapshot/JOINT_UPDATE_FAILURE_20260919.json.

V2 legger til én canonical ONLINE Exit-inferens. Bare dropoutfrie GRU-er får
training=True under gradientforward; resten av modellen forblir eval. Q må
stemme innen samme1e-4 og valgene være identiske. Flag/method gjenopprettes også
ved feil. Seks forwards, null optimizer. Fem fokuserte tester med ekte GRU
bestod, inkludert forwardfeil, paritetsbrudd og targetavvik. Tidligere AdamW-
tester gjenbrukes. Native CUDA-bevis gjenstår; ingen modell/trener ble endret.
Ny separat binding: NATIVE_JOINT_UPDATE_GRU_RETRY_20260919; gammel plan er brukt.

## Uavhengig gjennomgang av handelsfordel

Tre read-only underagenter undersøkte targets, signal og samlet økonomi etter
brukerens eksplisitte bestilling. De fant ingen dokumentert overførbar fordel.
Fuse256 lærer fellesnivået nær target: bias0,059 Bps; konstant optimal handling
på målmiddelverdiene er FLAT. Kontrastkorrelasjon0,248 og innenmåned0,238 viser
TRAIN-assosiasjon, men proben overlapper treningen. Ren middelfeilrettelse ville
fortsatt ikke gitt positiv LONG/SHORT mot FLAT. Targetbootstrap er bare0,045/0,040
Bps ved ankrene mot observerte komponenter17,369/17,373; svak lærer alene er
ikke en dokumentert forklaring på manglende amplitude.

Fast diagnose på samtlige lagrede TRAIN256-rader, uten fit/forward: velg høyest
LONG/SHORT uten FLAT ->234 LONG/22 SHORT, referanseverdi−1,4155 Bps mot konstant
LONG−3,7500, SHORT−7,9601 og FLAT0. Rangeringen hjelper relativt, men støtter
ikke tvungne handler. Kilde: fullført fuse TRAIN_OBSERVATION, sha908ad63a049cab00c668226f6cfd1a0ed8f00035f05efdbd4775ae31549a23b5.
Dette er Q_mu på gjenbrukt TRAIN, ikke realisert økonomi eller holdout.

Eldre forecast120 viste TRAIN+6,13 Bps, men sviktet allerede på juni-VAL:
korrelasjon0,0209, retning42,97%, MSE3225,05 mot null3081,70. Gjenbruk avslaget;
ingen ny direkte kobling er begrunnet. Se TRAIN_VAL_SEMANTICS_AND_TRANSFER_20260917.
Fulløkonomi/åpne posisjoner og én-posisjonsreplay finnes allerede; bygg ikke ny
motor. Q_mu er fast referansepolicy, ikke botens egen greedy Exit-verdi. Dette
må skilles fra profitt, men er ikke alene grunnlag for targetendring.

Falsifiserbar hypotese: samlet oppdatering motvirker nyttig Entry-kontrast eller
Exit-læring. Den bundne diagnosen avgjør retning/størrelse på denne batchen.
Viser den allerede forbedring av begge, forkast konfliktforklaringen i dette
måleomfanget. Ingen ny normalisering eller tapsvektsendring uten målt årsak.
