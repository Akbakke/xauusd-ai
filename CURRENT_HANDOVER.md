# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_SHORT_HEAD_PARITY_BLOCKER.
Målet fra11.11 er aktivt: finn målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle engangsplaner er konsumert, også den nye lesende observasjonen.
Training/native_training/native_launch/
full_epoch er false; ingen aktiv modelljobb. Begge c171-Windowsoppgaver er Disabled.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion er stengt.
Alle genuine indikatorer og M5/M15/H1/H4/D1-input beholdes.

## Hva indikator- og sammensettingskontrollene faktisk viste

Alle åtte TA-familier og fem tidsrammer når modellkonsum. Numerisk dekning og
ikke-konstante TRAIN-støtter er bekreftet; det beviser ikke indikatorenes nytte.
Alle11 temporal-/akse-encodere og228 undersøkte parametertensorer hadde faktisk
vektendring og Adam-andremomenter. En generell frakoblet læringsvei er avkreftet.
Mean+last kansellerer ikke systematisk. Målte variasjonsfall i blandingen er ikke
påvist prediktiv skade; fast MTF-intervensjon ga ingen bedre rangering.
Nærmeste trendlinjeslots mangler entydig samme-linje-identitet/ankere/endringsmarkør;
prediktiv kostnad er UNKNOWN. Ingen spekulativ modell-/featureendring er utført.

Låst head-/sannsynlighetsrettelse ga tidligere liten Brier/log-forbedring, med svak
rangering. [Tidligere native evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).
INITIAL i den matchede studien var et warm-start. Den eldre4096-raders kontrasten
var INTERLEAVED minus CONTROL, ikke INITIAL mot dagens R1. Lokal gradientmåling
viste konflikt mellom gjennomsnittsgradienter, men positiv gjennomsnittlig batch-dot
for begge M5-oppgaver. Dette begrunner ingen generell tapsvektendring.

## Nyeste faktiske læringskontroll

MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002 er COMPLETE/konsumert.
[Samlet kontroll og tolkning](docs/TA_M5_C171_SHORT_LEARNING_REVIEW.json).
RESULT1b25f095, kampanjeterminal974a79f7, kilde9de7e20b uendret ved terminal.
Fysisk Windows-boot640; guard og observer exit0.513,87sekunder i worker,
365,99sekunder i fit.256 genuine optimizersteg over første4096 kronologiske
TRAIN-rader, ingen hopp;288forwards og256backward. Samme c171-initial, uendret
originaltap, alle aktive parametere og fersk tom AdamW; ikke en fortsatt epoch.
Fersk full initial restore og uendret måletilstand er verifisert. Sluttilstand
5f54bbe4 beholdes kun som evidens; ingen kandidatvalg eller resume-autoritet.

Samme16 intakte B16-grupper målt før/etter.3grupper/48rader fikk nye steg;
13grupper/208rader er andre historisk brukte TRAIN-rader, ikke heldout.
Hele panelet: totaltap1,875017→1,892264; S/M5-side Brier0,246301→0,243797;
F/M5-side Brier0,251730→0,254568. Log-tap har samme fortegn som Brier.
I de3 eksponerte målte gruppene ble begge M5-sidetap og totaltap dårligere;
støtten er bare19 kjente S-sider og23 F-sider, ikke full4096-raders fit-evaluering.

På kjent retningsstøtte økte gjennomsnittlig opp-sannsynlighet med2,51/2,19
prosentpoeng for S/F. Pooled AUC flyttet bare0,53575→0,54121 /0,46939→0,47681.
S-AUC falt svakt innen begge3/13-delpaneler; pooled økning inkluderer krysspar.
Middelkomponenten utgjør86,2/71,5% av kvadrert sannsynlighetsendring på kjent
støtte. Dette er aritmetikk, ikke årsaksandel eller forklart tapsendring.
Uavhengig Brier/log-regnskap matcher originalene med maksimalt avvik0.
MATCHED_REVIEW f147b2dd; INDEPENDENT_RANK_REVIEW c414f743.
Ingen senere evaluering eller edge/økonomi påvist. Panelet er lite og avhengig.

## Konkret feil som ble rettet

001 ble kampanje-CLAIMED, men cap avviste3600sekunder før modellstart.
Den er konsumert; originale feilspor/ACTIVE_INVOCATION er bevart, ingen modell-
terminal konstruert. [Feilgjennomgang](docs/TA_M5_C171_SHORT_LEARNING_STARTUP_REVIEW.json).
Minste rettelse tillater bare eksakt002-oppskrift:3600sekunder vindu og3900
sekunder ytre guard.001 avvises fortsatt.38 berørte fokuserte tester og faktisk
materialisert vindusgate bestod; tidligere187-testers kvalifisering er gjenbrukt.
Native-vakter uendret:20GiB host/512MiB swap,12288MiB GPU-minne,250W konfigurert,
310W draw,85C kjerne,80C minne. Observerte topper:129,71W/53C/62C/2172MiB.

## Ny avgrensning av nivåskiftet og neste steg

[Trace og planlagt hodekontroll](docs/TA_C171_SHORT_HEAD_ATTRIBUTION_REVIEW.json).
Retained TRACE_ALGEBRA_REVIEW09d5e8dc: originale gradient-/AdamW-formler
matcher256 faktiske hodeoppdateringer. Direkte biasendring er negativ for S/F;
positivt felles logitnivå kommer derfor fra hodevekter og/eller representasjoner.
Summert lokal hodeoppdatering på hvert stegs faste input er gunstig for begge;
ingen generell optimizerfeil eller begrunnelse for ny sentrering er påvist.

Gamle fullcache-inputs har en annen numerisk eval-kontekst enn short002.
Ny001observer nå FAILED_CONSUMED etter1forward/16rader. Alle akser/queries/masker
og første tap er bitlike. Bare rå FP32-output avviker: S94/160felt, maks7,15e-7;
F102/160felt, maks2,38e-7. M5side begge maks2,98e-8. Hele modellsignaturen
og kilden522a7462 var uendret. Ingen gradient/optimizer/checkpointskriv.
RESULT/TERMINAL, controller exit1 og TRAIN_FUSION_PARTIAL.npz er bevart.
Dette er en målt numerisk reproduksjonsbegrensning, ikke påvist edgefeil.
Ingen hode-/representasjonsattribusjon er godkjent fra den mislykte fangsten.
Neste er smal kilde-/retained-output-kontroll av utføringsavviket før eventuell
ny avgrenset observasjon.001 relanseres aldri; ingen ny kjøring er innrømmet.
48 fokustester og ekte metadata/cap-gate bestod; de beviste ikke modellparitet.
Historisk dispatcher er biteksakt gjenopprettet, ingen numerisk eiergate svekket.
Fullprefix-fit, familiers nytte, senere generalisering og kapasitet er fortsatt
uavklart. Alle numeriske modell-/loss-/featureeiere er uendret.
