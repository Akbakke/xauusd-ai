# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_SHORT_HEAD_REPRESENTATION_REVIEW_COMPLETE.
Målet11.11 er aktivt: mål årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle engangsplaner er konsumert; ingen ny kjøring er innrømmet eller aktiv.
Training/native_training/native_launch/full_epoch er false. Begge gamle c171
Windowsoppgaver er Disabled. GC er på pause; TEST, Entry/Exit, broker, handel,
spending og promotion er stengt. Alle genuine indikatorer og fem TF beholdes.

## Indikatorer og sammensetting

Åtte TA-familier og M5/M15/H1/H4/D1 når faktisk modellkonsum. Numerisk dekning
og ikke-konstante TRAIN-støtter er bekreftet; det beviser ikke prediktiv nytte.
Alle11 temporal-/akse-encodere og228 undersøkte tensorer hadde vektendring og
Adam-andremomenter. En generell frakoblet læringsvei er avkreftet.
Mean+last kansellerer ikke systematisk. Variasjonsfall i blandingen er ikke
påvist prediktiv skade; tidligere fast MTF-intervensjon ga ingen bedre rangering.
Nærmeste trendlinjeslots mangler entydig samme-linje-identitet/ankere/endringsmarkør;
prediktiv kostnad er UNKNOWN. Ingen spekulativ feature-/modellendring er utført.
Låst head-/sannsynlighetsrettelse ga tidligere liten Brier/log-forbedring, med
svak rangering. [Native evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).

## Siste faktiske læring og det målte nivåskiftet

Native short002 er COMPLETE/konsumert:256updates over4096 kronologiske TRAIN-rader,
0skips,288forwards,256backward, fersk tom AdamW og samme c171-initial/originaltap.
RESULT1b25f095; kilde9de7e20b uendret;513,87s worker/365,99s fit;guard/observer0.
[Kontroll og begrensninger](docs/TA_M5_C171_SHORT_LEARNING_REVIEW.json).
Samme16B16 grupper før/etter;3grupper/48rader fikk nye updates,13/208 er andre
historisk brukte TRAIN-rader, ikke heldout. Kjent side S91/F104, avhengige rader.
Totaltap1,875017→1,892264; S/M5Brier0,246301→0,243797;F0,251730→0,254568.
Begge M5-sidetap ble dårligere i de3 nyeksponerte målte gruppene; dette er ikke
full4096-raders fit-evaluering. Pooled AUC0,53575→0,54121 /0,46939→0,47681.
Faktisk gjennomsnittlig opp-sannsynlighet på kjent støtte steg2,50553/2,19309pp.

## Ny hode-/representasjonsavgrensning

[Samlet ny evidens](docs/TA_C171_SHORT_HEAD_ATTRIBUTION_REVIEW.json).
Beholdt TRACE_ALGEBRA_REVIEW09d5e8dc gjenskaper256 M5-hodegradienter ogAdamW-steg;
direkte biasendringer er negative. Lokal summert head-only BCE-endring var gunstig
for begge. Dette beviser ingen generell optimizerfeil eller sentreringsrettelse.

Lesende headobserver001 stoppet etter1forward: akser/masker/tap like, men små
FP32råavvik; original feil og partial beholdt, aldri relansert. Kildekontrollen
fant1-versus8 numeriske tråder og ulike gradflagg. Egen002 matcher kjent original-
profil uten optimizer/autograd. Kilde446fcd42 uendret;RESULTf6e0c8a3,terminal49493bed,
controller0.16forwards/256rader,0gradient/oppdatering/checkpointskriv;116,14s.
Alle12 initialfelt og16tap er bitlike. Model/buffers/RNG/modus/hooks uendret.
Faktisk8 intra/10 defaultinterop;570 aktive/179 inaktive gradflagg, alle grad=None.
Producer-vakter:12GiB/512MiB,CPU0-7/pids64,250Wlimit/170Wdraw/65C/80C.
52 fokustester og faktiske metadata-/cap-gater består; alle numeriske eiere uendret.

På gjenskapt initialinput gir sluttlagets parameterendring alene opp-sannsynlighet
−0,03676pp S/+0,22419pp F, mot faktisk+2,50553/+2,19309pp. Head-onlyBrier er
0,246292/0,252027. Resten er gjenstående logitendring etter dette beregnede
hodeleddet; det peker mot representasjonsbanen, ikke en bestemt encoderårsak.
HEAD_ATTRIBUTION_REVIEW9b8f49f3 er uavhengig kontrollert:242float-/48tellefelt,
maksavvik1,11e-16;INDEPENDENT_REVIEW03743bb1. Gjenskapt128Dinput er ikke historisk
registrert128Dinput; skalarparitet beviser ikke skjult historisk vektoridentitet.

## Ikke avklart og neste steg

Beholdt faktisk4096-prefix har framtidig kjent-UP51,399%M5,62,926%H4,78,280%D1;
løpende pre-update p_up var50,876/56,585/55,484%. PREFIX_TASK_BALANCE2ba4453a.
Dette er ulike modelltilstander og marginaler, ikke målte delte parametergradienter
eller bevis for skadelig overføring. Tidligere konflikt mellom gjennomsnitts-
gradienter hadde samtidig positiv gjennomsnittlig batch-dot; ingen generell
lossvektendring er begrunnet. Gamle4096-kontrast var INTERLEAVED−CONTROL, ikke
INITIAL−R1. Gammel sentreringsplan er konsumert/FAIL_COMPARISON_NO_PROMOTION.
Neste: isoler om og hvor korte/lange TF-oppgaver flytter fellesrepresentasjonen
mot et felles oppnivå, med allerede beholdte målinger og kode først. Bevar alle
features/TF; ingen ny fit eller loss-/modellendring følger bare av marginalene.
Unik upstream-årsak, fullprefix-fit, familiers nytte, senere generalisering og
kapasitet ved tilstrekkelig trening er fortsatt uavklart. Ingen ny edge er påvist.
