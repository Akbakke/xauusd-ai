# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_FULL_USED_PREFIX_READ_ONLY_ADMITTED.
Målet 11.11 er aktivt: mål årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Tidligere modellplaner er konsumert. Én ny lesende fullprefix-evaluering er innrømmet; ingen ny trening.
Training/native_training/native_launch/full_epoch/full_val er false. Begge gamle
c171 Windowsoppgaver er Disabled. GC er på pause; TEST, Entry/Exit, broker,
handel, spending og promotion er stengt. Sluttcheckpoint er evidens, ikke promotion.

## Indikatorer og faktisk læring

Åtte TA-familier og M5/M15/H1/H4/D1 når faktisk modellkonsum. Numerisk dekning og
ikke-konstante TRAIN-støtter er bekreftet; det beviser ikke prediktiv nytte.
Alle 11 temporal-/akse-encodere og 228 undersøkte tensorer hadde vektendring og
Adam-andremomenter. Mean+last kansellerer ikke systematisk; tidligere fast
MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler entydig
samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features og tidsrammer er bevart. Ingen taps-/modellendring er valgt.
Tidligere låst M5-sluttlag/sannsynlighetsjustering ga liten Brier/log-forbedring,
med svak rangering. [Native evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).

Native short002 er COMPLETE/konsumert: 256 updates over 4096 kronologiske
TRAIN-rader, 0 skips, fersk tom AdamW og samme c171-initial/originaltap.
RESULT 1b25f095; kilde 9de7e20b uendret. Før/etter-evalueringen omfattet bare
16 B16-grupper/256 rader: 3 grupper/48 rader fikk nye updates, 13/208 er andre
historisk brukte TRAIN-rader, ikke heldout. Kjent side S91/F104; avhengige rader.
Totaltap 1,875017→1,892264; S/M5 Brier 0,246301→0,243797; F 0,251730→0,254568.
Begge M5-sidetap ble dårligere i de 3 nyeksponerte målte gruppene.
[Kontroll og begrensninger](docs/TA_M5_C171_SHORT_LEARNING_REVIEW.json).
Dette er fortsatt ikke en før/etter-evaluering av hele den brukte 4096-radersperioden.

## Fullført paret måling

[Resultat og metode](docs/TA_C171_PAIRED_JOINT_BOUNDARY_REVIEW.json).
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_JOINT_BOUNDARIES_20261111_001.
Kilde 69eb8abc uendret. RESULT 0cf0a2dc, TERMINAL a621aac0, controller 68d021bf/exit0.
136,35 s worker/138,74 s controller. Én modell, 32 forwards, to checkpointlesninger,
én eksplisitt import av finalvekter; 0 optimizer/autograd/updates/checkpointskriv.
Alle 12 historiske felt og 16 tap er bitlike per fase, initialt også begge headinputs.
Modell/buffers/RNG/modus/gradflagg/identiteter/hooks var passive innen hver fase;
vektimporten bevarte ikke-numerisk tilstand. Faktisk 8 intra/10 defaultinterop.
TRAIN_FUSION.npz 6b2d5532 har 512 evalueringsrader med fase0/1 og 256 unike rader:
originalpanel, M5-headinputs, faktisk 512D joint-source og 128D joint-hidden.
Størrelse 1 945 370 byte innen 2 MiB. 92 fokustester, faktisk preflight og cap-valg
bestod. Vakter: 12 GiB/512 MiB, CPU0–7/pids64, 250 W limit/170 W draw/65 C/80 C,
900 s worker/1200 s guard. Ingen kilder ble endret under kjøringen.

Første PLAN-publisering manglet foreldremappen og feilet før PLAN/CLAIM; original
feil er bevart i PRE_PLAN_PUBLICATION_FAILURE. Bare den nye mappen ble opprettet.
Eldre cache-avvik og tidligere kvitteringer er bevart i deres opprinnelige rapporter.

## Hva den parede beregningen viser

PAIRED_BOUNDARY_EFFECT_REVIEW 587650d3 er uavhengig kontrollert av 2f5f045d:
580 tallfelt/132 eksakte felt; ingen konkrete avvik, maks 1,49e-13 i L2-summering.
Kun CPU-aritmetikk på to beholdte checkpoints/fire NPZ; ingen modell/fit.
Maks FP64-replayavvik i M5-rålogit er 6,50e-8 ved endepunktene. Begge ordnede
identiteter lukker innen 2,78e-17, med begge endepunktsfeil eksplisitt medregnet.

Med sluttlagene holdt faste senker sourcebyttet initial→final AUC i alle tre grupper:
ALL S 0,548659→0,541212/F 0,486456→0,476809;
UPDATED3 S 0,422222→0,400000/F 0,371212→0,340909;
OTHER13 S 0,636728→0,623095/F 0,548526→0,536241.
Dette er betinget sensitivitet på et avhengig TRAIN-panel, ikke bevist informasjonstap.

Ved rekkefølgen tail-først er råmiddelskiftet S 0,042509 tail + 0,061252 source;
F 0,045919 + 0,043369. Motsatt rekkefølge gir andre ledd; interaksjonen er beholdt.
Av fire forhåndsvalgte semantiske enkeltbytter gir global_context størst positivt
middelsignal: +0,036349 S/+0,028676 F. Basebyttet gir mye av rangfallet i ALL/OTHER13.
For F i UPDATED3 er baseeffekten på rang null; kontekstbyttet gir rangfallet der.
Basebyttet endrer begge første sourceblokker. Effective MTF residual er postMTF−base
og omfatter både MTF- og cooperation-tillegget; pool og kontekst er egne blokker.
MTF-residual/pool har felles middelandel av kvadrert endring 0,995912/0,965571.
Dette er geometri, ikke andel av prognoser, årsak eller prediktiv familienytte.
Ingen generell frysing, featurefjerning eller ny tapsvekt er begrunnet av panelet alene.

## Neste konkrete avklaring

[Full brukt-prefix-kontroll](docs/TA_C171_FULL_USED_PREFIX_REVIEW.json) er ADMITTED.
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001.
Én modell, to beholdte vektsett, 256 B16-grupper per fase: høyst512forwards.
Alle4096brukte rader/tider/queries/masker og støttecounts bindes byteeksakt til TRACE.
TRACE sine pre-update-logits kommer fra skiftende vekter og brukes ikke som baseline.
Kontrollgruppene0/123/246 må matche gamle panelposisjoner0/1/2 i begge faser.
Original evaluate_fit_batch og _short_loss_components gjenbrukes uendret; ingen hooks.
Bare original12felt+phase lagres, cirka1,70MB under2MiB. 0optimizer/autograd/fit.
68fokustester, metadata-preflight og faktisk cap-selektor bestod. Uavhengig
kildereview fant ingen konkret blokkering. Numeriske eiere er uendret.
16-/32-forward-observerne gir et lineært512-forward-estimat på cirka743sekunder,
men fullprefix-varighet er ikke bevist. Behold900sworker/1200sguard og alle vakter.
Ved feil beholdes partial/FAILED_CONSUMED; ingen relansering av denne planen.
Kjør bare eksakt innrømmet plan, avslutt scopet og sammenlign komplett egen-prefix-fit
før valg av modellrettelse. Senere generalisering og ny edge er fortsatt ubevist.
