# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_PAIRED_BOUNDARIES_COMPLETE_M5_GAIN_AND_CALIBRATION_DRIFT.
Målet11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. Korte runder er bevisst feilsøking, ikke kapasitetsdom.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.

Alle engangsplaner nedenfor er COMPLETE/CONSUMED; ingen relansering.
Training/native_training/native_launch/full_epoch/full_val er false.
Ingen ny modell-, data- eller treningskjøring innrømmet. Ingen operasjonell blokkering.
GX1TFActivityShortLearning001, GX1NonM5ShortLearning001 og begge c171-oppgavene
ble deaktivert etter sine fullførte kjøringer; ingen ny oppgave ble opprettet.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion stengt; GC på pause.

## Indikatorer og sammensetting

Alle8genuine TAfamilier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning, varierende TRAIN-støtter og oppdatering i11encodere/228tensorer
beviser mekanikk, ikke prediktiv familienytte. Mean+last kansellerer ikke generelt;
fast MTF-intervensjon ga ingen bedre rangering. Ingen generell normaliserings-
eller konstantinputfeil er vist. Alle genuine features/TF og kausalitetsvakter er bevart.

Nærmeste trendlinjeslots mangler entydig samme-linje-identitet, ankere og endringsmarkør.
Dette er et målt semantisk hull; prediktiv kostnad er UNKNOWN. Neste kildearbeid er
å følge eksisterende produsent og pakking til faktisk tidssekvens, og avgrense minste
kausale rettelse. Ingen ny identitetsinfrastruktur eller modellendring er vedtatt.
[Samlet oversikt](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).

## Nyeste faktiske læring

[Samlet resultat og bundne kvitteringer](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json).
LæringsROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001.
Kilde7dcddb37586cdcf297ffac2e8b1f643912f4c50a uendret under kjøring.
RECIPE4ae0082f, PLAN63b7702e, RESULTaca65dc4, TERMINAL039d6a5b.
Samme4096kronologiske TRAIN-rader/256updates som originalshort002,1.–21.mai2024.
Originale startvekter og trenings-RNG, fersk AdamW, uendrede hyperparametre,
features/targets/tapsvekter. Ti separate aktivitetshoder på M5/M15/H1/H4/D1 forS/F;
20klonede tensorer/1290parametre. Originale M5sidehoder beholdt; nonM5-sidehoder av.
768forwards:256initial+256fit+256final;0skips. Arbeider948,98s, fit350,19s.
Alle mekaniske baseline-/vaktkontroller bestod. Ingen ny trening i siste bølge.

| Samme brukte TRAIN | Initial | Original final | Sidehoder final | Aktivitetshoder final |
| --- | ---: | ---: | ---: | ---: |
| Opprinnelig totaltap | 1,878003 | 1,892871 | 1,884922 | 1,884324 |
| S/M5 side-AUC | 0,546185 | 0,551466 | 0,551231 | 0,551352 |
| F/M5 side-AUC | 0,546125 | 0,553079 | 0,553037 | 0,553147 |
| S/M5 side-Brier | 0,247999 | 0,247924 | 0,248301 | 0,248027 |
| F/M5 side-Brier | 0,248788 | 0,248193 | 0,248318 | 0,248218 |

AUC er rangering, ikke treffprosent eller økonomi. Aktivitetshodene reduserte netto
aktivitetsregresjon87,23% mot original; dette er ingen87% forbedring i prediksjon.
Totaltapet forverres fortsatt. Ingen entydig M5gevinst mot originalfinal: AUC endres
i ulik retning, og beggeM5Brier/log er litt dårligere. Korte runder dømmer ikke kapasitet.

## Full paired observasjon: fullført og kontrollert

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_BOUNDARIES_20261112_001.
Kilde f170a91cdd20e045417654bf1531f6008e090052 uendret under observasjon og analyser.
PLAN6f0bc72e, PREFLIGHT3d964d2d, RESULT95f1fc32, TERMINAL87551f40,
CONTROLLER5872aeaf, TRAIN_FUSIONba1ea384. Engangsclaim konsumert; alle exit0.

1modell/2eksisterende checkpoints, strict INITIAL→FINALvektovergang uten RNG-/
optimizerrestore.512B16forwards/8192observerte rader på samme4096unikeTRAIN.
Begge512Djoint-source,128Djoint-hidden og alle10TFhodeinnganger beholdt.
Alle12originalfelt og256batchtap per endpoint byteeksakte; tidligere INITIALh0
også byteeksakt. Vekter/buffere/RNG/modi/flagg/hooks uendret innen hver fase.
20headregistreringer/5120hodeobservasjoner;512source/512hidden/1024queryinputs.
0optimizer/backward/autograd/RNGrestore/checkpoint-/kandidatvektskriving.
64.614.682byteNPZ innen64MiB. Arbeider572,36s, controller574,93s.
141fokuserte tester før kjøring. Producer12GiB/512MiB,CPU0–7/pids64,8tråder;
250Wlimit/170Wdraw/65Ccore/80Cmemory/GPU12288MiB,900sarbeider/1200svakt.
456telemetripunkter; topper53Ccore/54Cmemory/137,62W/554MiB; ingen vaktbrudd.

## Målt samspill og M5-rangering

PAIRED_BOUNDARY_REVIEW.json893e1962:4NPZ/2CPUcheckpoints,0modeller/nativeforwards.
Alle16 faste kombinasjoner av H(hoder)/Q(queryprojection)/J(joint)/X(source),
0=INITIAL/1=FINAL. Fire tidligere kvalifiserte semantiske sourcekoordinater:
base, effektivMTFrest, MTFpool, globalcontext. Base inngår i begge første blokker.
OriginalB16/tapsnevnere,/5,Fvekt1,M5q/N bevart. Både fremover- og bakoverrekkefølge;
komponentbidrag er rekkefølgeavhengige, ikke unik historisk gradientårsak.
Hypotetiske logits er ankret til faktiskINITIAL; faktiske endepunkter er separate.

| Fullt opprinnelig TRAIN-tap | Verdi |
| --- | ---: |
| Faktisk INITIAL | 1,878002535 |
| Bare FINALhoder | 1,871377344 |
| Bare FINALqueryprojection | 1,869806905 |
| Bare FINALjoint | 1,870275136 |
| Bare FINALsource | 1,853369240 |
| Faktisk FINAL, alle endringer | 1,884324128 |

Hver endring alene bedrer samlet tap på hele prefixet; dette gjelder ikke alle
oppgaver/kvartaler. Samlet endring forverrer. Det er samspill i representasjoner
og ikke-lineært tap, ikke bevis for at én komponent generelt er skadelig.
S/D1s215FLAT-rader er nå fullt dekket; den tidligere48raders pilotens hull er lukket.
På faktiskFLAT øker Paktivitet88,49→93,35%. Fremoverordnede logitbidrag:
hode0,094984/query0,069193/joint0,145482/source0,337376. Globalcontext er største
enkeltkoordinatbidrag her0,165515; bidragene er betingede og ikke additivt kausale.
F/M15-H1 PUP53,25→69,94% og53,51→70,88%, mot52,08/53,97%observertUP.
Alle fire fremoverordnede komponentbytter forverrer disse sidetapene.

M5_RANK_REVIEW.jsoncf7716ff:1NPZ/2CPUcheckpoints,0modeller/nativeforwards/trening.
Alle22varianter, fullprefix/4fastekvartaler;220tie-aware AUCer mot parvist orakel.
Faktiske20scoreceller matcher uavhengig rapport innen1,11e-16;220tapsledd innen5,55e-17.
CPUbeggeanalyser:4GiB/512MiB,CPU0–7/pids64,1tråd;3,76/3,37sarbeider.
Maks native hidden-replay2,40e-6, raw1,26e-6,512batchtap3,11e-7 innen faste grenser.
Uavhengig statisk etterkontroll90240b0d; ikke fremstilt som forhåndsgate.

| M5 side-AUC, brukt TRAIN | S | F |
| --- | ---: | ---: |
| Faktisk INITIAL | 0,546185 | 0,546125 |
| FINALH/Q/J, INITIALsource | 0,548663 | 0,549486 |
| Faktisk FINAL | 0,551352 | 0,553147 |
| Bare globalcontext endret fra forrige faste H/Q/J-rad | 0,550813 | 0,554644 |

Sourceendringen tilfører M5-rangering samtidig som den bidrar til kalibreringsfeil
andre steder. Globalcontext kan ikke generelt fjernes/fryses som skadelig.
Hypotetiske kombinasjoner er diagnostikk, ikke nye modeller eller kandidatutvalg.
S/M5Brier forverres litt; F/M5bedres. Effekter varierer med kvartal: F/M5Q2AUC
faller selv om fullprefixet bedres. Lavere totaltap er ikke det samme som bedre retning.

## Ikke gjort og neste steg

Ingen frys, indikatorfjerning, tapsendring eller valgt endpointblanding er begrunnet
av disse regnskapene. D1bias alene forklarer bare0,50% av middel-logitøkningen;
Adam-/biasregnskap og originale kvitteringer er beholdt i læringsROOT, ikke relansert.
Historiske delte source/querygradienter finnes ikke i beholdt treningsspor;
M5sideparametergradienter/momenter er de direkte fangede parameterstørrelsene.

Neste arbeid: kildeavgrens den konkrete trendlinje-identitetsmangelen mot eksisterende
produsent, pakking og modellkonsum. Minste semantiske rettelse må bevare alle genuine
familier/TF og årsakstid. Endret modellfunksjon krever fokuserte kontroller og fersk
initialbaseline; ingen numerisk kjøring er innrømmet nå. Ikke gjenta fullførte målinger.

Ett seed/256updates og avhengig brukt TRAIN. Senere generalisering, økonomisk edge,
familienytte, trendlinjemangelens prediktive kostnad og langtidskapasitet er ubevist.
Målet er fortsatt aktivt. TEST er urørt; fullhistorikk/30–40epoker er senere skalering.
