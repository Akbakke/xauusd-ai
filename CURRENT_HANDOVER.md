# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste maskinlesbare arbeidsstatus; originale
terminalkvitteringer/prosesser overstyrer tekst. Historikken finnes i Git.

## Gjeldende status og omfang

TA_DIRECTION_EDGE_CAUSE_FROZEN_FUSION_OBSERVATION_ADMITTED.
Det avgrensede målet om å isolere/reparere M5 TRAIN-svekkelsen og verifisere
sammeTRAIN + separat senereTRAIN er fullført. Det større prosjektet med
robust retningsverdi og senere trading er ikke fullført. Nytt aktivt mål11.11:
isoler målte svakheter i indikatorrepresentasjon, sammensetting og læring.
[Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json) binder neste kontroll.
Alle trenings-/launchporter false; training_enabled_scope=null.
Ingen CURRENT-jobb lever ved lukking. Ingen konsumert plan relanseres.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion forblir stengt.
M5-prognosetakt, genuine kausale M15/H1/H4/D1-input og femTF-mål beholdes.

## Hva som ble isolert og rettet

Ved frosne R1-representasjoner reparerte to faste readoutfits begge
M5 TRAIN-mål uten å endre encoder/input. Dette isolerte en korrigerbar
svakhet i de dedikerte retningshodene; det beviser ikke en unik historisk
Adam-/fellestreningsfeil. RESULT62e59619 og nativeW2 RESULTe5be50b5.
W2 bedret TRAIN men tapte INITIAL på senere data. To skalarer tilpasset
utelukkende lagrede indreTRAIN-prognoser dempet overdrevne sannsynlighetsutslag
ved fast rangering. S=.7981812905830987/F=.6385367563238673.
SLOPE_LOCKb4bd9003 ble publisert før fullhead/senere arrays ble åpnet.
Denne komponenttilpasningen er ikke encoder-OOF: encoder/normalisering
så heleTRAIN, og indre labels er kalibrator-fitdata.

[Native rettelse og kravvis sluttkontroll](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json)
er COMPLETE/konsumert, worker/controller/guard exit0.
ROOT MARKET_STATE_TA_M5_LOCKED_SHRINK_NATIVE_20261111_001.
Kildeca93d789 uendret; PLANc6ca851b; RESULTd8a4ef4d; TERMINALccd02a8a.
Kandidatmodell c171736d2de79396dd86a79ad895f8bf2226e5ad8d996abf1dd7f52055e4a467.
Private CANDIDATE_WEIGHTS.pt SHAef218008; FOLDED_COEFFICIENTS.npz SHAd3e7461f.
Originalvekter og samtlige originale kvitteringer er bevart.

Kun fire eksisterende M5-tensorer er endret. Folden bruker opprinneligeW2
FP32-verdier, FP64 w'=a*w/b'=prior+a*(b-prior), deretter én FP32-avrunding.
Eksisterende importer beviste uendret øvrig modell, buffere, RNG, modi,
parameteridentitet og trainability. Faktisk diskdecode/samme-instans
weights-only-roundtrip og to ekte row0-forwards ga bitlike outputs.
Ingen refit/optimizer/backward/treningssteg eller kanonisk checkpointskriving.

## Målte resultater

Brier INITIAL → faktisk kandidat / fryst prior:
TRAIN S .248473364 → .248300632 / .249651008,10010known.
TRAIN F .249327011 → .249240480 / .249876211,11909known.
Senere S .249652488 → .249638480 / .249827988,6096known.
Senere F .249855791 → .249786447 / .249833416,7165known.
Begge Brier og logtap slår INITIAL/prior i alle fire celler med uendrede
marginer. Hele29574TRAIN og17753senere foreldre er evaluert med sine masker.
5918faktiskeCUDAheadkall gjenbrukte representasjonene. To nye encoderforwards,
én modell, to kapseldekodinger og én privat weights-skriving.
Senere AUC S=.523043378/F=.510747753; dette er ikke treffprosenter.
All-rad makslogitavvik mot ideell krymping5,73e-8, maks logtapavvik6,10e-11.
Ingen tvetydig primærgrense fra avrunding.

Uavhengig24skårceller/120felt, maksavvik1,11e-16,0,510s:
foldede koeffisienter, faktiske row0-outputs og alle primære tall stemmer.
Ingen modell/fit/forward/matriseprojeksjon ble gjentatt.
Tallkontrollens første metadataoppslag brukte feil priornøkkel; feilen
oppsto før arrays og ble rettet til eksisterende reverse_recipe-eier.
Opprinnelig feilkvittering bevart; nativekjøringen ble ikke gjentatt.
28 fokuserte tester, syntaks, added-path-scan og git diff --check består.

Helekommando104,609s (~1m45s),worker100,606s. Tidligere full senerefangst
ble gjenbrukt. Dette er faktisk kontrolltid, ingen kontrollert heltreningsbenchmark.
250W konfigurert; guardtopp128,16W/43Ccore/46Cmemory/592MiB.

## Bevisgrense og neste arbeid

Den kravvise sluttkontrollen viser at dette avgrensede korrigeringsmålet
er oppfylt. Ingen ytterligere modellkjøring er nødvendig for målet.
Originale inputs/MTF/targets/vakter er bevart;85modell-/treningseiere
er verifisert med kun den nye eksakte cap-stien som tillatt kildebro.
Ingen generell refaktorering, ny terskelsøking eller ekstra epoke.

Senere oktober–desember2024 er gjenbrukt utvikling, ikke urørt generalisering.
Gevinstene er små; stabil markedspresisjon, statistisk edge og profitt er ubevist.
Unik optimizerårsak, fresh constructor og kanonisk optimizer-resume er ikke
kvalifisert og var ikke krav for denne avgrensede rettelsen.
Fullpopulasjonen brukte faktiske native hoder på kvalifiserte representasjoner;
bare to fullstendige encoderforwards ble gjort.
Bevar den kvalifiserte kandidaten. Neste arbeid er en passiv observasjon av
gater og representasjoner før/etter sammensetting. Én eksakt kontroll er
innrømmet: MARKET_STATE_TA_FUSION_OBSERVATION_20261111_001, PLANa6fb8f98.
512deklarerte rader/33forwards,0updates.45fokuserte tester og faktisk
metadata-preflight består;85modell-/treningseiere beholdes med kun ny
cap-sti. Modellen, prognosene og RNG skal være uendret. Kilde/policy fryses.
Korte runder er bevisst feilsøking. Full kvalifisert TRAIN-historikk og30–40
epoker er senere skalering når grunnlaget er klart, ikke et krav ved rettelser.
Ingen kort test avgjør endelig kapasitet eller garanterer senere edge.
