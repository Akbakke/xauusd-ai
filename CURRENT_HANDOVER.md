# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo er /home/andre2/src/GX1_CURRENT, branch work/gx1-current på
gx1-3090-lan, WSL Ubuntu-22.04. Root i denne tråden eier integrasjon, policy
og launch. Bruker har avvist oppdrag fra andre chatter. Den særskilt bestilte
uavhengige retningsagenten har fullført; ingen ny delegering er nødvendig.
GX1_RULES skal leses helt ved første arbeid. Kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er eneste arbeidsstatus; originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

DIRECTION_BPS_ALLOCATION_PLAN_001 er engangsadmittért, ikke startet.
15 fokuserte syntetiske kontroller bestod; originalene er verifisert,
cgroup borte og originallås ledig. Syntetisk autoritet er konsumert.
Den nye enheten leser9eksisterendeTRAINarkiver, med0fit/nativeforward/update,
CPUaudit4G/512M, én tråd,64tasks,180sworker/210souter. Faste bps-bånd,
originale kohorter, masker og B16-rekkefølge beholdes. Utfallsbånd er kun
diagnostikk. Sluttmodellens batchloss er ikke historiske optimizergradienter.

F_TERMINAL_READOUT_20261027_001 er COMPLETE/claimed/consumed. Ingen jobb kjører,
cgroup er borte, original tunglås er identitetskontrollert og ledig. Alle nye
payload-/fit-/forward-/treningsautorisasjoner er lukket. Oppfølgingen er PAUSED.
Ingen engangsplan relanseres. Alle originaler og checkpoints bevares.

Original worker og capped outer avsluttet med exit0. Worker brukte 12,4590s,
leste fem lagrede TRAIN-NPZ, utførte én prefikskoordinattilpasning og tre faste
L-BFGS-B-fits, med to prediksjonspopulasjoner og 12+12 blokk-/vektede rangkall.
Ingen native konstruksjon, forward, reverse, optimizersteg eller rådatasettlesing.
CPUaudit4G/512M, én numerisk tråd, 64tasks. Kilde, policy og originalinputs var
uendret. Root har bare lest kilde og JSON-aggregater/metadata, ikke payloads.

F_TERMINAL_READOUT_ACTUAL_ROOT_REVIEW_001.json binder originalene.
F_TERMINAL_READOUT_VERIFIED_TERMINAL_001.json er en verifisert metadatakonvolutt.
Aggregert repo-kopi: docs/DIRECTION_FROZEN_READOUT_RESULT_20261027.json.
Kilde: gx1/scripts/fit_entry_native_direction_frozen_readout_v1.py.

18 nye fokuserte syntetiske kontroller bestod. Før genuin admisjon ble en feil
i det nye verktøyets feltnavn for original receipt-status funnet og rettet.
16 originale metadata-/avvisningskontroller kvalifiserte den eneste endringen.
De 18 numeriske kontrollene og tidligere 21 readout-matematikktester gjenbrukes.
Dette var en diagnoseverktøyfeil før genuin kjøring, ikke en modellårsaksforklaring.

## Resultatet og betydningen

Samme frosne trente modell, 128Dhidden, 47 602 komplette prefiksforeldre,
17 288 senere foreldre og uendret 1/5/15-fasit. Alle tre fits konvergerte med
gradientmaksimum under 1e-6 og bedre prefiks-BCE enn originalhode/prior.
Fittet tilstand ble lagret før senere payload ble åpnet. Originale native-/prior-
skårer ble reprodusert. Identitet, klokker, fortegn, masker og presentasjonsrekkefølge
er kontrollert. Resultatet er én gjenbrukt utviklingsfold innen TRAIN.

Ved 5min steg TRAIN-AUC fra 0,515888 til 0,539421.
Senere AUC var 0,503021 originalt og 0,503360 etter fast head-fit:
delta +0,000339. Forhåndsbestemte parvise 99%-intervaller:
UTCuke [-0,009623; +0,010859], UTCdag [-0,010497; +0,010879].
Begge inkluderer0. Primærkravet om bedre senere rangering er ikke støttet.

TRAIN-AUC og proper scores ble bedre ved alle tre horisonter. Senere logloss
ble dårligere ved alle tre; alle seks deskriptive 99%uke-/dagintervaller for
forskjellen ligger over0. Ved5min ble logloss 0,693514 → 0,700082.
Disse sekundære intervallene er ikke en familievis eller økonomisk påstand.

En bedre tilpasset siste lineær head er dermed ikke en dokumentert løsning
på senere retning i denne folden. Kandidaten tas ikke inn som ny native head.
Dette svekker enkel head-underfitting som eneste forklaring; det avkrefter ikke
bedre backbone-læring, flere epoker eller ikke-lineær informasjon. Svakt eller
ustabilt fremoversignal og informasjonstap før slutt-hidden er fortsatt uavklart.
Fold1 ET252-AUC5 er 0,501344, under native i akkurat denne folden; tidligere
samlet firefold-ET-fordel må ikke tilskrives denne folden.

Ingen robust ny generalisering, Entry-/handelsedge eller økonomi er dokumentert.
Arbeidsmålet er aktivt og uferdig.

## Mål og uavhengig gjennomgang

docs/DIRECTION_TARGET_PATH_REVIEW_20261027.json er oppdatert med dette
resultatet, fem prioriterte hypoteser, kildelinjer, motargumenter, tester og stopp.
Agent direction_target_independent_review_20261027 er ferdig og hadde kun
lesende kilde-/aggregatmandat. Ingen ny fortegnsfeil er funnet i undersøkt kjede.

Brukerens 10min er eksplisitt inkludert i målkontrakten. Faste rapportpunkter er
1/5/10/15/30/60; foreslått full fasitkurve er hvert minutt1–60 fra samme M5-entry:
R_h = 10000 * (mid(D+h)-mid(D)) / mid(D), D=M5bar_start+5min.
Den målte native-modellen har fortsatt bare1/5/15utganger. 10/30/60 og dense
native-utganger er ikke implementert eller målt. Ingen mål-/modellkode er endret.

Endepunktsretning er gyldig: tidligere endepunkter kan se oppgangen selv om
et senere endepunkt går tilbake til entry. Men +0,1 og +50bps gir samme BCE-
supervisjon; eksisterende mid_return_bps velges bort av treningsadapteren.
Observerte nuller utelates, de lærer ikke FLAT. Det er en påvist målbegrensning,
ikke bevist årsak til svak retnings-AUC. Om små bevegelser dominerer faktiske
eksempler/loss er ennå ikke målt i denne undersøkelsen.

Modellens harde trigger er z_h=b_h+w_h·hidden >0 for UP, ellers DOWN.
Featurefølsomhet, prediktiv verdi, markedets årsak og en handelsordre er ulike
spørsmål. Gatevekter forklarer ikke kausalt markedet. Ingen handel er åpnet.

60 likevektede horisonter gir45/60vekten etter15min ved komplette masker.
Dette er et fremtidig designvalg, ikke årsaken til dagens1/5/15resultater.
60 marginale sannsynligheter er ingen felles prisbane. Ingen intraminuttrekkefølge
gjenopprettes. Senere short krever en ny kausal beslutning; ingen fasittopp.

## Neste minste steg

Neste foreslåtte måling er bps-fordeling og lossbidrag på eksisterende lagrede
TRAIN-fasiter og prediksjoner, uten fit eller modellkall. Bruk samme foreldre,
klokker og modenhetsmasker; bind alle bps-bånd før payloadlesing. Skill eksakt
null, manglende og umodne utfall. Rapportér alle bånd og eventuelle udefinerte
skårer. Utvalget etter realisert bevegelse er kun diagnostikk, ingen handelsfilter.

Denne målingen er foreslått, ikke admittért eller startet. Deretter vurderes
størrelsessupervisjon, horisontutvidelse og én-mot-tre native epoker som separate
faktorer. Ingen automatisk30epoker, beste-minutt-seleksjon eller terskelsøk.
Eksakt10-fasit kan eventuelt bygges av kvalifisert5+5kjede. Alleminuttfasiter
krever en avgrenset kvalifisert TRAIN-M1kilde for manglende minutter; aldri
interpolasjon eller selvstendig åpning av Parquet med senere2026data.

## Bevarte tidligere bevis og grenser

Alle fire terminal-uttrekksenheter er komplette. Den opprinnelige åtte-enhets
0U-versus-terminal-studien er fortsatt uferdig: seks av åtte uttrekk fullført.
To manglende own0U-enheter er utsatt. Den gamle planen er uendret og gir ingen
ny autoritet. Prefiksuttrekkene mangler historisk per-rad saved-output-paritet;
state/input/order-kontroller er ikke en erstatning for denne.

Firefoldsanalysen på67 112samme senere foreldre viste nativeAUC5=0,503735
mot ET252=0,513092, med negative parvise uke-/dagintervaller korrigert for seks
kontraster. Alle118 249TRAIN-foreldre hadde bitidentiske2622lokale felt/masker
mellom native og referanse. Full numeriskMTFparitet eller felles upstreamfeil
er ikke avklart. DIRECTION_1_60_CAUSE_SYNTHESIS_008 bevarer disse bevisene.

Tidligere TRAIN-only biasjustering forbedret5mproper score, men AUC var eksakt
uendret. GC-OFI hadde sterk samtidigGC-respons og svak fremtidigXAU-sammenheng
i det avgrensede sporet; dette avviser ikke alle inputenes fremoversignal.
731tensordiagnosen avkrefter helt stillestående trenbare vekter, ikke uproduktiv
læring. Ikke gjenta beståtte råinputkontroller eller avsluttede forsøk.

M5 Entry, åtte spesialistfamilier og D1/makro beholdes. M5/M15/H1/H4/D1 er
allerede i faktisk Entry-rute; 61D1/makrofelt eies av session_regime_encoder.
TRAIN mai2024–desember2025. CONTROL januar–mars2026 er lukket. Juni2026 er
gjenbrukt utviklings-VAL. TEST fra juli2026 er forseglet for tilpasning/utvalg;
forward-journalen gir ikke denne oppgaven TEST-autoritet. Ingen Exit-trening,
broker, live/paper, ordre, spending eller promotion. Teknisk PASS er ikke edge.
