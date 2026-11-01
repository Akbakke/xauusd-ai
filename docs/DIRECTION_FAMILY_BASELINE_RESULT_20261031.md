# Matchet familiebaseline: full TRAIN og senere utvikling

Baseline er gjennomført, men familiesammenligningen er ikke ferdig. Aktuell
programstatus står i NEXT_RUN_POLICY.json. Brukeren avviste 01.11 alle nye
modellkjøringer, også 256-raders kapasitetstesten. Ingen indikatorfamilie
erklæres nyttig eller unyttig av denne baselinen.

## Målt på faktiske data

Frossen kilde: 68d63d9210931663510f0a9cf490387e9c8b29be.
Baseline brukte samme initial, rekkefølge og numeriske oppskrift som den bundne
sign-referansen: ett pass, 47 602 TRAIN-foreldre og 2 976 oppdateringer.
Boot588 trente; boot589 skåret hele TRAIN og 17 288 senere Q1-foreldre uten
nye backward-/optimizersteg. Q1 er gjenbrukt utvikling, ikke urørt TEST.

Fem minutter er forhåndsvalgt hovedhorisont. Tallene gjelder observerte,
ikke-null utfall som modnes strengt før delens cutoff.

| Måling | TRAIN | Senere Q1 |
|---|---:|---:|
| Foreldre | 47 602 | 17 288 |
| Gyldige femminuttersutfall | 47 282 | 17 192 |
| Treff | 49,5347 % | 48,6156 % |
| Treff, alltid opp | 50,9052 % | 51,5821 % |
| AUC | 0,5023466974 | 0,4969610280 |
| Logloss | 0,6935680652 | 0,6943814277 |
| Logloss, fast TRAIN-prior | 0,6929832916 | 0,6927381622 |
| Brier | 0,2502103864 | 0,2506164220 |
| Brier, fast TRAIN-prior | 0,2499180600 | 0,2497955087 |

Modellen predikerte opp på 8 722/47 282 TRAIN-utfall og 3 226/17 192 senere
utfall. Hovedhorisonten er svak også på TRAIN; målingen identifiserer ikke
årsaken og beviser ikke at indikatorene mangler informasjon.

Alle60horisonter, kun deskriptivt: TRAIN hadde AUC over0,5 på59 og bedre
logloss/Brier enn fast TRAIN-prior på55. Senere hadde AUC over0,5 på54,
men logloss/Brier var dårligere enn prioren på alle60. Rangering og
sannsynlighetskvalitet må derfor holdes atskilt. Ingen nye intervaller eller
simultan signifikanspåstand følger av disse tellingene.

## Bevist konsistent innen kontrollens omfang

Kanonisk kvitteringskjede og faktiske checkpoint-/prediksjonshasher bestod
capped kontroll. Begge native vakter bestod. Treningen brukte2 993forwards;
skåringen4 057=2 976TRAIN+1 081Q1forwards, null backward/optimizersteg.
Modell/buffere/RNG/modi er bevart gjennom evalueringene. De to terminalene
rapporterer samme semantiske full-state-segl; etterkontrollen deserialiserte
ikke modell-/optimizertilstanden.

Separat4GiB-kontroll leste initial256, fullTRAIN og Q1-prediksjonene,
kontrollerte dimensjoner/domener og kanoniske rad-/beslutningstidsdigester
fra den fullførte inputauditen med samme manifestlineage, og gjenskapte alle60
modell-/priorpaneler eksakt med eksisterende metrikk-eier. Fast prior ble
gjenskapt fra hele TRAIN og brukt uendret senere. Dette er ny avspilling av
lagrede utdata, ikke uavhengig metrikkalgebra eller ny råfasit-revisjon.
Initial256 og fullTRAIN er forskjellige populasjoner, ikke en matchet
før/etter-læringskurve.

Engangstasken er deaktivert;250W-grensen er beholdt og faktisk ettertrekk
var26,83W ved0%GPU. Treningsporter er stengt. TEST, Exit og handel er urørt.

## Avvik og ubeviste grenser

En foreldet protokollhash i current_work-sammendraget er rettet.
Overleveringskontrollen krevde eksplisitt exit_code i den nye oppsummerte
terminalen; et korrigert dokument binder de faktiske nullkodene og bevarer
originalen uten overskriving.
De to eksakte program-/stagebindingene var allerede riktige; native kilde,
input og numerikk er uendret.

Forbindelsen kom tilbake før Windows-innlogging ved boot588 og589, men var
forsinket ved589. Tjenesten startet automatisk; årsaken til forsinkelsen er
ukjent. Ingen varig nettverksretting hevdes. Alle nye kontrollerte
prebootkvitteringer overlevde begge omstarter.

Inputauditens målte dekning, sjeldne H1-flowfelt og øvrige kunnskapshull
består: se DIRECTION_FAMILY_INPUT_FULL_RESULT_20261030.md.
Familieverdi, bekreftet generalisering og netto handelsøkonomi er ubevist.

## Bundet evidens

Runrot: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_BASELINE_001

- VERIFIED_TERMINAL_002.json binder original native terminal, capped hovedkontroll,
  separat lagret-prediksjonskontroll og POWER_RETAIN.json.
- ACTUAL_ROOT_REVIEW_001.json binder begge native vinduers kjede og full-state.
- SAVED_PREDICTION_VERIFY_001/RESULT.json og TERMINAL.json binder
  faktisk avspilling, kilde-/inputbookends og eksplisitte evidensgrenser.
- RUNTIME/prefix-fit/window-0002/FINAL_TRAIN.json og OOF.json binder
  sine komplette prediksjonsarkiver.

## Lesende før/etter-diagnose 01.11 — ingen modellkjøring

De 256 initialradene ble funnet entydig i lagret full-TRAIN-sluttfil. Foreldre,
M5-/beslutningstider, alle 60 targettider, fortegn, observasjonsmasker og
kontekstmerker matcher eksakt; initialidentitet, feature-view og prior er like.
Dette er de første 256 presentasjonene i den lagrede treningsrekkefølgen.
Uavhengig NumPy-regning gjenskapte alle 60 initial- og full-TRAIN-sluttpaneler.

Samme 256 gyldige femminuttersutfall (128 opp, 128 ned):

| Måling | Før | Etter |
|---|---:|---:|
| Riktige | 129/256 | 126/256 |
| Treff | 50,3906 % | 49,2188 % |
| AUC | 0,440979 | 0,545166 |
| Logloss | 0,784835 | 0,692317 |
| Brier | 0,291121 | 0,249585 |
| Gjennomsnittlig oppsannsynlighet | 32,6995 % | 49,3432 % |
| Standardavvik i oppsannsynlighet | 6,5024 prosentpoeng | 0,8328 prosentpoeng |

Tilpasning og innsnevring av sannsynlighetene er målt; femminutters treffgevinst
er ikke målt. Fast 50 %-prognose gir logloss 0,693147 på disse radene. Sluttens
gevinst mot denne er bare 0,000830, mot 0,092518 reduksjon fra start. Dette er
en deskriptiv sammenligning, ingen årsaksfordeling av læringen. Logloss bedres
på 56/60 horisonter på samme utvalg; utvalget beviser ikke generalisering.

Tre originale tilstandsfiler (276 497 592 bytes samlet) ble SHA-256-verifisert
og lest som tensorordbøker på CPU med weights_only=True og mmap=True. Ingen
modell eller optimizer ble konstruert eller gjenopprettet. Alle 627 aktive
parametertensorer er endret, og alle 627 har Adam-stegteller 2 976. Alle 179
inaktive er uendret; alle buffere er identiske. Trente parametere er identiske
i trenings- og scoringterminalen, og optimizerdigesten er lik. Forklaringen
er dermed ikke uteblitte parameteroppdateringer eller scoring av startvekter.

Opprinnelig kilde 68d63d9210931663510f0a9cf490387e9c8b29be krever eksakt
parent-join mellom inputs og targets. Loss er BCE over gyldige, observerte,
ikke-null terminalfortegn ved 1–60 minutter, uten særprioritet for fem minutter.
Størrelse, MAE/MFE og FLAT-supervisjon inngår ikke i denne baseline-loss.
Dette er et dokumentert avgrenset mål, ingen bevist årsak til svak rangering.

Årsaken til svak senere retningsprediksjon er fortsatt ukjent. Mellomliggende
matchede prediksjoner mangler i denne kontrollen; vedvarende svak tilpasning
kan ikke skilles fra læring som senere glemmes. Rå featureformler og rå quote-
fasit er ikke revidert på nytt. Ingen trening, forward/backward, ny evaluering,
omstart, TEST eller handel er utført.

Aggregert evidens og eksakte originale filhasher:
[DIRECTION_FAMILY_BASELINE_SAVED_DIAGNOSIS_20261101.json](DIRECTION_FAMILY_BASELINE_SAVED_DIAGNOSIS_20261101.json).

## Utvidet årsaksdiagnose og løsningsforslag 01.11

Brukerens nye mål ble opprettet etter at retention-eieren slettet tre
forlatte, byte-identiske materialiseringsmapper: **72 128 162 964 bytes**
(72,13 GB / 67,17 GiB). Ledig Linux-plass økte målt 72 128 102 400 bytes.
Aktive datasett, originale vekter/prediksjoner og gjenbrukt SUPPORT_SCRATCH
er beholdt. Ingen omstart eller VHD-komprimering ble utført.

### Ny måling uten modellkjøringer

Alle sju komplette varianters lagrede TRAIN/Q1-prediksjoner ble undersøkt
i én capped 4 GiB CPU-analyse. 396,6 MB arkiver, 3,45 sekunder i analysen.
Samme 47 602 TRAIN- og 17 288 senere foreldre; gyldige femminuttersutfall
47 282 og 17 192. Foreldre, klokker, fasit og masker matcher mellom armene.
Den faktiske lagrede treningsrekkefølgen er digest-verifisert. Alle 14
hele femminutterspaneler er gjenskapt; arkivene ble rehashet etter analysen.
Dette fullfører ikke den avbrutte ti-armers familiesammenligningen.

På forhånd fastsatte paneler: alle ti desiler i treningsrekkefølgen,
første/siste 256 presentasjoner og alle kalendermåneder. Ingen beste arm,
måned eller terskel er valgt. TRAIN dekker mai–desember 2024; senere data
er januar–mars 2025. Disse periodene er gjenbrukt utvikling.

- Siste TRAIN-desil har AUC **0,49946–0,52054**. Seks av sju varianter har
  lavere AUC der enn i første desil. Ingen konsekvent fordel for de sist
  presenterte radene; glemsel er likevel ikke formelt avvist.
- Alle 21 variant/måneds-paneler i Q1 har AUC **0,49116–0,51745**. Ingen
  sterk deskriptiv måned skjules av kvartalsgjennomsnittet. Dette er ikke
  statistisk bevis for fravær av ethvert signal.
- Svakhet finnes allerede på TRAIN. Et senere domene-skift kan derfor
  ikke alene forklare hele problemet.

Baseline viser også en tydelig horisontforskjell:

| Prognosehorisont fra M5 | TRAIN AUC | Senere Q1 AUC |
|---|---:|---:|
| 1 minutt | 0,50038 | 0,49914 |
| 5 minutter | 0,50235 | 0,49696 |
| 15 minutter | 0,53277 | 0,50743 |
| 30 minutter | 0,54556 | 0,51253 |
| 60 minutter | 0,54801 | 0,51895 |

Det finnes høyere målt rangering på lengre horisonter, særlig i TRAIN.
Dette kvalifiserer ikke nyttig fremtidig retning; Q1-sannsynlighetsscorene
er fortsatt svakere enn den faste TRAIN-prioren på alle 60 horisonter.

### Konkret designasymmetri bekreftet fra kode og lagrede vekter

Den utvidede prisruten for RSI14, MACD-linje, MACD-histogram og ret_1
nullinitialiseres eksplisitt i `entry_v10_ctx_hybrid_transformer.py:1560`.
Nullstarten er et bevisst valg for å bevare forgjengerens funksjon.
GC-projeksjonene får ordinær Linear-initialisering. Den aktuelle kilden er
byteidentisk med den opprinnelige baselinekilden.

Alle 512 prisprojeksjonsvektene startet på null. Etter 2 976 Adam-steg er
alle endret, vekt-RMS er 0,002574 og Frobenius-normen 0,058242. De fem
GC-projeksjonene startet ikke på null. Matrisene har ulik bredde, så deres
Frobenius-forhold er ikke et mål på aktiveringsforholdet. Den separate,
allerede utførte aktiveringsmålingen viser prisruten på 0,81–1,52 % av GC.

For p=Wx+b med W=b=0 er prisbidraget og gradienten tilbake til denne
grenens input/gate null ved start, mens W selv kan få gradient. MTF har
en direkte vei til entry_q_joint_hidden via mtf_repr, uten de nullstartede
cross_tf/family_tf-residualene. Hele MTF-ruten er dermed ikke stengt.

Dette viser en initial asymmetri og en liten terminal prisrute. Det
beviser ikke hvorfor terminalruten forble liten, at råfeltene har nyttig
retningsinformasjon, eller at annen initialisering vil gi edge.

### Reelt inputskift, med avgrenset forklaringskraft

Den fullførte inputauditen viser GC-D1-historikkdekning på 0–168 steg i
TRAIN mot 169–230 senere, og 17 aktive nåkontekstfelt helt uten senere
dekning. Dette er tilgjengelighet; D1-tallene er ikke volumverdiene.

Rangeauditen finner GC-verdier utenfor TRAIN-området på 86,16 % av Q1-radene.
Eksempel: gc_m5_observed_vwap_usd hadde TRAIN-maks 2 801,405 USD og
Q1-median omtrent 2 907,042 USD. Fold og prefixnormalisering samsvarer med
familiebaseline, men den eldre analysens fulle feature-view/modell er ulik.
Rå domenefunn kan brukes som spor; eldre prognoseundergrupper kan ikke
ommerkes som målinger av dagens modell.

### Tiltak som allerede er prøvd

| Kontroll | Målt senere femminuttersresultat | Begrensning |
|---|---|---|
| Ett til to pass, 60 mål | AUC 0,496961→0,506181; logloss 0,694381→0,696234. Begge parrede AUC-intervaller omfatter null. | Ingen kvalifisert forbedring; samme256 TRAIN bedres mens senere sannsynlighetsscore forverres. |
| 10× femminuttersvekt | AUC 0,496961→0,500775; logloss 0,694381→0,693230. | Under fast prior; parrede AUC-intervaller omfatter null. |
| Ett til tre pass, eldre tre-headmodell | TRAIN AUC 0,515888→0,533534; Q1 0,503021→0,500950. | Eldre funksjon; større TRAIN-fit løste ikke senere rangering. |
| Kronologisk rekkefølge, eldre modell | Q1 AUC 0,503021→0,498501. | Ikke dagens 60-headfunksjon. |
| Konvergert lineært hode på frosset eldre representasjon | TRAIN AUC 0,539421; Q1 0,503360, med dårligere logloss. | Svekker siste lineære hode som eneste forklaring. |
| Nulle de fire momentumfeltene i trent modell | Ingen retningsskifter på 256 TRAIN/1 736 Q1, alle 60 horisonter. | Bare denne smale ruten; ikke all prisinformasjon. |

Disse forsøkene er grunnlag for prioritering, ikke forslag om å kjøre dem igjen.

### Prioriterte, konkrete løsningsforslag

**1. Gjør prisrutens initialisering til et eksplisitt designvalg.** Minste
kandidat er å beholde den eksisterende ordinære Linear-initialiseringen
for de fire prisfeltene, med samme RNG-isolering, i stedet for å nullstille
vektene etter konstruksjon. Bruk eksisterende profil-/kontrakteier og bind
ny funksjonsidentitet; dagens sammenligningsbaseline skal ikke omskrives.
Ingen vilkårlig oppskalering, nye felter eller endrede masker/targets trengs
for dette avgrensede forslaget. Det er en hypotese, ikke en bevist rettelse.

**2. Forbered native prisrelative GC-koordinater som et separat tiltak.**
Gjenbruk kilde-, enhets- og ankerlogikken i gc_relative_price_coordinates_v1,
som allerede brukes av snapshot-ruten. Native adapter må eksplisitt
kartlegge nivåfeltene for hver tidsramme, bevare rådata, observasjonsmasker,
prisanker og basis/kontraktsinformasjon, og måle numerisk roundtrip-feil.
Dette skal ikke blandes med initialiseringsendringen i ett uatskillelig forsøk.
Snapshot-evidensen beviser ikke at native innføring vil virke.

Ingen av forslagene er implementert eller kjørt. Alle genuine familier og
tidsrammer beholdes. En senere godkjenning av modellkjøringer må komme fra
brukeren; da kreves fersk egenbaseline og forhåndsbestemte sammenligninger
på hele populasjonen. Høyere aktivering eller TRAIN-fit alene er ikke suksess.

### Uavklart

Rotårsaken til manglende nyttig fremtidig retningspresisjon er ikke endelig
bevist. Råinputenes potensielle informasjon, faktisk konvergens, historisk
glemsel og effekten av de foreslåtte endringene er fortsatt uavklart.
De 60 binære terminalretningene dekker heller ikke hele M5 Entry-målet
med prisforløp, MAE/MFE og FLAT. Dette scopegapet forklarer ikke i seg
selv svak retningsprediksjon.
Ingen ny modellkonstruksjon, forward/backward, optimizeroppdatering, TEST,
broker eller handel er utført. Kun lagrede arrays/tensorer og kilde er lest.

Maskinlesbar sammenstilling, eksakte artefakthasher og målegrenser:
[DIRECTION_EDGE_CAUSE_REVIEW_20261101.json](DIRECTION_EDGE_CAUSE_REVIEW_20261101.json).

## Implementert etter nytt brukervedtak 01.11

Brukeren vedtok «Lag nytt mål og gjør disse 2». Begge forslag er nå koblet
inn i eksisterende eiere som separate kildevalg. Ingen genuine modellkjøringer,
trening eller omstart er utført; eksisterende launch-porter forblir stengt.

Prisruten har en egen profil med ordinær Linear-initialisering og bevart
RNG-isolering. Den gamle profilen beholder nullstart og gammel identitet.
Valget bindes eksplisitt i feature-view og følger den native konstruktørstien.

GC-varianten konverterer alle 242 kildeklassifiserte nivåer i de fem native
GC-historikkflatene til log(P/A), etter USD-/tick-enhetseieren. Prisankeret
velges fra samme historiske rad/tidsramme med eksisterende kildeprioritet.
Et faktisk USD-anker og dets observasjonsmaske legges til; alle opprinnelige
masker, øvrige felt og rådata beholdes. Feltbreddene blir M5 639→641,
M15 493→495, H1 453→455, H4 431→433 og D1 399→401. De to ankerfeltene
rutes til den eksisterende price_action_candle-familien. Alle åtte familier
og D1/makro beholdes. Arkitekturmetadataens gc_mtf_history.per_tf_widths
angir projeksjonsbredde; rålayoutens hash/bredder gjelder fortsatt fysisk
input. Koordinatvalget er bundet av normaliseringshashen.

Varianten bruker egne statistikker for de transformerte TRAIN-kolonnene
og separate ankerstatistikker tilpasset bare observerte TRAIN-ankre.
Transformasjonen er koblet før native normalisering/projeksjon; fysisk
radutvalg, klokker og target er uendret. Signal- og nåkontekstflatens absolutte
GC-felt er fortsatt urørt. Eksisterende basis-/kildemetadata bevares; ingen
manglende kontrakt-ID eller prisopprinnelsesklokke konstrueres. Feature-view
avviser å slå sammen initialiserings- og koordinatendringen i samme variant.

Første kontroll avdekket en foreldet kildehash i den eksisterende
koordinatkontrakten. Semantisk kontroll viste bare additive BBO-/maske- og
Boolean-endringer i én kildefil; pris-/enhetsprodusenter og originalregistry
var uendret. Historisk snapshot verifiseres nå eksplisitt mot den opprinnelige
Git-kilden, mens native_current får korrekt egen kildeidentitet.

**76 fokuserte syntetiske kontroller bestod:** 14 initialiserings-/viewkontroller
ble gjenbrukt, og 62 GC-/integrasjons-/regresjonskontroller bestod etter rettelsen.
De fem rapporterte roundtrip-tilfellene omfatter 247 observerte syntetiske
celler; største tilbakeberegningsfeil var 0,0000976562501 USD. Ingen av disse
cellene endret F32-verdien ved tilbakekonvertering, og ingen målte
feildekomponeringsgrenser ble overskredet. Dette er ikke en generell garanti
for bitlik invers eller dokumentasjon på markedsdata/GPU-paritet.

Genuine koordinat-/dekningstall, egne ekte normaliseringsstatistikker,
fersk modellbaseline, native konstruksjon, gradienter, resume og senere edge
er ikke undersøkt for de nye variantene. To eksisterende commit-tester som
konstruerer/kjører native modell utelates uttrykkelig under brukerforbudet;
øvrige commit-kontroller beholdes. Ingen eldre engangscampaign er gjenåpnet.

Eksakte kildehasher, testkvitteringer, feltkart og ubeviste grenser:
[DIRECTION_EDGE_DESIGN_IMPLEMENTATION_20261101.json](DIRECTION_EDGE_DESIGN_IMPLEMENTATION_20261101.json).
