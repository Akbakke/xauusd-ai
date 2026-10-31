# TRAIN-diagnose av indikatorgrenen — 31.10.2026

Den undersøkte prisgrenen er koblet til modellen, men har liten
aktiveringsstørrelse ved sammenslåingen med GC. Rotårsaken til svak retning
er ikke bevist.

## Målt på ekte data

Én uendret terminal baseline,256fasteTRAINforeldre,16forwards,32autogradreverser,
nulloptimizer. Lagrede terminalprognoser/akser/targets gjenskapt eksakt;
modell/buffere/RNG/modus bevart og alle parametergrader None.
237,366sekunder. Ingen senere rader eller TEST scoret.
Alle26steder×2tap×16batcher haddeikke-nullgradienter; uavhengig aggregatreview
fant0avvik.

| TF | Prisprojeksjonens RMS / GC-projeksjonens RMS |
|---|---:|
| M5 | 1.0754 % |
| M15 | 0.9509 % |
| H1 | 0.9655 % |
| H4 | 0.8096 % |
| D1 | 1.5240 % |

Prisinputenes RMS før projeksjonen lå0,646–0,747; etterpå0,00453–0,00524.
GC-projeksjonene lå0,344–0,597. Dette gjelder fire momentumprisfelters
MTF-rute, ikke all prisinformasjon. Output inkluderer eventuell bias.

Femminuttersgradientens norm ved gatede prisinputs var3,71–6,38% av øvrige59
horisonters samlede komponent; cosinus var negativ påM15/H1/D1. Dette er
én horisont mot59 med felles eligible-denominator, ikke bevis for feil vekting.
Hodeinputforhold0,08736668966 samsvarer numerisk med tidligere måling
(absolutt forskjell2,23e-10).

## Bevist konsistent og grenser

Pris-/GC-projeksjoner summeres: like outputgradienter er en algebraisk
identitet, kontrollert her. Gradientnormer avhenger av skala/dimensjon/lengde;
gradient×aktivering er kun en lokal skaladerivert. Dette er terminale
eval-gradienter, ikke historiske treningsgradienter eller parameteroppdateringer.
Sammen med tidligere svak effekt av å nulle denne ruten gir målingen et
konkret spor om svak funksjonell bruk. Den beviser ikke informasjonsmangel,
kodefeil eller at oppskalering forbedrer senere prognoser.

41syntetiske tester og uavhengig kildegjennomgang bestod.001 stoppet før
modellkonstruksjon på feil par historiske syntetiske testkvitteringer.
Originalen er bevart;002 validerer bare de to opprinnelige115-testkvitteringene,
og øvrige TEST-stier avvises. Ingen omstart. Vakt avsluttet0, topper59/60°C,
137,9W,3959MiB;250Wgrense beholdt.

## Ikke undersøkt eller bevist

Ingen ny generalisering, forbedring etter retrening, lønnsomhet eller
rotårsak til50/50. Tidligere10×femminuttersvekt, ekstrapass og enklere referanser
skal ikke gjentas som nye hypoteser. Fem av ti matchede familievarianter er
ferdige; fem og alle forhåndsbestemte parrede intervaller gjenstår.
Før neste genuine kjøring vurderes funnet mot eksisterende enkle referansers
faktiske profil-/representasjonsdekning. Ingen automatisk modellendring følger.

Evidensrot:
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/UPSTREAM_GRADIENT_20261031_002
Autoritativ status: NEXT_RUN_POLICY.json.
