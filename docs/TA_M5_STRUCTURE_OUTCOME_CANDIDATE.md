# M5-strukturutfall fra kjente topper og bunner

Status: **kildekandidat med syntetisk kontroll og målt Fold2 TRAIN-dekning;
ikke låst target eller kjøreplan**. Implementasjon:
`gx1/audit/measure_ta_m5_structure_candidate_v1.py`.
M5 er fortsatt beslutningsklokken. Full GC, modellfit, TEST og handel er stengt.
Dette utfallet erstatter verken lokal tilstand S eller nettoendepunkt F.

## Frys det som er kjent ved beslutningen

Ved slutten av en observert M5-bar `t` fryses identitet og pris for siste
bekreftede svingtopp og svingbunn, samt siste lukkede M5-ATR. Bruk den
eksisterende `replay_smc_causal_structure_v2`-eieren. En pivot på bar `j`
er først kjent etter bar `j+3`; uferdige pivoter er ikke input ved `t`.
Manglende nivå, ATR eller lukkede kildebarer får egen maske.

## Observer det neste strukturparet

Finn første *nye* bekreftede M5-topp og første *nye* bekreftede M5-bunn med
pivotbar etter `t`. Begge må modnes innen samme TRAIN-folds deklarerte
målgrense (eksklusiv). Bevar pivotbar og bekreftelsesbar hver for seg. Sammenlign de to
pivotprisene med nivåene fryst ved `t`; senere oppdaterte SMC-nivåer får ikke
endre sammenligningsgrunnlaget.

Lagre signerte avstander `(neste_topp - fryst_topp)/ATR(t)` og
`(neste_bunn - fryst_bunn)/ATR(t)`. Avled nivåtreff som topp `>=` fryst
topp og bunn `<=` fryst bunn. Registrer separat om pivotbarens wick går
forbi nivået og lukker tilbake, eller lukker på andre siden. Likhet,
wick-avvisning og sluttprisbrudd må ikke slås sammen. Begge nivåer kan
bli truffet. Hvis samme bar er både neste topp og neste bunn, behold
begge treff, men la parretningen være ukjent siden rekkefølgen i baren
ikke er observert.

Først når begge nye pivoter er bekreftet kan paret beskrives som høyere
topp/høyere bunn, lavere topp/lavere bunn, blandet eller likt. En ren
strukturell vending kan *undersøkes* når et tidligere rent OPP- eller
NED-par etterfølges av et rent motsatt par. Dette er ikke automatisk
framtidig S-side, Entry-signal eller økonomisk Edge. Behold mellomtilstand
og størrelse/varighet; ikke lag en hard UP/DOWN-etikett av blandede par.

## Sensur og kontroll før modell

Ukjente M5-klokkebrudd mellom fryst nivås bekreftelse og beslutning eller
etter beslutning, ufullstendig bekreftelsesvindu og TRAIN-foldens målgrense
sensurerer berørt komponent. En markedsstengning kan bare passeres når hele
intervallet med manglende M5-barstarter ligger i en eksisterende, eksakt
verifisert lukningsautoritet; ellers sensureres bruddet. ATR-, frystnivå-
og framtidspivotmasker beholdes separat. Manglende neste pivot er ikke «ingen treff»
eller FLAT. Ingen maksimal holdetid eller resultatutvalgt ATR-terskel
innføres. Lesende TRAIN-kontroll må vise dekning, modningstid, klasse-
fordeling, samtidige treff og eksempler på faktiske M5-forløp før dette
kan bindes som supervisjon. Fold2 TRAIN-lesningen er nå fullført:
47 567/47 784 beslutningsbarer har begge neste pivoter bekreftet, og
47 195 har entydig parklasse (OPP 18 043, NED 14 514, BLANDET 14 581,
LIKT 57). De 589 ukjente er 217 ufullstendige par og 372 topper/bunner
på samme bar med uobservert intern rekkefølge. Se
`docs/TA_M5_STRUCTURE_FOLD2_AUDIT_RESULT_REVIEW_20261116.json` for eksakt
kilde- og kjøringsbinding. En på forhånd låst SMC-persistensregel scoret
8 253/23 302 (35,42 %) mot fryst TRAIN-prior 8 936/23 302 (38,35 %)
på senere september–desember; se
`docs/TA_M5_STRUCTURE_PERSISTENCE_RESULT_REVIEW_20261116.json`.
Dette kvalifiserer ikke regelen eller en modellfit. Sammenlign senere enhver
prognose med fryst TRAIN-prior, reelle M5-vendinger og alle fem TF-retninger. Økonomisk Edge
krever i tillegg egne kost- og posisjonsmålinger; denne definisjonen
beviser ingen verdi alene.
