# M5-strukturutfall fra kjente topper og bunner

Status: **forslag til separat TRAIN-kontroll, ikke låst target eller kjøreplan**.
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
målgrense. Bevar pivotbar og bekreftelsesbar hver for seg. Sammenlign de to
pivotprisene med nivåene fryst ved `t`; senere oppdaterte SMC-nivåer får ikke
endre sammenligningsgrunnlaget.

Lagre signerte avstander `(neste_topp - fryst_topp)/ATR(t)` og
`(neste_bunn - fryst_bunn)/ATR(t)`. Avled nivåtreff som topp `>=` fryst
topp og bunn `<=` fryst bunn. Registrer separat om pivotbarens wick går
forbi nivået og lukker tilbake, eller lukker på andre siden. Likhet,
wick-avvisning og sluttprisbrudd må ikke slås sammen. Begge nivåer kan
bli truffet; ukjent rekkefølge inne i en bar forblir ukjent.

Først når begge nye pivoter er bekreftet kan paret beskrives som høyere
topp/høyere bunn, lavere topp/lavere bunn, blandet eller likt. En ren
strukturell vending kan *undersøkes* når et tidligere rent OPP- eller
NED-par etterfølges av et rent motsatt par. Dette er ikke automatisk
framtidig S-side, Entry-signal eller økonomisk Edge. Behold mellomtilstand
og størrelse/varighet; ikke lag en hard UP/DOWN-etikett av blandede par.

## Sensur og kontroll før modell

Kildegap, ufullstendig bekreftelsesvindu og TRAIN-foldens målgrense
sensurerer berørt komponent. Manglende neste pivot er ikke «ingen treff»
eller FLAT. Ingen maksimal holdetid eller resultatutvalgt ATR-terskel
innføres. Lesende TRAIN-kontroll må vise dekning, modningstid, klasse-
fordeling, samtidige treff og eksempler på faktiske M5-forløp før dette
kan bindes som supervisjon. Sammenlign senere enhver prognose med fryst
TRAIN-prior, reelle M5-vendinger og alle fem TF-retninger. Økonomisk Edge
krever i tillegg egne kost- og posisjonsmålinger; denne definisjonen
beviser ingen verdi alene.
