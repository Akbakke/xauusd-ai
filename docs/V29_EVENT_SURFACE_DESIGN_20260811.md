# Kodebundet eventdesign

Filnavnet er beholdt fordi de eksisterende, hash-bundne feature-eierne refererer
det. Dette er gjeldende designgrenser, ikke historisk framdrift eller kjøreordre.
Feltlister, dimensjoner, skjemaer og parametre eies alltid av kildekoden.

## 1. Ett horisontalt nivåregister

gx1/features/level_registry_v1.py eier vedvarende horisontale nivåidentiteter,
touch-/reaksjonsminne og kausale break-/retesthendelser. Bekreftede pivoter,
sesjonsankre og likviditetsnivåer har hver sin navngitte opprinnelse;
et register kan ikke introdusere en ekstra detector eller retningsoverstyring.
Prisavstand, alder, pending retest og sided hendelsesinformasjon er evidens
for den lærte modellen, ikke ferdige stemmer eller handelsregler.
Livssyklus, oppdateringsrekkefølge og feltorden leses fra den eksisterende eieren.

TRAIN-fittede toleranser bindes med eksakte grenser, kilde/pair, klokke/lane,
utvalgsstøtte og immutable parametre. VAL, TEST og serve refitter aldri.
Pris-geometriske distanser kan ikke erstattes av terskelvotes eller parkerte nuller.

## 2. Separat skrålinje-/kanalregister

gx1/features/trendline_registry_v1.py eier skrå linjer/kanaler med immutable
ankre og kausal bekreftelses-/break-/retesttilstand. Dette er en annen
geometrisk autoritet enn horisontale nivåer, ikke en separat Exit-modell.
chart.geomline_*-rutingen kommer fra den eksisterende feature-layer-eieren.
Ingen etterpåklok pivot, framtidig bekreftelse eller forming HTF-candle brukes.
Fitted parametre er lane-korrekte og krever samme fit-/serve-provenans.

## 3. Eventprimitiver og routing

Featureformler beregnes på hver native lukkede klokke gjennom sine eksisterende
eiere. M5 Entry og M1 Exit bruker samme kontrakter, ikke kopierte/resamplede
indikatorverdier. HTF bruker bare lukkede M15/H1/H4/D1-candles og korrekt M5-kontekst.
Felt-tupler eies av produsentene; entry_model_native_feature_layers_v1
sekvenserer dem, og entry_specialist_feature_groups_v1 eier specialist-rutingen.
Alle åtte familier beholdes; ingen håndlaget confluence eller post-model filter.

Hendelsesflagg kan disambiguere en faktisk off-event-null. Manglende hendelse/
historikk eller kausal warmup kan ikke fylles med en oppfunnet «nøytral» verdi.
Bekreftelses-, elapsed- og alderklokker skal beskrive det faktisk observerte
bar-/wallclock-domenet. Tilstand må være kausal og bevart over kronologiske chunks.
Sweep-ankret AVWAP er prisoppdateringsvektet markedsevidens, ikke utført order flow.

## 10. Presence-mask og metningsgrense

Trendline-eierens beholdte kildekommentar bruker dette avsnittet til å begrunne
at redundante presence-masker ikke gjeninnføres som ekstra markedsevidens.
Tidligere måling på en supersedert D1-akse er ikke nåværende liveness-/edgebevis.
Aktuelle geometriske carriers og ukjent/warmup-semantikk eies av dagens kode;
saturation, klipping, aliases og døde required fields må kontrolleres på den
faktiske bundne populasjonen, ikke antas ut fra et gammelt feltantall.

## Bevisgrense

Kausalitet, kjent referanseregning, chunk-paritet, feltorden/aliaser,
lane-/clockbinding og liveness kontrolleres av eksisterende featuretester og
inputauditer. Disse viser ikke lært verdi, generalisering eller lønnsomhet.
Endret ONLINE-funksjon krever ny initialbaseline og bundlet train/serve-paritet.
Gjeldende data-/launchomfang står i NEXT_RUN_POLICY.json.
