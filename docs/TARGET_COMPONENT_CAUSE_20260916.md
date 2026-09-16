# Hva som gjør læringsmålet skjevt — målt 2026-09-16

På512 faktisk trente Entries /2048 parvise Exit-tilstander ble den eksisterende
femstegsberegningen gjenskapt bit-for-bit fra lagrede Q/trace-data. Ingen modell,
forward, backward, ny materialisering eller trening. CPU26,23s. FP64-oppdelingen
av samme mål avvek maksimalt0,00001003Bps fra FP32; selve target-eieren var eksakt.

## Observerte rewards dominerer; lærerens stopp velger horisonten

| Komponent, gjennomsnitt Bps | LONG | SHORT |
|---|---:|---:|
| Første observerte HOLD-reward |0,15376|−0,16968|
| Rewards under frossen lærerpolicy |0,18505|−0,15351|
| Diskontert lærer-bootstrap ved stopp/grense |0,00414|0,08790|
| Faktisk femstegsmål |0,18919|−0,06561|
| Kontrafaktisk likt tilgjengelig femstegsløp med samme bootstrap |0,14875|−0,06561|

Læreren stopper LONG etter ett beregningssteg1880/2048 ganger; SHORT følger
fem2047/2048 ganger. Den siste SHORT-tracen har bare tre tilgjengelige steg.
Likt tilgjengelig femstegsløp endrer fortegnet på648/2048 LONG-mål,31,64%, og
0 SHORT-mål. LONG-målets standardavvik øker5,837→9,204Bps; SHORT er9,210Bps.
Månedlige LONG-gjennomsnitt påvirkes betydelig, selv om totalmidlet bare endres
0,04045Bps. Det viser ulik undervisning, ikke at en tvungen femstegsregel løser
læringen. Kontrafaktisk beregning er ingen ny targetpolicy eller holdetidsgrense.

## Entry får nesten bare første likvidasjonsverdi

På alle512 Entries er LONG-lærerens positive videreverdi nøyaktig0. SHORT får
0,04967Bps i snitt. Første likvidasjonsverdi er−5,38588/−6,31757Bps, og Entry-
målet blir−5,38588/−6,26790Bps. LONG har33 positive mål, SHORT25; disse58 var
allerede positive ved første likvidasjonsverdi. Bootstrap skaper ingen nye
positive Entry-mål i dette utvalget.

Første likvidasjonsverdi kan inkludere både kostnader og markedsbevegelse fram
til første tilstand. Det er ikke riktig å kalle hele beløpet ren spread/kostnad.
Modellen lærer derfor i hovedsak verdien av en svært tidlig avvikling gjennom
Entry-Q, selv om de separate forecast-hodene har andre tidshorisonter.
Dette er en målt undervisningsmekanisme. Det er ikke bevis for at en alternativ
Entry eller lengre holding har positiv forventet kostnadsjustert avkastning.

## Beslutning

Ingen kode-, target-, lærer- eller treningsendring er åpnet. Klippeforsøket
forblir forkastet. Bootstrap-verdienes størrelse forklarer lite av Exit-målenes
spredning; lærerens valg bestemmer derimot hvor mye faktisk prissti som inngår.
Felles femstegsløp ville fortsatt gitt positivt LONG- og negativt SHORT-middel
på disse trente dataene og endrer ikke dagens Entry-anchor-supervisjon.
Det er derfor ikke tilstrekkelig grunnlag for bare å tvinge fem steg eller øke
bootstrap-verdier.

Neste ene måling: undersøk observerte, kostnadsførte framtidsutfall ved
Entry-ankeret separat fra lærerens estimat. Gjenbruk allerede lagrede traces
for samples som kan verifiseres å være samme tilstand som Entry-ankeret;
rapporter tilgjengelighet og utvalgsskjevhet. Ingen nye targets, modeller,
terskelsøk eller trening. Eventuelt manglende ankercache er et datagap som skal
påvises før ny materialisering, ikke fylles med antagelser. Positive framtidige
utfall er alene ikke bevis for en kausalt predikerbar handelsfordel.

Resultat: handover_snapshot/EXIT_PRIVATE_CLIP_20260916/TARGET_COMPONENT_RESULT.json.
Autoritativ mappe: BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/
TARGET_COMPONENT_AUDIT_20260916, med PLAN, OPERATOR,32 batchfiler og RESULT.
RESULT-SHA b34d6f7211b029eb8d4745284eb4bfa0b7a2cfe4e76196172dbdb71fc0ab6b53.
Kilde8691899457d9dfdd2405d3d37e759a25fddc7576. TEST forblir forseglet.
