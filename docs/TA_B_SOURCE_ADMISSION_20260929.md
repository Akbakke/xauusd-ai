# B — kildeport før måling, 29.09.2026

B skal undersøke A pluss høyst 15 felter fra de seks avtalte kildene.
Ingen B-modell eller meravkastning er målt. A-resultatet beholdes uendret.
Denne registreringen gjelder kildehenting og datatilgjengelighet, ikke autorisasjon
til en delvis makromodell eller ny prisbasert fit.

## Kilder og tiltenkte felt

Frys to felt per kilde, totalt 12: DFII10 nivå/endring over 21 D1; DTWEXBGS
loggnivå/loggendring over 21 D1; T10YIE nivå/endring over 21 D1; GLD tonn
loggnivå/loggendring over 21 D1; COT noncommercial netto delt på total open
interest og endring over fire publiserte ukesrapporter; VIX loggnivå/loggendring
over 21 D1. Endringer beregnes på den kausale serien som faktisk kunne vært
kjent, aldri en tilbakefylt sluttserie. COT skal være COMEX gull 088691,
Legacy Futures Only; ingen bytting til andre traderkategorier.

Alle seks kildene inngår i den bestilte B-kontrakten. Hvis en nødvendig kilde
mangler historisk publikasjon/vintage-bevis, kan denne fulle B-kontrakten ikke
måles. Den skal rapporteres som en kildebegrensning, ikke byttes mot en modell
med bare tilgjengelige felter. Manglende data er ikke dokumentert null effekt.

## Manifestbundet ALFRED-henting

Eksekverbar hentekontrakt er configs/research/TA_B_ALFRED_SOURCE_20260929.json.
Den committes før det hentes makroverdier. Bruk eksisterende campaign-eierens
fetch-alfred gjennom capped audit. Hent DFII10, DTWEXBGS, T10YIE og VIXCLS som
nivåer, observasjoner 2009-01-01–2025-12-31, alle annonserte vintager innen
samme datogrenser, format Observations by Real-Time Period / Zipped CSV.
Det offentlige skjemaets egne vintagevalg bindes sammen med POST-bytene.

ALFRED beskriver historiske versjoner med observasjonsdato, verdi og start/slutt
for periodens gyldige dataversjon. Arkivdatoen kan bygge på originalutgiver,
dataleverandør eller først tilgjengelig dato i FRED. Dette må bevares som
begrensning; en release date er ikke et verifisert klokkeslett. Bruk slutten av
releasedatoen i New York som konservativ øvre grense, så én ekstra komplett
kanonisk XAU-D1-periode før input kan brukes.
[ALFRED-hjelp](https://alfred.stlouisfed.org/help),
[formatbeskrivelse](https://alfred.stlouisfed.org/help/downloaddata).

Skjemametadata viste første vintage 2005-10-12 for DFII10, 2019-02-04 for
DTWEXBGS, 2014-01-27 for T10YIE og 2010-11-22 for VIXCLS. Dette er metadata,
ikke godkjent dekningsbevis. Spesielt kan dagens USD-serie ikke gis til
modellen før 2019 ved å bruke dens senere tilbakeførte observasjoner.
Faktiske mottatte bytes og dataversjoner skal kontrolleres først.

Rå skjema, POST-kropp og ZIP bevares lokalt med SHA-256, størrelser og
kvitteringer. Manglende skjema, timeout, ikke-ZIP, størrelsesgrense eller CRC-feil
er eksplisitt feil for den kilden. Uavhengige kilder kan likevel hentes ferdig.
Ingen kvittering fra hentingen alene godkjenner et predictor-input.
Ingen TEST-priser, gullutfall eller modellfits brukes i kildeporten.

## GLD og COT krever fortsatt bevis

[GLDs offisielle side](https://www.spdrgoldshares.com/usa/gld/) viser et historisk
XLSX-arkiv. Siden skiller beholdning på handelsdato fra barlisten på oppgjørsdato.
Dette dokumenterer ikke historiske publiseringsklokker eller alle dataversjoner.
Arkivet er ikke automatisk et as-of-datasett.

[CFTCs endringsmeldinger](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)
dokumenterer både utsatte publiseringer og revisjoner. I 2018/2019, 2023 og 2025
kunne normal ukerytme ikke brukes ukritisk. Historiske komprimerte og viewable
rapporter kan være reviderte; dato i rapporten beviser ikke den tidligere versjonen.
Direkte HTTP-kontroll av GLD- og CFTC-metadatasidene ga 403 i denne gjennomgangen;
webverktøyet kunne lese primærkildeteksten. Dette er en tilgangsbegrensning for den
kontrollerte ruten, ikke bevis for at data er umulige å skaffe.

## Før en eventuell B-fit

Alle seks kilder må ha godkjente immutable dataversjoner og tilgjengelighetsgrenser.
Deretter bindes faktisk felles populasjon, fryste årsfolds innen 2011–2025, og
komplett B-modellregistrering før utfall beregnes. Samme A-eiere, to horisonter,
lærere, risiko, kostnader og økonomiske effektgrenser gjenbrukes.
B må sammenlignes med A på identiske TRAIN-/evalueringsrader og med identisk
risikobudsjett; fullhistorisk A er ikke en gyldig direkte referanse på redusert B-dekning.
Hypotesefamilie, styrke/MDE og GO/NO_GO/INKONKLUSIV må omfatte B-minus-A.

Hvis kildeporten ikke kan bestå, publiser konkret manglende bevis og la B være
umålt/inkonklusiv med kildebegrensning. Ikke lag fiktive resultater eller inferens.
