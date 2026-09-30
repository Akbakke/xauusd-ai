# B: undersøkelse av historiske dataversjoner — 30.09.2026

B er fortsatt umålt. Vi har nå hentet og kontrollert ekte historiske GLD- og
COT-filer fra Internet Archive. Det er et konkret fremskritt fra bare dokumenterte
nettadresser. Vi har ennå ikke kvalifisert den sammenhengende historikken som
full B trenger, og ingen kilde er godkjent som modellinput.

B beholder de seks avtalte kildene og 12 feltene i
[opprinnelig kildekontrakt](TA_B_SOURCE_ADMISSION_20260929.md). A/C og tidligere
B-resultat er bevart. Manglende datagrunnlag sier ingenting om B sin effekt.

## Hva som faktisk er kontrollert

| Kilde/rute | Målt resultat | Hva det beviser |
|---|---|---|
| GLD, arkivkopi 08.07.2019 kl. 14:45:03 UTC | HTTP 200, 487 292 bytes. 3 814 unike, sorterte datarader 18.11.2004–05.07.2019; 3 680 med positiv, numerisk tonnverdi og 134 uten slik verdi. | En historisk versjon av den offisielle GLD-filen kan hentes. Den beviser ikke at alle verdiene var kjent på sine observasjonsdatoer. |
| COT, arkivkopi 07.04.2019 kl. 16:45:52 UTC | HTTP 200, 6 020 bytes. COMEX Gold 088691, Futures Only, posisjonsdato 02.04.2019, med noncommercial- og open-interest-felt. | Riktig rapporttype finnes i arkivet. Denne kopien er neste ukes rapport, ikke originalen for 26.03.2019. |
| COT-indeks for samme nettadresse, 01.03–07.04.2019 | Én arkivkopi, datert 07.04; grensen på 200 treff ble ikke nådd. | Den undersøkte adressen og perioden gir ikke de manglende ukesversjonene. Dette er ikke en påstand om alle CFTC-adresser eller andre arkiver. |
| GLD-arkivindeks | Timeout både for den lange perioden og en avgrenset juli 2019-spørring. | Dekningen er ukjent. Timeout er ikke bevis for at arkivkopier mangler. |
| ALFRED, DFII10 juni 2025 | Skjemaet ble hentet på Mac; 20 annonserte vintager valgt. POST fikk timeout etter henholdsvis 20 og 60 sekunder. Ingen makroverdier mottatt. | Mac løser skjema-GET, men har ikke løst selve nedlastingen. |
| Macrobond | Offentlig dokumentasjon bekrefter historiske dataversjoner og API for revisjonstidspunkter. | En mulig leverandørrute; de eksakte GLD-/COT-seriene og historiske dekningen er ikke kvalifisert. |

Arkivtidspunktet er en konservativ grense for når innholdet kan dokumenteres
kjent. Det er ikke det opprinnelige publiseringstidspunktet. En GLD-fil fanget i
juli 2019 kan derfor ikke uten videre gis til modellen i 2004–2018, selv om den
inneholder observasjoner fra disse årene. Eventuell bruk krever også den avtalte
ekstra komplette XAU-D1-perioden.

Internet Archives Availability API returnerer nærmeste tilgjengelige kopi, som
kan ligge etter ønsket dato. Begge COT-oppslagene, for 30.03 og 04.04, returnerte
07.04. Denne forskjellen er kontrollert, ikke tolket som to ulike versjoner.
[Internet Archives API-beskrivelse](https://archive.org/help/wayback_api.php),
[CDX-indeksens felter](https://github.com/internetarchive/wayback/blob/master/wayback-cdx-server/README.md).

## CFTC: en konkret grunn til at publiseringsklokken må bevises

Dagens FAQ sier at historiske data ikke oppdateres etter publisering.
Samtidig dokumenterer en melding fra 03.04.2019 tillegg i rapporten med
posisjonsdato 26.03, publisert 29.03; COMEX Gold 088691 står uttrykkelig på listen.
FAQ-en alene kan dermed ikke garantere originalversjoner for hele historikken.
Vi har ikke gjenfunnet originalen i det undersøkte arkivutvalget.

CFTC oppgir normalt fredag ettermiddag som publisering av tirsdagens posisjoner,
men det finnes både helligdagsavvik og utsatte publiseringer. Posisjonsdato,
publiseringsdato, revisjonsdato og arkivdato må skilles.
[CFTC FAQ](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm),
[daterte endringsmeldinger](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).

## GLD: riktig innhold, fortsatt manglende versjonsdekning

Det gjenfunne arkivet har et særskilt tonnfelt for beholdningen kl. 16:15 New York.
Dagens offisielle side skiller dessuten beholdning på handelsdato fra barlisten
på oppgjørsdato. Barlisten eller en samlet serie for alle gull-ETF-er er derfor
ikke en automatisk erstatning. Manglende tonnverdier skal ikke fylles med
oppdiktede tall. Før full B trengs flere daterte versjoner og målbar dekning,
med eksplisitt behandling av datagap.
[GLDs offisielle informasjon](https://www.spdrgoldshares.com/usa/gld/).

## ALFRED: liten rettelse, uløst nedlastingsrute

Det faktisk mottatte HTML-skjemaet inneholder submit-knappen
form[download_data]. Den tidligere POST-kroppen manglet dette feltet.
Henteeieren sender nå feltet, og de to eksisterende, fokuserte ALFRED-testene
besto gjennom capped audit. Samme lille datasnitt fikk likevel timeout med
60 sekunders grense. Rettelsen gjenoppretter forespørselens samsvar med skjemaet;
den har ikke dokumentert eller løst årsaken til timeouten.

ALFRED beskriver dataverdier med observasjonsdato og historiske gyldighetsperioder.
Den offisielle FRED/ALFRED-API-ruten støtter slike forespørsler, men krever en
registrert API-nøkkel. FRED_API_KEY og ALFRED_API_KEY var ikke satt i de to
kontrollerte prosessmiljøene. Det sier ikke at brukeren mangler en nøkkel andre
steder; ingen nøkkelring eller bredt privat fillager er undersøkt.
[ALFRED-formatet](https://alfred.stlouisfed.org/help/downloaddata),
[API-observasjoner](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[API-nøkler](https://fred.stlouisfed.org/docs/api/api_key.html).

Metadata fra forrige kildeaudit viser første DTWEXBGS-vintage 04.02.2019.
Med den ruten kan ikke full B tilbakefylles gjennom hele A-perioden 2011–2025.
Felles A/B-periode må bestemmes fra faktisk godkjent dekning, warmup og
publikasjonslag før en fit. Deretter må A sammenlignes på samme TRAIN- og
evalueringsrader som B.

## Konkret krav til en mulig leverandør

Macrobond oppgir at vintagefunksjonen krever en relevant lisens, og at ikke alle
serier eller perioder har versjonshistorikk. API-et har get_revision_info og
get_one_vintage_series. Dokumentasjonen advarer også om at visningen kan vise
første tilgjengelige data når den forespurte versjonen mangler; slik respons
må avvises for tidligere beslutningstidspunkter. Eksakte serie-ID-er, dekning,
pris og tilgang for våre to kilder er fortsatt ubekreftet.
[Lisens og dekning](https://help.macrobond.com/tutorials-training/macrobond-analysis-user-guides/2-finding-data/browsing-data-tree/the-data-tree-structure/databases-in-macrobond/),
[vintagebegrensninger](https://help.macrobond.com/tutorials-training/macrobond-analysis-user-guides/3-analyzing-data/analysis-tree/using-the-series-list/vintage-data/),
[API-funksjoner](https://macrobond.github.io/macrobond-data-api/).

Før en eventuell avtale bør en dataprøve dokumentere:

1. GLD alene: beholdning i tonn på handelsdato, ikke pris, AUM eller alle gull-ETF-er.
2. CFTC 088691, Legacy Futures Only: noncommercial long/short og total open interest.
3. Opprinnelige og senere versjoner, med kilde-ID, posisjons-/observasjonsdato,
   faktisk tilgjengelighet og revisjonstid. Første leverandørinnlasting må skilles
   fra opprinnelig publisering.
4. Både før- og etterversjonen rundt gullrevisjonen 03.04.2019, samt håndtering
   av publiseringsforsinkelser. Dagens sluttserie med påført standardlag er utilstrekkelig.
5. Full liste over datagap, første historiske versjon og eksportmulighet som
   kan bevares og hashes for reproduserbar offline forskning.

Dette er en forberedt kvalifikasjon, ikke en sendt leverandørhenvendelse eller
et kjøpsvedtak. Det anbefales ikke å kjøpe et generelt «historisk datasett»
uten dette beviset.

## Beslutning og neste steg

Kildeundersøkelsen er gjennomført. Det finnes et verifisert gratis arkivspor,
men full B er fortsatt ikke klar til måling. Neste nødvendige arbeid er tilgang
til makroversjonene og dokumentert GLD/COT-dekning. Ingen ny modellmåling er
meningsfull før alle seks kilder kan innlemmes etter den avtalte kontrakten.
En API-nøkkel alene løser ikke GLD/COT-delen.

Når dette er dokumentert, bindes faktisk felles populasjon og B-minus-A-testen
før utfall beregnes. Samme modeller, kostnader og risikobudsjett gjenbrukes.
Ingen redusert firekildemodell erstatter full B automatisk.

## Etterprøvbarhet

[Maskinlesbar kontroll](TA_B_SOURCE_INVESTIGATION_20260930.json) binder tre
manifester, kildecommits, terminaler og alle 25 opprinnelige forespørselsartefakter.
Manifestene var committet før henting, og identiske bytes er kontrollert etter
overføring fra Mac-transporten til CURRENTs runtime. De to snapshot-filene og
tidligere feil er bevart. En arvet peker i tredje manifests metadata viser
første manifest; korrekt umiddelbar forgjenger er eksplisitt rettet i rapporten,
uten å omskrive det frosne manifestet. Hente- og kildehashene var riktige.

Native trening, TEST, handel og spending er fortsatt stengt. Ingen økonomisk
effekt, ny modell eller forbedret lønnsomhet er målt i denne undersøkelsen.
