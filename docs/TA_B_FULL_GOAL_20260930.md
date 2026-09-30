# Aktivt mål: full B og videre kvalifisert modelløp — 30.09.2026

Brukeren har uttrykkelig bestilt full implementasjon, deretter build/trening og
til slutt VAL, endelig TEST og backtest over alle år med gyldige data. Målet er
ufullført og nå blokkert på dokumentert kildebevis. Tidligere A/B/C-plan var
avsluttet; dette er en ny videreføring.
En kildebegrensning alene fullfører ikke dette målet.

## Rekkefølge og ferdigkriterier

1. **Kvalifiser alle seks B-kilder.** Gjenbruk de fire kontrollerte ALFRED-arkivene.
   Lukk dokumenterbar GLD-tonnasje og COT 088691 Legacy Futures Only med faktisk
   tilgjengelighet og historiske versjoner. Ingen kildebytte eller tilbakefylling
   av reviderte sluttserier.
2. **Implementer hele B hos eksisterende eiere.** De tolv avtalte makrofeltene
   legges til de sju A-feltene. Bevis kausalitet, manglendeverdibehandling,
   publikasjonslag og identisk A/B TRAIN-/eval-populasjon. Bevar native familier.
3. **Mål den registrerte B-minus-A-kontrakten.** Gjenbruk modell-/kostnads-,
   baseline- og inferenseiere. Frys faktisk populasjon, årsfolds, konstanter,
   hashbindinger og beslutningsregel før utfall undersøkes eller fits kjøres.
   A/C gjentas ikke blindt; en nødvendig matchet A-referanse til B er eget
   eksplisitt sammenligningsgrunnlag.
4. **Ferdigstill bygge- og treningskontrakten.** Krev målt verdi før omfattende
   trening. Bind mål/horisont, native innføring, datasett, datadeling, ressursprofil,
   kausalitet, paritet og alle nødvendige kontroller. Oppgi eksplisitt dersom
   økonomiske porter ikke gir grunnlag for denne overgangen.
5. **Kjør kvalifisert build og trening.** Bruk eksisterende eiere, prosjektlås,
   capped wrapper og vakter. Ingen blind utvidelse eller svekkelse av porter.
6. **VAL, endelig TEST og backtest.** Frys modell og evalueringsprotokoll før
   TEST åpnes. TEST brukes én gang til endelig vurdering, aldri tuning.
   Backtest alle år med gyldig, deklarert dekning, med kostnader og åpne posisjoner.
   Historiske utviklingsår skilles fra kronologisk OOS og forseglet TEST;
   resultatene får ikke alle merkelappen urørt OOS.

Brukerens bestilling autoriserer nødvendig arbeid og disse trinnene i denne
rekkefølgen. Den er ikke bevis for læring, data eller klarhet. Native trening og
TEST er fortsatt stengt i dagens maskinpolicy fordi foregående porter ikke er
bestått. Kjøringspolicy kan først flyttes sammen med dokumentert oppfylt kontrakt.
Ingen live/paper, spending eller eksterne leverandørmeldinger er bestilt.

## Verifisert utgangspunkt

CURRENT 74ba5f64, rent tre og ingen native prosess ved overtakelse.
Fire makroarkiver er kontrollert; full B er umålt. Første dollarvintage er
04.02.2019, så alle år betyr alle faktisk gyldige år, ikke syntetisk dekning av
hele prisarkivet. Ingen modellgjennomføring eller bygge-/treningsklarhet påstås.

Et nytt avgrenset metadataoppslag er bundet i
configs/research/TA_B_FULL_GOAL_SOURCE_20260930.json etter nesten tre timer uten
arkivkall. Resten av batchen stoppes ved første HTTP429; fullførte råfiler hentes
ikke på nytt. Metadata er ikke predictor-admission.

## Første videreføring

Arkivtjenesten ga HTTP429 på første forsøk 10:51:16 UTC; to gjenstående
forespørsler ble hoppet over av den kontrollerte eieren. Ingen nye rådata ble
hentet. Ingen ny arkivprøve gjentas automatisk. Den frosne forespørselens felt
prior_last_request_utc oppga 07:51:22Z; tidligere RESULT.json viser eksakt
07:51:28.141925Z. Feltet var bakgrunnsmetadata, ikke tilgjengelighetsinput.
Korrekt målt opphold før den nye forespørselen var 10 788,14 sekunder.

Makroklokken implementeres og kontrolleres uavhengig av den blokkerte
GLD/COT-tilgangen. Dette er åtte komponentfelt, ikke en redusert B-modell.
Ingen XAU-priser eller utfall leses i denne komponentforberedelsen.

## Nåstatus etter første implementasjonsbølge

Makroklokken er implementert og tre fokuserte tester består. Den bruker
publikasjonsdagens slutt i New York og én komplett senere kanonisk XAU-D1-periode.
En senere revisjon kan ikke endre tidligere input. Manglendeverdirevisjoner
fjerner den aktuelle observasjonen fra kjent numerisk tilstand. Proveniens
bevares ved hver beslutning; ingen antatt fredagspublisering eller framtidig
revisjonsslutt får inputautoritet.

Kjøring på ekte, hashbundne arkiver stoppet på TA_B_FUTURE_OBSERVATION.
Alle fire kilder ble undersøkt for samme feilklasse: DFII10, DTWEXBGS og T10YIE
hadde null slike rader. VIX hadde 26 numeriske observasjonsversjoner der
realtime_start ligger før observasjonsdatoen. Ingen av de 26 verdiene er lik
siste kjente numeriske observasjon ved den oppgitte vintagen. De kan derfor
ikke forklares som ren videreføring av sist kjente verdi på denne evidensen.

Første berørte observasjon er 08.10.2018, siste 01.09.2025.
Datoene og verdiene er ikke omskrevet; ingen rad er stilletiende utelatt for
å få et grønt resultat. Den avviste kjøringen, originalfilene og begge
diagnosene bevares. Ingen komponentpanel ble publisert.

Den tidligere grønne ZIP-/skjema-/intervallkontrollen gjelder fortsatt akkurat
de kontrollene. Den beviste ikke at alle observasjoners publiseringsklokker var
kausalt gyldige. Denne nyere kontrollen sperrer VIX videre inntil kildebevis
eller en eksplisitt og dokumentert datakontrakt løser avviket.

[Maskinlesbar status](TA_B_FULL_GOAL_20260930.json) binder kvitteringer, hasher og
testresultater. Full B-kode er **delvis implementert**, B-minus-A er **ikke kjørt**,
og build/trening/VAL/TEST/backtest er **ikke klare**. Ingen ny markedsutfallsmåling,
fit eller TEST-tilgang er gjort i denne bølgen. Oppgaven/målet er fortsatt aktivt.

Uavklarte datagrenser er nå GLD, COT og VIX-klokken. Leverandørtilgang er spurt om;
ingen leverandør er kontaktet og intet kjøp er foretatt. Kilder som kaller data
historiske eller point-in-time må fortsatt dokumentere akkurat våre felter,
versjoner og tilgjengelighet.

Primærkilder revidert i denne bølgen:
[ALFREDs definisjon av gyldighetsperioder](https://alfred.stlouisfed.org/help/downloaddata),
[CFTCs publiserings- og revisjons-FAQ](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm),
[CFTCs dokumenterte unntak](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)
og [GLDs offisielle holdings-/arkivside](https://www.spdrgoldshares.com/usa/gld/).
Ingen av disse gir ennå komplett GLD/COT-versjonsdekning for vårt datasett.

## Andre videreføring: kildeformatkontroll og presisering av hva som er ferdig

To ALFRED-eksporter ble forhåndsbundet i commit a55f92e6: periodeformat og
vintagematrise, observasjoner 01.–08.07.2025 og vintager 03./07.07.2025.
Alle ti sammenlignede celler er like. Begge offisielle formater inneholder
en numerisk 04.07-observasjon i 03.07-vintagen. Vår periodeparser skapte
dermed ikke dette konkrete avviket. Årsaken og en forsvarlig datokontrakt
er fortsatt uavklart; dette beviser ikke samme formatlikhet for alle 26 avvik.
Ingen ny serie eller rad er godkjent som modellinput.

Første lokale diagnoseaudit feilet fordi README-parseren også tok med
strekseparatoren. Originalen er bevart. En separat audit med full ISO-dato-
gjenkjenning besto gjennom capped wrapper. Ingen ny nedlasting eller omskriving
av rådata var nødvendig. Resultat, kode- og filhasher er bundet i JSON-statusen.

**Tekniske indikatorer og nytt læringsmål må skilles fra B-kildene.**
Den fullførte v37-inputforberedelsen har 242 signalfelt, alle åtte familier,
652 552 TRAIN-rader og 70 880 VAL-rader. Dette bygges ikke på nytt.
A-forskningen testet sju avtalte D1-felt på 15 kronologiske årsfolds 2011–2025.
Fasit var faktisk senere prisendring ved neste utførbare midpris, normalisert
med kjent ATR14, over 20 og fem observerte D1-biner. Exit-lærerens estimat
var ikke fasit i A. Ridge- og HGB-målingene var INKONKLUSIVE uten GO.

Dette er ikke det samme som å ha trent hele v37-modellen på et nytt native mål.
Den nye native mål-/horisontkontrakten og trening er fortsatt ikke innført.
GLD/COT-versjonsdekning stopper det vedtatte fullstendige makrotillegget B;
den stopper ikke beregning av de eksisterende OHLC-indikatorene.
Tidligere SMC-/klokke-/datasettreparasjoner og rettet risiko-, kostnads- og
referanseregnskap beholdes. De gjør inputs og målinger mer pålitelige,
men er ikke alene dokumentasjon på bedre handelsbeslutninger.

## Tredje implementasjonsbølge: felles A/B-kjerne, fortsatt manglende kildebevis

Eksisterende campaign-eier har nå en streng sammenkobling på identiske
session_open-/decision_time-rader og nøyaktig de tolv navngitte B-feltene.
Ufullstendig feltsett, ulik klokke og uendelige verdier avvises. Manglende
publikasjoner kan gi NaN og en felles utilgjengelig rad, aldri oppdiktet input.

A (sju felt) og B (nitten felt) tilpasses på samme TRAIN-/evalueringspopulasjon.
Alle opprinnelige D1-rader beholdes når h20/h5-fasiten beregnes; et manglende
makroinput kan dermed ikke forkorte målets faktiske horisont. De samme indre
splittene, purge-posisjonene og kjente TRAIN-utfallene brukes. Posisjonslister
hash-bindes per fold, og den lærte konstanten må være identisk i begge armer.

Samme portefølje- og inferenseier vurderer B-ridge mot matchet A-ridge og
B-HGB mot matchet A-HGB, i tillegg til LONG, konstant, trend og kjøp-og-hold.
Alle 120 endepunkter inngår i samme erklærte familie med felles resampling;
primærbeslutningen krever også merverdi over tilsvarende A. Rå kjøp-og-hold
og h5 er fortsatt diagnostikk. Ulike prognosepopulasjoner avvises.

Fem fokuserte tester besto gjennom capped audit. De dekker nye mekaniske
egenskaper og eksisterende A-integrasjon/terminalpublisering. To ble gjentatt
etter rettelse av armnavnet i logging og eksplisitt avvisning av tom klokke.
Dette bruker bare syntetiske data. Ingen markedsutfall, ekte B-fit eller TEST
ble lest. Hele A/C og inputbyggingen ble ikke kjørt på nytt.

**Implementasjonen er fortsatt delvis.** Kildetilpasset GLD/COT-versjonsimport,
komplett godkjent komponentpanel, faktisk B-forhåndsregistrering og run-b-
integrasjon er ikke ferdige. Vi lager ikke en generisk leverandøradapter rundt
en ubestemt dataleveranse eller en godkjenningskvittering uten kildebevis.
Nytt native mål, build/trening og endelig evaluering er fortsatt ikke klare.

Samme kildebegrensning er bekreftet gjennom minst tre målrettede arbeidsrunder.
Uavhengig implementasjonsarbeid i denne bølgen er ferdigstilt og bevart.
Målet markeres nå blokkert, ikke fullført, i påvente av dokumenterbar
GLD/COT-versjonshistorikk og avklaring av VIX-klokken. Ingen leverandørtilgang
er oppgitt, ingen henvendelse er sendt og intet abonnement er kjøpt.
Den komplette sekskilde-kontrakten og det videre modellmålet beholdes.
