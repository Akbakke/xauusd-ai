# Aktivt mål: dokumentert automatisk XAUUSD-bot — 01.10.2026

<!-- GX1_CURRENT_RESTART_POINTER -->
## Statusoppdatering 05.10.2026

Faktisk indeksbundet 254-felts TRAIN/VAL-featurekilde er kontrollert med
exit 0, uendret kilde, null TEST og null modell-/optimizerarbeid.
INDEX_FEATURE_SOURCE_REVIEW_001 er konsumert. Neste port er én
forhåndsregistrert workload-matchet samplerbenchmark, så immutable
koordinater og fersk nullstegs initialmåling. Se RESTART_POINT_20261005.md.

De to godkjente OANDA practice-lesekallene er fullført. Ingen ordre,
transaksjonsoppslag eller nye broker-kall er utført etter engangskontrollen.
258 historiske fyll og lagrede finansieringsobservasjoner er bevart.

Dagens instrumentvilkår er -0,0569 for LONG og +0,0323 for SHORT.
Kostnadskoden hadde hardkodet de gamle ratene. Eksisterende eier binder nå
rater til hvert artefakts eksakte broker-evidens og bruker samme metode:
negative renter belastes, gunstige kreditter klippes til null.
Policy, komponentfakta og parameterautoritet bruker samme bundne tall.
Gamle artefakter kontrolleres fortsatt mot sine opprinnelige kilder.

CURRENT_TERMS_POLICY_001 publiserte en ny, separat prospektiv kostnadspolicy
for 01.06.2011–01.07.2026. Årlig finansieringskostnad er 0,0569 LONG og
0 SHORT. Kommisjonsgrunnlag, slippage 2 bps per utførelse, sensitivitet
1/2/4 bps, no-GSLO-policy og null ekstra risikostraff er uendret.
Policyen er forhåndsbundet før native måling og bruker ekte før-TEST-quotes.

36 fokuserte tester består. Kontrollene dekker endrede fortegn/rater,
avvisning av gamle eller underrapporterte kostnader, atomisk publisering
og gammel evidens. Gamle ekte policybytes og den nye publiserte autoriteten
er strengt lest med kildeverifisering. Kilde var uendret under kontrollen.
Etter de to godkjente GET-kallene var nettverk sperret i policykontrollen.

Dette kvalifiserer en prospektiv beregningspolicy, ikke historisk
finansieringsfasit eller økonomisk edge. Neste avhengighet er nye faktiske
TRAIN/VAL-økonomi-/indeksbindinger til denne autoriteten, deretter målt
sampler, koordinater og fersk initialmåling. Ingen indeksbygg eller
modellmåling er gjort i denne bølgen. Native trening og TEST er stengt;
engangs broker-tillatelsen er brukt opp.

## Tidligere målstatus — historikk

Operatøren ba om å gjøre den foreslåtte veien til et aktivt mål og arbeide videre
gjennom alle punktene. Målet er ikke oppnådd. Dette er gjeldende arbeidsrekkefølge;
historiske forsøk er bevis og skal ikke relanseres.

## Oppdatert operativ grense — 02.10.2026

Offline lagring av handelstilstand er rettet og testet med feilinjeksjon.

Før rettelsen ble en katalog eller brutt lenke med forventet tilstandsnavn
tolket som fravær av handel. To samtidige lagringer delte dessuten samme
midlertidige fil og kunne skrive i en allerede publisert tilstand. Tre
målrettede feiltilfeller ble først reprodusert på gammel kilde.

Hver lagring bruker nå en eksklusivt opprettet midlertidig fil med 0600,
før eksisterende fsync og atomisk replace. Ugyldig filtype, brutt lenke og
feil ved filinspeksjon stopper innlesingen; det samme gjelder flytting fra
den pensjonerte plasseringen. En faktisk manglende fil er fortsatt fravær.
Ingen ny ordreautoritet, låseeier eller handelsregel er innført.

136 fokuserte tester består med nettverksadgang sperret i testprosessen.
Kontrollen dekker overlappende lagring, korte writes, skriveavbrudd,
fil-/katalog-fsync og replace-feil. Ved feil er synlig tilstand en komplett
gammel eller ny versjon. Dette er syntetisk feilinjeksjon, ikke målt
strømbrudd eller OS-omstart. Ubrukte close-intent-testhjelpere er fjernet.

OFFLINE_PERSISTENCE_REVIEW_001 sluttet med exit 0 og uendret kilde.
Statisk AST-søk i 265 sporede produksjonsfiler fant ingen direkte kall til
create_market_order, get_order_by_client_id eller get_open_trades.
Full ordrekoordinering, idempotent gjenoppretting og brokeravstemming er
derfor fortsatt ukvalifisert. Hjelpertestene er ikke ende-til-ende botbevis.

Den foreldede next_native_step-kopien i målmetadata er samstemt med
fullført inputbygging. Ingen fullført plan skal relanseres. Neste native
avhengigheter er fortsatt faktiske kostnader/indekser, målt sampler,
koordinater og fersk initialmåling. Det avgrensede brokerspørsmålet er
ubesvart. Ikke legg til flere generelle sikkerhetskontroller uten påvist
feil. Trening, TEST, broker og live/paper er fortsatt stengt.

Kontrollerte PC-omstarter inngår fortsatt i driftshensynet og skal bare
skje etter maskinvid kontroll av jobber, GPU og låser. Ingen PC-omstart ble
gjort i denne kontrollen. Læring, økonomisk edge, train/serve-paritet og
full operativ kvalifisering er fortsatt ubevist.

## Avgrenset første måling: MACRO_CORE

En egen navngitt forskningsarm bruker prisfeltene fra A og seks makrofelt:
DFII10 (nivå og 21 observerte D1-raders endring), DTWEXBGS (loggnivå og endring)
og T10YIE (nivå og endring). DTWEXBGS er bred handelsvektet USD, ikke ICE DXY.
Dette er et eksplisitt eget forsøk, ikke en omdefinering eller fullføring av B.
VIX, GLD og COT legges ikke til denne armen etter at resultater er kjent.
Seks-kilde B beholder opprinnelig mål og dokumenterte kildeblokkeringer.

1. Kvalifiser de allerede mottatte ALFRED-versjonene med
   configs/research/TA_MACRO_CORE_COMPONENTS_20261001.json.
   Bruk bare hash-bundne arkivbytes og de eksisterende XAU-beslutningsklokkene.
   Arkivdatoens slutt i New York er konservativ publiseringsgrense; deretter
   kreves en hel observert kanonisk handelssesjon før første beslutning.
   Ingen bakoverfylling eller historiske sluttversjoner. Bevar både observasjonsdato,
   versjonsdato og første tillatte beslutning. Dette trinnet leser ingen prisutfall.
2. Frys målemanifest etter faktisk målt inputdekning, før fits/utfallsinspeksjon.
   Bruk A's allerede deklarerte mål: observert fremtidig mid-endring delt på
   kausal ATR14 over 20 observerte D1-rader (primært), 5 (diagnostisk).
   Dette er prognosemål, ikke fasit fra en lærer eller en maksimal holdetid.
3. Pris alene og pris+makro bruker identiske kausale ytre og indre TRAIN-rader,
   samme hold-rader, purging, ridge/HGB-konfigurasjon og felles porteføljekapasitet.
   Gjenbruk A's kostnader, finansieringsproxy/zero-sensitivitet og statistikkeier.
   Konstant lært fra samme TRAIN, kausal trend, samme-risiko LONG og kjøp-og-hold
   rapporteres. Matchet A inngår i korrigert familie; ingen sammenligning mot
   gammel A på en lengre eller annen periode. Ingen parameterjakt.
4. Bekreft cached inputidentitet og kausalitet separat. Etter målingen:
   kontroller utfall og porteføljeregnskap fra lagrede bytes, rapporter alle
   år/folds, handler og sluttlikvidering. Gjenbrukt utviklingshistorikk skal
   aldri kalles urørt OOS. Uavklart effekt betyr ikke GO.

## Separat native v38 og senere innføring

Native v38 har nå et fullført og kontrollert datasett med 254 felt.
Inputkontrollene fra 01.10 er bevart; de beviser ikke læring. Før native fits
må eksisterende kontrakteiere binde nye feature-/datasett-/normaliseringsbytes,
ekte mål og sammenligningspopulasjon. Ny initialbaseline kreves ved endret
ONLINE-funksjon. En avgrenset sammenligning må skille TRAIN-tilpasning fra
senere generalisering og observerte kostnadsjusterte utfall. Gjenbruk først
input- og outcome-cacher der kontraktene tillater det. Ingen full epoch/full VAL.

Makro kan innføres i den samme delte Entry/Exit-modellen når den deklarerte
evidensporten er bestått og native mål-/inputkontrakt er bundet. Ingen separat
Exit-modell eller håndskrevet makroveto. Native training_enabled er fortsatt
false mens kontrakt og måleplan mangler; denne planen er ingen launch-oppskrift.

## Godkjenningsbevis før handelsklarhet

- Kronologisk senere, uavhengig generalisering etter fryst modellvalg.
- Positiv kostnadsjustert økonomi og relevant baselinefordel med usikkerhet.
- Eksakte features, normalisering, klokker og handlinger fra samme bundle i
  reell train/serve-paritet; vekthash alene er utilstrekkelig.
- Offline kontroll av ordretilstand, idempotens, gjenstart og brokeravstemming
  med observerbar tilstand og feil-lukket oppførsel. Dette gir ikke brokeradgang.
- TEST forblir forseglet. Live/paper, spending og automatisk promotion er stengt.
  Eventuell endret operativ autorisasjon krever særskilt vedtak.

Én agent og én tung jobb innen CURRENT; alle tunge steg går gjennom capped-eieren.
Målet holdes aktivt ved negative/inkonklusive resultater, men slike armer utvides
ikke med flere forsøk uten ny konkret hypotese. Arbeid videre på uavhengige
punkter og registrer presise blokkeringer.

## Status ved forhåndsregistrering

MACRO_CORE kilde- og sammenligningsmekanikk er implementert i eksisterende
research_ta_campaign_v1-eier. 48 fokuserte syntetiske tester består, inkludert
hele eksisterende A/B/C-testfilen, eksakt matchede rader og fortsatt stengt B.
Kildekontroll på ekte bytes og læringsmåling er ennå ikke kjørt.
Testlogg: /home/andre2/GX1_RUNS/TA_MACRO_CORE_20261001/CODE_REVIEW_001/TESTS.log.

## Kildekontroll fullført, separat måling fryst

Tre kilder er kvalifisert på ekte arkivbytes. En separat verifikator har
kontrollert nivå, endring og valgte historiske versjoner på alle 4518
beslutningsklokker per kilde. 1761 rader har alle seks felt, fra
06.03.2019 til 30.12.2025. USD er begrensningen; ingen eldre verdier bakoverfylles.
Median observasjonsalder for USD er 7,92 kalenderdager etter det konservative laget.
Dette er langsom kontekst. Cachebyggerens daily_panel og gjenværende HTF-definisjoner
har identisk AST med den fullførte A-byggingen; indikator- og klokkeeier er bundet.

configs/research/TA_MACRO_CORE_PREREG_20261001.json fryser den separate målingen.
Årlige folds 2020–2025, felles input- og prognoserader, samme mål/hyperparametre,
kostnader og 120 sammenligningsendepunkter er bundet før fits.
49 fokuserte tester består, inkludert komplett syntetisk kjøring med
receipt-/cache-/populasjonskontroll. Ingen markedslæring er målt ennå.
Kilde- og klokkebevis: docs/TA_MACRO_CORE_RESULT_20261001.json.

## Måling fullført — videre arbeid

MACRO_CORE MEASUREMENT_001 er komplett og kontrollert. Begge learnerne er
INKONKLUSIV, uten grunnlag for native makroinnføring. Se
[full resultatrapport](TA_MACRO_CORE_RESULT_20261001.md) og JSON-bindingsrapporten.
Kildekontroll og den matchede målingen er ferdige delmål; ikke gjenta dem.
Neste aktive delmål er å binde native v38s faktiske læringsmål og uendrede
gjenbrukbare inputdeler, deretter en konkret avgrenset læringskontrakt.
Entry-Qs frosne Exit-verdi og observerte D1-mål må ikke behandles som samme fasit.
training_enabled forblir false til den konkrete kontrakten og evidensporten er løst.
Senere generalisering, paritet og operativ kvalifisering er ikke undersøkt her.

## Native v38-inputgrunnlag bundet

Rådatapar og squeeze-kalibrering er kontrollert som uendrede avhengigheter.
Ferske v38-inputartefakter og separat readiness er fullført og kontrollert.
Se [gjeldende native scope](NATIVE_V38_BOUNDED_LEARNING_20261001.md).
Gjeldende Entry-/Exit-mål er kartlagt; observerte økonomiske utfall må
rapporteres separat. Konkret normaliserings-/læringsrecipe gjenstår. Byggegodkjenningen er brukt;
ikke relanser ferdig inputbygg. Økonomisk porteføljemåling er en egen port.


## Native design fryst; første kalenderrettelse målt

Det separate v38-designet og CONTROL256 er nå fryst før mål-/modellutfall.
Se configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json og native-rapporten.
TRAIN 2011–mai 2025 og senere fysisk VAL juni 2025–juni 2026 har separate
radkoordinater. Første klargjøringseier er rettet og kontrollert på hele
kalenderen; 15 fokuserte tester besto denne første rettelsen. M1-/TEST-seal-
admission er senere landet som beskrevet nedenfor; øvrige native eiere
trenger fortsatt den dokumenterte migreringen. Det er ingen
native launch eller læring ennå; mål og trening er fortsatt separate statuser.


## En M1-kilde: rådataparitet fullført, TEST-admission rettet

Alle 1 215 514 M5-barer før TEST gjenskapes eksakt fra den opprinnelige
M1-kilden. Eksisterende komplett pre-TEST-M1-fil er kvalifisert og skal
gjenbrukes. 21 fokuserte tester og ekte metadata beviser pilotens rettede
TEST-seal-admission. Se native-rapporten og NATIVE_M1_SOURCE_PARITY_20261001.json.
Kildebytte, øvrig native komponentklargjøring og læringsmålingen er ikke ferdige.


Den komplette M1-kilden er nå bundet til en konkret klargjøringsrecipe gjennom
sin genuine pre-TEST-parent. 144 tester og ekte samlet kilde-/kalender-/seal-
kontroll består. Gammel filtrert M1-visning og komplett M1 har ulike fysiske
radkoordinater. Child-views og videre kalenderadgang er senere rettet som
beskrevet nedenfor; tilstandsindekser og øvrige native eiere gjenstår.
Ingen ny rådatakopi eller kvalifiseringsrunde trengs. Klargjøring, normalisering
og læringsmåling er fortsatt egne ufullførte porter; se native-rapporten.


## Entry/M1-klargjøring fullført; tilstandsstøtte og læring gjenstår

Den vedtatte recipe har nå Entry-adoption, child-admission og separate M1-views
med fryste perioder. Alle 652 552 TRAIN-/70 880 kontrollrader har eksakt første
M1-tilstand; alle M1-markedsfelt er identiske med råkildens korresponderende rader.
21,16 GB Entry-data gjenbrukes direkte. 68 fokuserte tester og ekte
normaliseringsadmission består. Ingen normer eller modell er tilpasset.
Markedslukking/gap og klokkestøtte er senere kontrollert på alle Entries:
34 pause-signaturer er tilpasset bare ny TRAIN og brukt uendret på kontroll.
Alle 256 fryste kontrollpunkter har minst 339 observerte overganger.
13 fokuserte tester består for normaliseringskalender og immutable publisering.
Dette er ikke full feature-/målstøtte. Eksisterende M1-featureflate har fortsatt
gammel filtrert alignment. Neste grense er kvalifisert før-TEST-featuregjenbruk
eller nødvendig rekonstruksjon, deretter indekser, mål, normer og læringsmåling.
Gjenbruk ferdige Entry-/M1-visninger, schedules og geometriarrayer.
Resultater og siste readiness er bundet i begge status-JSON-ene.

Den konkrete M1-reparasjonen er nå forhåndsregistrert separat i
configs/research/NATIVE_V38_M1_REALIGNMENT_20261001.json. Eksisterende eier
gjenbruker beriket M1 og fryste parametre, med nye utdatarader bare på komplett
før-TEST-klokke. Lesesomfang for den delte berikede kilden og krav til
terminal/klokkedekning er eksplisitt bundet i native-rapporten. Ingen normfit
eller læring er åpnet; faktisk runtime-status avgjør om produsenten lever.

M1_FEATURE_REALIGNMENT_001 er fullført og kontrollert: 5 523 147 feature-
rader på komplett før-TEST-klokke, med full dekning av alle nye TRAIN-/kontroll-
M1-visninger. Parametre og ordnede felt er bevart. Neste aktive delmål er
binding av denne flaten til normaliseringspopulasjon, indekser og mål.
Ingen læring eller økonomisk fordel er dokumentert av inputbygget.
Brukerønsket PC-omstart er gjennomført ved trygg grense; WSL, GPU og capped-
kjøring er kontrollert. LAN fungerer; opprinnelig Tailscale-rute er uavklart.

01.10 kl. 21:11 UTC: hele TRAIN-sekvenskontrollen og fysisk
normaliseringspopulasjon er fullført, med terminal exit 0. 652 552 entryer,
955 670 unike M5-kontekstrader og 3 995 148 unike observerbare M1-rader;
uavhengig geometri stemmer eksakt. 19 fokuserte tester består. Ingen
normaliseringsfit, modellforwards eller optimizersteg. Neste avhengighet
er native tilstandsindekser og norm-/målbinding; øvrige mål består.
