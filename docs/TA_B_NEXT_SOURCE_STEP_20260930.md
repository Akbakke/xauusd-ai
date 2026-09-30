# Neste datasteg for B — 30.09.2026

De fire ALFRED-arkivene er hentet og kontrollert via det offentlige nettleserskjemaet.
API-nøkkel er ikke nødvendig for denne dokumenterte ruten. Full B er fortsatt **ikke
målt**: historisk versjons-/publikasjonsdekning for GLD og COT er ikke kvalifisert.
Ingen redusert firekilders variant er åpnet.

## Målt på ekte kildebytes

27 ZIP-filer, 592 577 bytes, dekker alle 11 249 annonserte versjonsdatoer innen
2009–2025 for de fire navngitte seriene. Dette er summen per serie, ikke antall
uavhengige datapunkter eller handelsmuligheter.

| Serie | Valgte versjonsdatoer | Første arkiverte versjon | Observasjonsdatoer | Unike observasjon/versjon-rader | Observasjoner med flere versjoner |
|---|---:|---|---:|---:|---:|
| DFII10 | 4,182 | 2009-01-02 | 4,434 | 4,435 | 1 |
| DTWEXBGS | 361 | 2019-02-04 | 4,432 | 17,555 | 3,845 |
| T10YIE | 2,948 | 2014-01-27 | 4,435 | 4,437 | 2 |
| VIXCLS | 3,758 | 2010-11-22 | 4,435 | 4,446 | 7 |

Observasjoner er avgrenset til 2009-01-01–2025-12-31. Siste faktisk tilgjengelige
observasjon er 2025-12-30 for DFII10, 2025-12-26 for DTWEXBGS og 2025-12-31 for
T10YIE/VIXCLS. Manglende verdier er bevart og telt, ikke fylt inn.
Tidlige valgte DFII10-vintager inneholder ikke nødvendigvis en observasjon innen
vårt datointervall; første realtime_start i de hentede radene er 2009-01-06.

Samtlige ZIP-hasher, CRC-er, medlemsfiler, serienavn, historiske enhets- og
frekvensetiketter, CSV-skjemaer, datogrenser og numeriske verdier er kontrollert.
README-versjonsdatoene samsvarer med hashene av de faktiske utvalgene i nettleseren.
Sammenslåingen har null motstridende duplikater, null overlappende intervaller og
null hull *mellom påfølgende versjoner av samme observasjon*. Dette beviser ikke
full kalenderdekning eller at en komplett B-featurematrise finnes.
Identiske rader som gjentas mellom filblokker er deduplisert bare i kildeauditen.

Dollarindeksen har 3 845 observasjonsdatoer med flere versjoner. En nåtidskopi av
historikken er derfor ikke en erstatning for disse arkiverte versjonene.

## Konsekvens for en senere B-måling

Første dollarindeksvintage er 04.02.2019. Eldre observasjoner i denne filen kan
ikke brukes til beslutninger før denne datoen. Tidligste felles firekildeperiode
ligger derfor tidligst etter denne datoen, deretter vedtatt publikasjonslag,
indikator-warmup og faktisk XAU D1-dekning. GLD/COT kan begrense den ytterligere.

Verdier blir tilgjengelige ved slutten av realtime_start-datoen i
America/New_York pluss én komplett kanonisk XAU D1-periode. Fremtidig realtime_end
er kun revisjonsmetadata, aldri beslutningsinput. Første historiske observasjon
er ikke samme ting som første historiske tilgjengelighet.

Før fit kreves fortsatt alle seks navngitte kilder, eksplisitt kausal
featurebygging, felles A/B TRAIN-/eval-populasjon og commit av forhåndsregistrering.
Den gamle samlede A-statistikken fra 2011–2025 kan ikke brukes som direkte
referanse mot en B-populasjon som først starter i 2019.

## GLD/COT og avviste prøver

Åtte GLD-arkivoppslag og to oppslag for den daterte COT-rapporten ga ni HTTP 429
og én HTTP 503. Ingen dekning kan utledes fra disse feilresponsene. Ingen videre
arkivoppslag ble gjort i denne bølgen. Henteeieren er rettet til å hoppe over
resten av metadata-batchen etter første HTTP 429 og bevare Retry-After.
Den fokuserte testen bekrefter nøyaktig ett nettverksforsøk ved en simulert 429.

CFTC-indeksen identifiserer den daterte Legacy Futures Only COMEX-rapporten
26.03.2019. Dette er et konkret arkivsøkepunkt; indeksen beviser ikke at vi har
opprinnelige og reviderte rapportbytes. Tidligere enkeltkopier av GLD og COT
bevares; de er fortsatt utilstrekkelig dekning.

ALFRED avviste først 4 182 DFII10-vintager i ett kall med dokumentert grense på
450. Den avviste prøven bevares; alle etterfølgende 27 blokker lyktes uten
ny nedlasting av fullførte blokker. AUDIT_001/002 bevares som avviste kontroller:
kontrolloppsettet antok først bare VIX-enheten Index, deretter frekvensetiketten
Daily. Kildens faktiske historikk er Percent til 16.04.2014, deretter Index,
og frekvens Daily, Close. Disse metadataene er nå eksplisitt bundet.
Ingen verdier er skalert om for å få kontrollen til å bestå.

## Bevist konsistent og fortsatt ubevist

To fokuserte mekaniske tester besto under capped audit. Den endelige filkontrollen
kjørte gjennom samme ressursvakt med 4 GiB minnetak, 512 MiB swap og begrensede
CPU-/trådressurser. Hashbindinger og antall unike rader ble deretter kontrollert
uavhengig fra de bevarte ZIP-filene. Ingen modell-/treningslogikk er endret.

Det er **ikke undersøkt** om B forbedrer prognoser eller netto handelsøkonomi.
Ingen XAU-utfall, TEST, fit, native trening, handel eller betalt datatilgang er brukt.
Neste datagrense er dokumentert GLD-tonnasje og COT 088691 Legacy Futures Only
med tilstrekkelig historisk publikasjon-/versjonsdekning. Kildebevis må komme før
en ny sammenlignende måling; ytterligere henting krever et navngitt manifest.

## Bevis og primærkilder

- [Maskinprodusert sammendrag](TA_B_NEXT_SOURCE_STEP_20260930.json).
- Endelig kildeaudit:
  /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/B_ALFRED_BROWSER_CHUNKS_20260930_001/AUDIT_003/RESULT.json.
- Resultat-SHA256: 9092bdb7a95742519e939a02321f86d00e62c4cd6b48d3f3918697bee8e80f29.
- Hentemanifest: configs/research/TA_B_ALFRED_BROWSER_CHUNKS_20260930.json,
  committet i 950a01cc før de 27 nedlastingene.
- Endelig auditmanifest: configs/research/TA_B_ALFRED_ARCHIVE_AUDIT_METADATA_20260930.json,
  kildecommit 6edd3de7.
- [ALFREDs eksportdokumentasjon](https://alfred.stlouisfed.org/help/downloaddata).
- [CFTCs daterte rapportindeks](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cot032619).
- [CFTCs særmeldinger om blant annet rapportrevisjoner](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).
