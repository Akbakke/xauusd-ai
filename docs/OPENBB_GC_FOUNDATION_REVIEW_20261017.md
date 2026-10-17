# OpenBB og GC/P0 — kontrollert forskningsgrunnlag, 17.10.2026

GC-grunnlaget er bygget og kvalifisert som **maskert forskningsinput**. P0 er
fortsatt stengt: kontrollens COT-tilgjengelighet har en dokumentert feil, og den
endelige regel-/XGBoost-recipen er ikke fryst. Ingen modelltrening, strategi-
simulering, native TEST, produksjon eller livehandel er endret/kjørt i bølgen.
Ingen nye Databento-kall eller betalte datakall.

## OpenBB: faktisk merverdi og eksisterende data

Separat miljø: `OPENBB_FREE_SOURCE_REVIEW_20261017_014/ENV`, core 2.0.1,
FRED/CFTC 2.0.0 (Python V5, Apache-2.0). Native miljø og innhenting er beholdt.
Transitive pakker er logget i ENV_FREEZE.txt. To små gratis CFTC-kall lyktes.
FRED_API_KEY er ikke satt i dette miljøet: **ingen FRED-kall**, ingen påstått
FRED-verdisammenligning eller ALFRED-kvalifisering.

- Eksisterende 61 makrofelt: 4 legacy COT, 44 panel og 13 drivere. Realrente,
  nominelle renter/kurve, breakeven og VIX finnes. GLD-feltene er **beholdning
  i tonn**, ikke ETF-pris. Bred USD finnes allerede i separat H.10-forskning.
- OpenBB legacy: 113 overlappende rapporter, fem OI-/posisjonsfelt, **null avvik**
  mot eksisterende arkiver. OI er også lik disaggregated i alle 113 rapporter.
- Managed Money er en annen kategori: netto/OI skiller seg fra legacy
  noncommercial i alle 113 rapporter. Long-/short-endringer stemmer i alle
  112 sammenlignbare rapportpar per side. Past-52-report z bruker kun tidligere
  rapporter og har 61 observasjoner etter oppvarming. Alt er **review-only**;
  ingen nye makrofelt er registrert eller lagt til modellen.
- FRED bør primært kontrollere eksisterende kilder. Stress (NFCI/STLFSI4) og
  rentekurve-/forventningshypoteser må først avstemmes mot eksisterende rådata.
  DGS2 er en rente med term-/risikopremier, ikke en ren forventet styringsrente;
  T5YIFR gjelder inflasjonskompensasjon. Ingen ny signalverdi er bevist.

**Konkret PIT-feil:** CFTC stoppet publisering høsten 2025. Rapporten datert
30.09 var annonsert publisert 19.11, mens native kode bruker rapportdato +7 døgn.
18 462 M5-observasjoner flagges mot den annonserte etterslepsplanen.
Dette er ikke en full faktisk utgivelses-/vintagekalender. Den foreløpige
utestengingen nedenfor fjerner hele november/desember 2025; den erstatter
ikke en korrekt join til siste faktisk publiserte rapport. Baseline er urørt.

OpenBB-datoer er ikke publiseringskvitteringer. FREDs standard er dagens
vintage; ALFREDs realtidsdato trenger dessuten dokumentert intradagtilgjengelighet.
De eksisterende ALFRED-auditkvitteringene viser mislykkede hentinger, ikke
bevis for at et komplett vintagegrunnlag finnes. NFCI og bred USD kan revideres.

## Verifisert GC-dekning

26 måneder, 66 MBP-1-filer / 892 100 712 records. Original 004-kontraktvalg
og alle 11 Trade-metrikker er eksakt bevart på 765 839 native-eligible M1.
Rullering følger opprinnelig tilgjengelighet ved minuttstart; ingen prisjustert
kontraktblanding eller fallback til dagens mest praktiske kontrakt.

| Familie | Gyldige native M5 | Måneder med støtte |
|---|---:|---:|
| OFI, depletion/refill, quote-aktivitet | 83 776 | 26/26 |
| Tidsvektet BBO, spread, microprice-offset, prisrespons | 83 776 | 26/26 |
| Streng rolling 15 min VWAP/POC/VAH/VAL | 48 164 | 26/26 |
| Streng rolling 60 min VWAP/POC/VAH/VAL | 7 330 | 26/26 |
| Kvalitetsforankrede segmentprofiler | 89 203 | 26/26 |
| Komplett GC-sesjonshistorikk **fram til beslutning** | 2 263 | 26/26 |

OFI/BBO og usignert profil krever ikke kjent aggressorside. Eksisterende signert
footprint/delta/CVD beholder den opprinnelige masken på 16 745 M5.

GC-historikken bruker nå hele kildeklokken, også 18:00–18:03 New York som
manglet i spot-utvalget. Alle tidligere godkjente rullerende verdier er identiske.
Sesjonsprefiksene er median 25, maks 185 minutter: **ingen hel ren avsluttet
GC-sesjon er bevist**. I den disjunkte kvalitetsvannfallen forklarer gjentatte
T-signaturer 133 533 forkastede sesjonsprefikser etter tidligere prioriterte
historikk-/rulle-/flaggårsaker. Ingen deduplisering eller svakere regler er innført.
Minutter uten trusted Trade er ikke automatisk klassifisert som manglende data.

Value Area er en forhåndsdefinert, sammenhengende 70 %-utvidelse rundt POC med
lavpris ved like volumer; den er ikke q25/q75. Prisnivåer og avstander gjelder
GC, ikke direkte XAUUSD-ordrepriser. Ufullstendige sesjonsprofiler er diagnostikk
og er ekskludert fra primær P0. ATR/RV/regimer gjenbrukes fra original eier.
MBP-1 gir L1-proksier, ikke skjult likviditet eller individuell ordretracking.

Faktiske L1-prefikser: 12 kontrakter / 60 M1. Profilprefiks, råvolumavstemming og
NaN-/null-bevarende Parquet-readback består. Uavhengig M1-prefiks gir identisk
sesjonsmaske for alle 141 806 beslutninger. Alle kvalifiserte BBO-verdier består
fysiske grenser; heltallsakkumulering for OFI har målt sikker float64-grense.
Baselinefiler, Entry/Exit-eiere og historiske økonomifunksjoner er uendret;
se SOURCE_AND_PUBLICATION_REVIEW.json. 115 unike fokuserte tester er kontrollert;
ett policyprojeksjonsavvik ble rettet og gjenkjørt målrettet. Alle testkvitteringer
bevares. 015/016/018 stoppet i kontroller og er
konsumert; feilene er rettet og logger/originaler beholdt. Ingen gammel plan relansert.

## P0: åtte forberedte armer, ingen modelltest

Kontroll, F, L, V, FL, FV, LV, FLV bruker 12 **nye** F-felt og åtte relative
rolling-15-nivå-/responsfelt, med original V16. 37 181 observasjoner har felles
endelig F/L-maske. Dette er kandidatbeslutninger, ikke handler.

| Beregningshorisont | F/L + eksakt M1-bane | Etter foreløpig COT-utestenging |
|---|---:|---:|
| 30 min | 37 181 | 34 948 |
| 120 min | 36 264 | 34 080 |
| 360 min | 31 544 | 29 560 |

Ny forskningsvariant krever entry etter 60 s, null ekstra M1-stempelforsinkelse
og ingen manglende M1 inne i banen. Dette er label-/evalueringskvalifisering,
ikke maksimal holdetid. M1-stempel er ikke dokumentert tick-/filltid.
Av 373 136 manglende minutter forklares
332 220 av eksisterende ukestengningskontrakt;
40 916 dag-/helligdags-/datagapminutter er fortsatt uklassifisert.
Native D1/h20 holdes separat.

Alle armer er bundet til korrekt SHORT-inngangsnotional og felles v3-økonomi:
spread i bid/ask-prisene, sentralt 4 bps execution og årlig LONG 569 / SHORT 0
bps som **nåvilkårsprojeksjon**. 11 forhåndsregistrerte avgifts-/finansierings-
scenarioer omfatter også ugunstig finansiering på begge sider. Det er ikke
rekonstruert historisk brokerfinansiering. Execution bruker XAUUSD bid/ask;
GC-spread er en input. Ingen strateginetto, Sharpe,
drawdown, eksponering eller prediktiv edge er målt i denne bølgen.

Neste P0-utviklingstest krever en eksplisitt korrekt forskningskontroll for
makrotilgjengelighet, eller et forhåndsregistrert kontrollsett uten ukvalifiserte
makroinputs; deretter eksakt regel-/liten XGBoost-recipe, felles risiko/utførelse,
walk-forward/purge/embargo og ny avgrenset kjøreautoritet. Ingen stille endring
av baseline. Alle 26 måneder er utviklingsdata. Manglende urørt periode sperrer
uavhengig edgebekreftelse/promotion; en senere innrømmet P0 kan bare være utvikling.

Kilder: [OpenBB V5](https://docs.openbb.co/odp/python/installation),
[CFTC API](https://docs.openbb.co/odp/python/reference/cftc/cot),
[CFTC etterslepsplan](https://www.cftc.gov/PressRoom/PressReleases/9138-25),
[ALFRED/FRED-vintager](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html),
[NFCI-revisjoner](https://www.chicagofed.org/research/data/nfci/current-data),
[H.10](https://www.federalreserve.gov/releases/h10/About.HTM).
