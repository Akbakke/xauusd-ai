# Sweep, ankret VWAP og aktivitet — fullført 30.09.2026

**Resultat: NO_GO for den ene forhåndsregistrerte regelen.** Den anbefales ikke
innført i native modellen eller fulgt av større trening. Full B er fortsatt blokkert
på egne kildekrav og er ikke erstattet av denne testen.

## Målt på OANDA

2011–2025 er gjennomgått; 2021–2025 er den forhåndsbestemte senere vurderingsperioden.
Hele historikken er gjenbrukt utvikling, ikke urørt holdout. Ingen parametere er tilpasset.
Den senere perioden har 10 899 felles muligheter. Tabellen bruker observerte bid/ask,
1 bp slippage ved hver utførelse og eksisterende finansieringsproxy EFFR + 1,29 pp.

| Variant | Utførte handler | Netto bps per felles mulighet |
|---|---:|---:|
| Sweep alene | 10850 | -3.709 |
| Rullerende VWAP20 + aktivitet | 1635 | -0.650 |
| Sweep + ankret VWAP + aktivitet | 2588 | -0.919 |
| LONG på samme muligheter | 10850 | -3.516 |

Filtrerte og utløpte muligheter beholdes med nullresultat; tabellen er ikke snitt per
utført handel. Totalt ble 32 774 muligheter valgt gjennom 2011–2025, med 93 uten
utførelse og én avsluttende sensurert posisjon som ble gjort opp med kostnader.
Alle armer reserverer de samme tidspunktene; et filter får ikke velge nye handler.

Ankret-kombinasjonen taper 0,919 bps per felles mulighet. Simultant intervall mot
FLAT er [-1,235; -0,603] bps. Mot rullerende VWAP med samme aktivitet er forskjellen
-0,269 bps, intervall [-0,446; -0,091]. Mot sweep alene er tapet redusert med 2,790 bps.
Dette tilfredsstiller ikke det samlede kravet om bedre beslutninger og positiv økonomi.
Uten slippage er kombinasjonen fortsatt negativ: -0,444 bps etter spread/finansiering.
Målt mid-price-bidrag før kostnader er -0,031 bps per mulighet.

96 endepunkter var deklarert. 64 definerte endepunkter ble korrigert samlet med
paret stasjonær bootstrap/max-t (1999 trekk, forventet blokk 20 kalenderdager).
32 Sharpe-endepunkter er udefinerte grunnet insolvens i de sammenhengende
fast-startkapital-regnskapene. Tap etter slik insolvens er algebraisk
mulighetsdiagnostikk, ikke en påstand om en fortsatt gjennomførbar portefølje.
Ingen Sharpe ble konstruert for å få analysen gjennom.

## Regelen som faktisk ble testet

Eksisterende kausale SMC-eier gir ensidige, bekreftede M5-sweephendelser.
Up-sweep fades SHORT; down-sweep fades LONG. Etter fem nye lukkede sammenhengende
M5-barer kreves close på fadesidens side av VWAP ankret ved sweepbaren, og
vol_ratio_5_20 > 0. Enhver ny sweep før bekreftelsen ugyldiggjør den gamle kandidaten.
VWAP bruker close × prisoppdateringsantall fra sweepbaren gjennom bekreftelsesbaren
(seks barer). Det er en aktivitetsvektet prisproxy, ikke transaksjons-VWAP.

Sweep alene og rullerende VWAP20 med samme aktivitetskrav vurderes på samme
bekreftelsestidspunkt. Utførelse bruker første observerte quote ved/etter
beslutningen. Målehorisonten er 12 M5-barers veggklokketid, med faktisk neste
quote og eksplisitt sluttoppgjør. Dette er ikke en maksimal native holdetid.
Ingen brede regel-/terskel-/horisontsøk er gjennomført.

## Dukascopy

268 eksisterende filer, 16 024 354 bytes, ble kontrollert. 264 filer inneholder
3 903 452 strukturelt gyldige ticks; fire filer er tomme. Ingen observerte
kryssede quotes, bakovergående relative tidsstempler eller negative kvoterte størrelser
i de strukturelt gyldige filene. Én fil ligger i 2025-mappe, 263 i 2026-mapper.
Originale hentekvitteringer og verifisert absolutt datokobling mangler.
Sammenhengende dekning er derfor ikke dokumentert, og cachen er ikke brukt i
økonomitesten. Ingen nye markedsdata ble lastet ned. Order flow fra utførte handler
eller full historisk ordrebok er ikke etablert.

## Kontroller og bevaring

Åtte fokuserte syntetiske tester besto: klokke/ankring, eksakt vektet pris,
prefiksinvarians/fremtidsmutasjon, ugyldiggjøring/gap, BI5-layout, felles reservasjon,
eksisterende C-regnskap og ende-til-ende insolvensrapportering.

Første kjøring stoppet på avkastning fra negativ egenkapital. Original manifest,
terminal, signaler, utvalg og bok er bevart. Minste rettelse gjenbruker etablert
insolvenshåndtering. Fire forberedelsesfunksjoner ble kontrollert kildeidentiske;
regel, kostnader, perioder og parametere ble ikke endret. Ferdige signaler ble
gjenbrukt, og det nye utvalget måtte være eksakt likt det lagrede.

Uavhengig kontroll besto: alle utfall og signert finansiering i 32 bøker,
kontantregnskapene, 32 gjennomsnittsforskjeller, felles klokke/utvalg og
originale inn-/utgangsquotes for 30 handler fordelt over alle 15 årene.
Dette er teknisk og økonomisk historikkevidens, ikke urørt OOS eller native læring.

## Autoriteter

- Opprinnelig forhåndsregistrering: configs/research/TA_SWEEP_PREREG_20260930.json
- Uendret hypotese med regnskapsrettelse: configs/research/TA_SWEEP_ACCOUNTING_REPAIR_20260930.json
- Cache-audit: configs/research/TA_SWEEP_DUKASCOPY_AUDIT_20260930.json
- Aggregert rapport: docs/TA_SWEEP_RESULT_20260930.json
- Måling: /home/andre2/GX1_RUNS/TA_RESEARCH_20260930_SWEEP/MEASUREMENT_002/RESULT.json
- Uavhengig kontroll: /home/andre2/GX1_RUNS/TA_RESEARCH_20260930_SWEEP/MEASUREMENT_002/VERIFICATION.json

Native features, modellvekter og treningskontrakter er ikke endret. TEST forblir
forseglet. Neste steg er ingen automatisk utvidelse av denne regelen; en ny
hypotese trenger en egen begrunnelse og forhåndsregistrering.
