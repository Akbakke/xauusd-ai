# Sweep, ankret VWAP og aktivitet — fullført 30.09.2026

**Resultat: NO_GO for den ene forhåndsregistrerte handelsregelen.**
Ingen modell ble trent på denne kombinasjonen; ingen 20-timers treningsøkt ble kjørt.
Lært featureverdi er **ikke målt**. Den tidligere anbefalingen om å avvise større
modelltrening på grunnlag av denne regeltesten trekkes tilbake 01.10.2026.
Det negative regelresultatet og de opprinnelige målingsartefaktene endres ikke.
Full B er fortsatt blokkert på egne kildekrav og er ikke erstattet av denne testen.

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

Selve regeltesten endret ingen native features, modellvekter eller treningskontrakter.
Brukerens etterfølgende bestilling 01.10 gjelder den separate native funksjonen nedenfor.
TEST forblir forseglet.


## Bestilt native funksjon 01.10.2026

Brukeren ba uttrykkelig om å bygge en avansert funksjon i den lærte boten.
Implementasjonen utvider eksisterende SMC-eier og eksisterende lokale featurelag;
den innfører ingen ny beslutningsregel eller ny modellarkitektur.

Hver ensidig, bekreftet sweep oppretter et anker på den lukkede hendelsesbaren.
Opp- og ned-ankre huskes uavhengig. En ny sweep på samme side erstatter det ankeret;
en motsatt eller dobbel sweep sletter ikke den andre sidens informasjon.
Ingen fem-bars bekreftelse, fast handelsgrense, filter eller 60-minutters exit
fra den avsluttede regeltesten følger med inn i modellen.

Hver side gir seks kontinuerlige målinger (12 nye felt totalt):

| Felt etter smc_sweep_{up,down}_avwap_ | Eksakt betydning |
|---|---|
| age_bars | Antall observerte lukkede native barer siden ankeret |
| dist_atr | (close minus aktivitetsvektet close siden ankeret) / gjeldende ATR |
| dispersion_atr | Aktivitetsvektet standardavvik siden ankeret / gjeldende ATR |
| anchor_close_dist_atr | (close minus close på ankerbaren) / gjeldende ATR |
| level_dist_atr | (close minus det bekreftede nivået som ble sveipet) / gjeldende ATR |
| mean_activity_ratio | Gjennomsnittlig prisoppdateringsantall siden ankeret / ankerbarens antall minus 1 |

Ankerbaren er inkludert. Volumvekten er OANDA prisoppdateringsantall, ikke
utført volum eller aggressor-/ordrebokflyt. Ingen tak, sentinel eller nøytral
utfylling erstatter manglende hendelseshistorikk. En sides målinger er NaN før
dens første hendelse; prisnormalisering venter også på observert kausal ATR.
Lagring med to ankertilstander og vektet Welford holder minnebruken konstant
under replay og bevarer eksakt resultat over vilkårlige kronologiske chunks.
Markedsstengning legger ikke inn syntetiske barer; alder teller observerte barer.

Funksjonen er obligatorisk i eksisterende native lokale SMC-lag for M5 Entry
og M1 Exit og rutes til smc_liquidity_encoder. De øvrige åtte-familie-rutene,
høyere tidsrammene og den eksisterende lærte attention/fusion beholdes.
Signalidentiteten er v38 og full-stack-identiteten v26. Gamle v37-datasett,
normaliseringer og vekter kan ikke behandles som ferdige v38-artefakter.

Kontrollen bruker syntetisk referanseregning, kausalitet, hendelsesidentitet,
uendret eksisterende SMC-evidens, chunk-paritet, faktisk native reader på M1/M5,
signalmanifest og gradientvei til modellens Entry-verdier uten optimizersteg.
Deretter kjøres én manifestbundet inputkontroll på eksisterende 2009–2011-priser:
2009–2010 er historisk prefiks, og 2011 måles for inputdekning. Ingen utfall
eller økonomi evalueres. Manifest: configs/research/TA_SWEEP_NATIVE_INPUT_AUDIT_20261001.json.

Dette implementerer en representasjon modellen kan lære fra. Før læring kan
vurderes må de endrede feature-, normaliserings- og datasettartefaktene bygges
og bindes, og en avgrenset treningssammenligning spesifiseres med samme
kronologiske populasjon, mål, kostnader, initialisering og beregningsbudsjett.
Sammenlign dagens features med de nye ankermålingene; rapporter faktisk TRAIN-
læring og senere generalisering separat. Den kontrollen er ikke kjørt.
En ny ONLINE-funksjon krever også ny initialbaseline. Urørt endelig TEST brukes
først etter fryst modellvalg. Ingen 20-timers kjøring eller lønnsomhet er etablert.


### Fullført kontroll av den native funksjonen

Inputkontrollen besto på 352 256 M1-rader og 72 040 M5-rader fra hele 2011.
900 480 M1-rader og 186 736 M5-rader fra juni 2009 til desember 2011 inngikk
i beregningen, slik at tidligere hendelser var tilgjengelige som prefiks.
Alle 12 nye felt er endelige og varierer i de vurderte radene; ingen av dem
er en eksakt kopi av et annet nytt felt. Ingen påstand om full rang eller
uavhengighet fra alle eksisterende features følger av dette.
Native reader-paritet og eksakt chunk-carry besto på begge klokker; 128 direkte
vektede referanseberegninger stemmer. Nåværende signalbredde er 254.
82 fokuserte syntetiske tester samt de obligatoriske pre-commit-portene besto.
Første audit-start feilet ved import før datalesing; feilloggen er bevart.
Samme immutabelt bundne skript ble deretter kjørt med korrekt importsti.
Rapport: docs/TA_SWEEP_NATIVE_RESULT_20261001.json.

Dette er implementasjons- og inputbevis. Ingen ny treningsnormalisering,
komplett v38-datasett, modelltrening, ny bundle-paritet eller økonomisk test
av en lært v38-modell er ferdig. Regeltest og inputtest erstatter ikke disse.
