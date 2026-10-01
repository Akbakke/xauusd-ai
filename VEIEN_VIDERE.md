# Veien videre — oppdatert 01.10.2026

## Native v38 — nye inputs før læringsmåling

Gjeldende neste steg er det særskilt bundne v38-inputbygget:
[scope og læringsmål](docs/NATIVE_V38_BOUNDED_LEARNING_20261001.md).
Rådatapar og seks-klokkers squeeze-parametre er kontrollert og gjenbrukes.
V38-featureflater og datasett bygges i ny namespace med uendrede splitvinduer.
Les levende prosess og terminal receipt før videre arbeid; kilde fryses under
kjøringen. Trening og normaliseringsfit er ikke åpnet. Eldre neste-steg-tekst
nedenfor er historikk inntil ny komplett evidens foreligger.

## Aktivt mål 01.10 — makrotest fullført, native læringsbevis gjenstår

Den separate MACRO_CORE-armen er ferdig målt og uavhengig kontrollert.
HGB med makro ga +66,77 % netto i gjenbrukt 2020–01.12.2025, mot +56,56 %
for matchet pris-HGB og +66,42 % for samme-risiko LONG. Ridge ble svakere.
Begge fikk INKONKLUSIV; ingen dokumentert makrofordel eller native promotion.
49 tester består. [Resultat og begrensninger](docs/TA_MACRO_CORE_RESULT_20261001.md).

Gjeldende [aktive målplan](docs/AUTOMATIC_BOT_EVIDENCE_PLAN_20261001.md):
land native v38-mål, matchede læringsrader og cache-/byggbinding før ny måling.
Deretter gjenstår senere generalisering, ny paritet og offline driftskvalifisering.
Makrotesten skal ikke relanseres. Full B er fortsatt et eget ufullført mål.
Ingen native trening, TEST eller handel er åpnet.
Eldre neste-steg-tekst nedenfor er historiske delmål.

## Native sweep-/AVWAP-funksjon 01.10 — implementert og inputkontrollert

Brukerens bestilling er nå implementert i eksisterende feature- og kontrakteiere:
to uavhengige sweep-ankere, 12 kontinuerlige målinger og obligatorisk native
M5/M1-ruting til modellens likviditetsfamilie. Dette er ingen fast handelsregel.
82 fokuserte tester har bestått, inkludert modellens gradientvei uten optimizer.
Signal v38 krever nye bygg-/normaliseringsartefakter; ingen modell er trent med
denne utvidelsen. Den manifestbundne inputkontrollen besto på hele 2011:
352 256 M1-rader og 72 040 M5-rader, med prefiks fra 2009. Alle nye felt
er endelige og varierende. Native innlesing og replay i deler er eksakt like.
Signalbredden er 254. Rapport: docs/TA_SWEEP_NATIVE_RESULT_20261001.json.
Detaljer: docs/TA_SWEEP_AVWAP_20260930.md. Ingen ny lærings- eller lønnsomhetspåstand.
Neste steg er bundne v38-bygg-/normaliseringsartefakter og en avgrenset,
sammenlignbar treningsmåling. Ny initialbaseline og senere validering gjenstår.

## Fast sweep-regel: NO_GO; lært featureverdi er ikke målt

Den godkjente datakontrollen og faste tekniske hypotesen er ferdig målt og kontrollert.
Se docs/TA_SWEEP_AVWAP_20260930.md og docs/TA_SWEEP_RESULT_20260930.json.
På 10 899 felles muligheter i gjenbrukt 2021–2025 ga ankret-kombinasjonen
-0,919 bps etter spread, 1 bp per utførelse og finansieringsproxy. Den var
0,269 bps svakere enn rullerende VWAP med samme aktivitet. Ingen positiv økonomi.
32 regnskap og primære gjennomsnittsforskjeller er uavhengig kontrollert.
Dukascopy-cachen har 3 903 452 strukturelt gyldige ticks, men mangler verifisert
datokobling og sammenhengende dekning; den er ikke brukt i økonomitesten.
Presisering 01.10: ingen modell ble trent på kombinasjonen. NO_GO gjelder bare den
faste regelen, ikke indikatorenes mulige verdi som lærte inputs. Anbefalingen om
å avvise videre modelltrening på dette grunnlaget trekkes tilbake.
Brukeren har nå bestilt en avansert kausal funksjon i den faktiske modellkjeden.
Den nye implementeringen og statusen står øverst; historiske målinger er bevart.
Full B er fortsatt blokkert; TEST, native trening, live/paper og spending er stengt.
Denne seksjonen erstatter anbefalingen og kjøringsinstruksene lenger ned i historikken.

Gjeldende status eies av [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md). Bruk bare
`/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Én agent og én tung jobb innen CURRENT.

Avgrenset oppfølging 30.09 er fullført: den bestilte indikatorrevisjonen og
én uttrykkelig godkjent OANDA-demo-COT-prøve. Prøven ga HTTP 403/Cloudflare;
ingen COT-data, ingen konklusjon om token eller tjenestens tilgjengelighet.
Indikatorrevisjonen fant aktiv sweep/rullerende tick-VWAP, men ingen aktiv
hendelsesankret VWAP eller ekte order flow. Se kilde-/testbevis og receipts i
CURRENT_HANDOVER.md. Dette endrer ikke B-kildekravene eller treningsportene.

## Videreføring: import av de to bevarte GLD/COT-kopiene

Målet er gjenopptatt med uendret fullstendig omfang. Eksisterende campaign-eier
har nå kildespesifikke parsere for de faktisk observerte GLD CSV- og CFTC
Legacy Futures Only-formatene. GLD tonn skilles fra pris/volum/ounces; de to
observerte helligdagsmarkørene bevares som manglende verdi. COT velger bare
Gold 088691 og kontrollerer regnskapsidentiteter mot open interest.
Kopier bindes til råfil/receipt-hash og eksakt arkivtid, aldri observasjonsdato
som konstruert publiseringstid. Det senere komplette D1-laget er fortsatt påkrevd.
Seksten fokuserte mekaniske tester besto gjennom capped audit; fire importtester
besto på nytt etter atomisk Parquet-skriving.
Manifest: configs/research/TA_B_ARCHIVED_SNAPSHOT_IMPORT_20260930.json.
Import og uavhengig kontroll av de to originalfilene er nå fullført:
3814 GLD-rader (3680 numeriske, 134 HOLIDAY/NYSE Closed) og én COT-rapport
for 02.04.2019. Alle kildeverdier og tilgjengelighetsgrenser stemmer.
Resultat: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/B_ARCHIVED_SNAPSHOT_IMPORT_20260930_001/RESULT.json
(SHA256 b89071177855a3143eb0a97ec67127784a34acb5d271f5a762b741f1e5d1aa89).
Uavhengig kontroll: samme katalog, VERIFICATION.json
(SHA256 21311fb29ffc019d1d7da5c53cf53698111fc09002cadaef90b8ca996931745b).
Ingen nye nettdata, markedsutfall, fits eller TEST er brukt. Den komplette
sekskilde-porten er fortsatt stengt på GLD/COT-historikk og VIX-klokke.
Dette er kildeimport, ikke full GLD/COT-dekning, komplett B-panel eller B-fit.

## Siste kildeavklaring 30.09: VIX-verdier bekreftet, publiseringsklokke uavklart

En forhåndsbundet forespørsel hentet Cboes offisielle VIX-historikk (HTTP 200,
473176 bytes). Alle 26 omstridte ALFRED-verdier stemmer med Cboe CLOSE på
observasjonsdatoen. Ingen stemmer med Cboe CLOSE på eller senest før den
tidligere realtime_start-datoen. Kontrollens siste vurderte dato er 01.09.2025;
ingen XAU-utfall eller TEST-utfall er lest.
Dagens Cboe-fil beviser ikke opprinnelig publiseringstid. En senere ALFRED-
matrisekolonne bygger på samme versjonsintervaller og er ikke et uavhengig
arkivbevis som kan rette den omstridte klokken. Ingen dato er flyttet, ingen
rad er fjernet og ingen Cboe-serie er innført som erstatning for VIXCLS.

Brukeren svarte først «Kun OANDA», og oppga deretter mulig Dukascopy-tilgang.
En lesekontroll fant eksisterende XAUUSD-filer under GX1_DATA/data/external/
dukascopy og dukascopy_cache (sistnevnte har mapper for 2025 og 2026).
Bare filoversikten er kontrollert; ingen tickverdier, dekning eller konto er kvalifisert.
Dukascopys offisielle ITick-kontrakt beskriver beste bid/ask og tilgjengelig
kvotert volum, ikke utførte kjøp/salg; historiske ticks har ett prisnivå per side.
Den offentlige COT-siden oppgir seks valutaer og dokumenterer ikke Gold 088691
med historiske versjoner. Dette løser ikke de aktuelle GLD/COT/VIX-bevismanglene.
Anbefalingen er en avgrenset kvalitets-/nyttevurdering av eksisterende tickdata
før eventuell ny henting, og en separat forhåndsregistrert teknisk hypotese om
sweep, kausalt hendelsesankret VWAP og aktivitet. Dette er en anbefaling,
ikke et vedtak om ny indikator, redusert B eller åpning av trening.
Primærkilder: https://www.dukascopy.com/client/javadoc3/com/dukascopy/api/ITick.html
og https://www.dukascopy.com/swiss/english/marketwatch/cot/.
Den godkjente OANDA-prøven er avsluttet med HTTP 403/Cloudflare, uten COT-data.
Ingen leverandørhenvendelse er sendt, ingen ny konto eller spending er opprettet.
En usendt, presis feilrapport til FRED er klargjort i samme diagnosekatalog.

Samme kildeavhengighet består gjennom den gjenopptatte OANDA-runden,
GLD/COT-importen og denne Cboe-kontrollen. Uavhengig nødvendig arbeid er
ferdigstilt der kildebeviset tillater det. Videre full B krever dokumenterbar
GLD/COT-versjonshistorikk og kildeavklaring av VIX-tilgjengeligheten.
Målet markeres blokkert, ikke fullført; hele sekskilde-/modellomfanget består.
Ingen B-fit, native bygging/trening eller endelig TEST er klar eller startet.

Diagnose: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/B_VIX_CBOE_SOURCE_COMPARISON_20260930_001/REVIEW.json
(SHA256 06802fe9a3a93eec39825cd47520aa6ca222cc55393e3e3351713c3298070e8b).
Sammenligning: samme katalog, COMPARISON.json
(SHA256 32beca96942d0034a5386043543003c515b65806b134d64c4a20fab701b55717).
Manifest: configs/research/TA_B_VIX_CBOE_COMPARISON_20260930.json,
committet før henting i d1256f1c.
Primærkilder: https://www.cboe.com/tradable_products/vix/vix_historical_data
og https://alfred.stlouisfed.org/help/downloaddata.

## Ufullført mål: full B, nå blokkert på kildebevis

Brukeren har bestilt [hele B-løpet og videre kvalifisert modelløp](docs/TA_B_FULL_GOAL_20260930.md).
Målet er ikke oppnådd og er blokkert på manglende kildebevis. Alle seks kilder og faktisk tilgjengelighet
kreves før B-fit. Neste konkrete arbeid er GLD/COT-dekning og komplett kausal
B-implementasjon. Fire ferdige ALFRED-arkiver gjenbrukes.
Makroklokken er implementert; tre fokuserte mekaniske tester besto.
Ekte komponentbygging ble stoppet på 26 VIX-rader der en numerisk verdi er
datert før observasjonsdagen. Ingen tilsvarer siste kjente verdi; samme feilklasse
ga null rader i de tre andre makroseriene. Ingen komponentpanel ble publisert.
Den nye kildegrensen er GLD/COT-dekning og dokumentert VIX-klokke.
VIX-formatkontrollen er fullført: ti av ti celler stemmer, og begge offisielle
formater viser 04.07.2025-observasjonen allerede i 03.07-vintagen.
Dette konkrete avviket er altså i kildeeksporten; klokkerettelse er fortsatt uavklart.
Fullført v37-inputbygging (242 felt) og A-forskningens direkte prisfasit bevares.
Det nye native læringsmålet er ennå ikke innført; se presiseringen i målplanen.
Felles A/B-kjerne er nå implementert: eksakt klokke, samme TRAIN/eval-rader,
bevarte prishorisonter og B-minus-A i felles inferens. Fem fokuserte mekaniske
tester besto på syntetiske data. Kildeimport/fullt panel, ekte registrert B-kjøring
og native mål/trening er fortsatt ufullført. Ingen ny markedsmåling er gjort.
Full B er delvis implementert og umålt; build/trening/evaluering er ikke klare.
Native trening/TEST er fortsatt stengt ved dagens uoppfylte porter.
Ingen løfte om kunstig historikk, lønnsomhet eller et grønt treningsvedtak.

## Fullført neste datasteg for B — 30.09.2026

[Oppfølgingen](docs/TA_B_NEXT_SOURCE_STEP_20260930.md) har hentet og kontrollert
alle fire ALFRED-arkivene: 27 filer og 11 249 valgte historiske versjonsdatoer.
Offentlig nettlesereksport fungerer uten API-nøkkel. Alle fire har bestått
endelig kildeaudit; ingen motstridende duplikater eller overlappende intervaller.
Originalfiler, to avviste metadata-kontroller og den endelige kontrollen er bevart.

Full B er fortsatt umålt. GLD/COT har ikke tilstrekkelig historisk versjonsdekning;
de nye arkivoppslagene ga ni HTTP 429 og én HTTP 503. Nye oppslag er stoppet.
Henteeieren stopper nå resten av en batch etter HTTP 429; fokusert test besto.
Ingen redusert kildevariant, B-fit, TEST, trening, handel eller spending er åpnet.

Dollarindeksens første vintage er 04.02.2019. Ingen tilbakefylling til tidligere
beslutninger. Etter kildekvalifisering må A/B sammenlignes på samme faktiske
TRAIN-/eval-populasjon med publikasjonslag og warmup, etter ny forhåndsregistrering.
Neste datagrense er dokumenterbar GLD-tonnasje og COT 088691 Legacy Futures Only.
Ferdige nedlastinger og kontroller gjenbrukes; ingen jobb kjører og ingen relanseres.

## Historikk: tidligere kildeundersøkelse før nettlesertilgangen ble løst

Brukeren ba 30.09.2026 «Ja undersøk B».
[Undersøkelsen](docs/TA_B_SOURCE_INVESTIGATION_20260930.md) hentet og kontrollerte
ekte historiske GLD- og COT-filer. Full B er fortsatt umålt: to enkeltkopier
dokumenterer ikke sammenhengende publiserings-/versjonsdekning. Den kontrollerte
COT-adressen har bare én arkivkopi 01.03–07.04.2019; GLD-indeksen fikk timeout.
ALFRED-skjemaet virker på Mac, men POST fikk fortsatt timeout etter rettet
submit-felt og 60 s grense. To fokuserte tester besto.

Tre tidligere kildeprøver er avsluttet og hashkontrollert. Makrotilgangen som
manglet i denne historiske undersøkelsen er nå løst i oppfølgingen ovenfor.
Leverandørens generelle vintagefunksjon alene er ikke godkjent GLD/COT-dekning. Ingen B-fit eller redusert
kildevariant er åpnet; A/C-resultatene bevares. Native trening, TEST, handel og
spending forblir stengt.

## Fullført autorisert forberedelse

V37-inputbygging, fersk post-rebuild/readiness, TRAIN/VAL lifecycle-filbindinger,
repo-gjennomgang/testtriage og den bestilte kompleksitetsvurderingen er fullført.
Sluttrapport:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/FINAL_PREPARATION_REVIEW_20260929/FINAL_PREPARATION_REPORT.json`.
Les eksakte bevis via NEXT_RUN_POLICY.json. Alle videreføringer er konsumert.
Bevar opprinnelig RED og ny eksplisitt readiness-recovery; ingen ny rebuild.

- [Kompleksitet og redundans](docs/FEATURE_COMPLEXITY_REVIEW_20260928.md).
- [Repo-dekning, rettelser og teststatus](docs/REPO_REVIEW_20260928.md).
- [Byggets feilrettinger og historikk](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md).

Ingen ny fullsuite eller gjentatt bestått datakontroll uten konkret nytt funn.
Alle features, åtte familier og tidsrammer beholdes. Ingen data/checkpoints slettes;
ekstern diskopprydding krever retention-eierens rekkeviddebevis og godkjente plan.

## Bevar de fullførte målingene

[Samlet beslutning 29.09](docs/TA_RESEARCH_DECISION_20260929.md): A/C er
INKONKLUSIV og B er umålt. Oppfølgingen 30.09 gjelder bare kilder. Ingen omkjøring,
ny native kontrakt, utførelsesforskning, modell- eller terskelsøk er åpnet.

## Fullførte historiske spor

- [Konsolidering og tidligere sletting](docs/CONSOLIDATION_20260926.md).
- [Tidlig kalibrering og historisk beslutningskontroll: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
- [Modellfrie baselines](docs/MODEL_FREE_BASELINES_RESULT_20260927.md),
  [makrohendelser](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md) og
  [intradag-mekanismer](docs/INTRADAY_MECHANISMS_RESULT_20260927.md).

Disse planene skal ikke relanseres. Målet om kostnadsjustert edge er ikke oppnådd.
