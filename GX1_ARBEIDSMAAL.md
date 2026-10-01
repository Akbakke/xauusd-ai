# GX1 arbeidsmål — oppdatert 01.10.2026

## Native v38 — M1-klokker kontrollert, featurebinding og læring gjenstår

Inputbygg og etterkontroll er ferdige; ikke relanser dem. Læringsdesignet er
nå fryst med fysisk TRAIN 01.06.2011–31.05.2025, senere utviklingskontroll
01.06.2025–30.06.2026 og 256 faste kontrollrader. Kontroll er gjenbrukt
utviklingsdata. Alle 254 felt og åtte familier består.

En konkret gammel kalenderbinding i klargjøringen er rettet i eksisterende
eier. 15 fokuserte tester består; faktisk ny eier har kontrollert samtlige
652 552 TRAIN- og 70 880 VAL-tidsstempler mot det frosne designet.
Dette er kalender-/populasjonsbevis, ikke full native admission eller læring.

[Design, feilfunn og neste steg](docs/NATIVE_V38_BOUNDED_LEARNING_20261001.md).
Full før-TEST-kontroll av M1-forslaget er ferdig: 5 959 045 M1-rader
gjenskaper alle 1 215 514 M5-barer fra juni 2009 til juni 2026 eksakt i alle
13 markedsfelt. 2024-beviset er gjenbrukt. Eksisterende komplett M1-quote-fil
er også verifisert av den kanoniske eieren; ikke bygg en kopi.
[Aggregert kildebevis](docs/NATIVE_M1_SOURCE_PARITY_20261001.json).

TEST-seal-admission er nå rettet i pilotens eksplisitte designsti: 21 tester
og ekte metadata-kontroll består. Feil TEST-filpeker avvises før stat/hash/read.
Den komplette M1-filen er nå bundet gjennom sin ekte pre-TEST-parent i en
konkret klargjøringsrecipe. 144 tester og full klargjøringskontroll på de ekte
652 552 TRAIN-/70 880 kontrollradene består kilde-, kalender- og seal-portene.
TEST og original råkilde med TEST-rader fikk null tilgangsforsøk.
Full native admission og M5-produksjonskildebytte er fortsatt ufullført.
Entry-adoption og separate fysiske M1-visninger er nå publisert og kontrollert.
Alle 652 552 TRAIN- og 70 880 kontrollrader finner eksakt første M1-tilstand.
Samtlige markedsfelt i M1-visningene er identiske med råkildens respektive rader.
Entry gjenbruker 21,16 GB ferdige bytes uten kopi. 68 fokuserte tester består.
Normaliseringsadgangen godtar de faktiske radantallene fra det fryste designet;
ingen normalisering er tilpasset. Markedspauser og observerbar M1-tilstandsstøtte
er nå kontrollert på de nye klokkene. 34 gjentakende pauseregler er tilpasset bare
på TRAIN før 01.06.2025 og brukt uendret på kontrollen. Alle 256 fryste
kontrollpunkter har minst 339 observerte etterfølgende overganger; dette er
klokkestøtte, ikke mål-, feature- eller læringsbevis.
En gjenværende gammel TRAIN-sluttdato i normaliseringspopulasjonen er rettet,
og publisering av pauseartefakter overskriver aldri eksisterende filer/kataloger.
13 fokuserte tester består. Eksisterende M1-featuremanifest peker fortsatt på
gammel filtrert base28; bindingen matcher ikke komplett M1. Neste steg er å
bygge bare M1-featureflaten mot komplett før-TEST-klokke gjennom eksisterende
eier, med gjenbruk av ferdig beriket M1 og fryste parametre. Den separate
forhåndsregistreringen M1_FEATURE_REALIGNMENT_001 gir ingen modell-/fit-adgang.
Runtime START/TERMINAL avgjør faktisk status; ikke start en ny kopi.
Den delte berikede kilden inneholder også senere inputrader og blir fullhashet
og kausalt transformert av byggeieren. Alle nye utdatarader skal være før TEST;
forseglet TEST-datasett/manifest og utfallsmåling forblir utilgjengelige.
Gammel lærer fjerner dagens parameterfrie normalisering; ny fersk måling skal
bruke samme aktuelle funksjon for ONLINE og TARGET. Indekser, normalisering
og mål-/økonomibindinger gjenstår. Native trening og normaliseringsfit er
fortsatt stengt; TEST er urørt. Eldre neste-steg-tekst er historikk.

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

Målet er en fullstendig automatisk XAUUSD-bot som tar retning på en dokumentert
nyttig tidsskala og slår relevante baselines etter kostnad. Målet er aktivt og ikke
oppnådd; dagens godkjente arbeidsomfang er fortsatt offline forskning.

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

## Suksesskriterier

Senere LONG/SHORT/FLAT-valg må slå relevante kausale baselines etter kost, gjennom
flere markedsperioder. Økonomi inkluderer alle valgte handler og åpne posisjoner,
utførbare BID/ASK-priser og kostnader. TRAIN-fit, senere generalisering og samlet
økonomi rapporteres hver for seg. Konstant bias, all-FLAT/all-HOLD, teknisk PASS
og bedre hjelpeprognoser alene er utilstrekkelig. Tidligere tidsskalamålinger
beskriver de undersøkte oppsettene; de beviser ikke at en hel markedstype er ulærbar.

## Bevares

Alle features, alle åtte familier, alle tidsrammer og kausale inputs. Gjennomførbar
BID/ASK-økonomi og kostnader. Ingen fast tapsgrense eller maksimal holdetid; en
beregningshorisont er ikke en handelsregel. TEST er forseglet; ingen live/paper eller
spending. Ingen modell loves å være lønnsom «evig».

Én agent og én tung jobb om gangen innen CURRENT, gjennom `scripts/gx1_capped_run.sh` og eksisterende
vakter. Ingen blind trening, søk eller forebyggende refaktorering; mål før du bygger.
Stående publiseringsautorisasjon gjelder ferdig kode, dokumentasjon og aggregater; rådata,
vekter og hemmeligheter publiseres aldri.
