# Gjeldende status — 01.10.2026: native v38 og kontrollert kodeopprydding

Sluttbindingens første forsøk feilet før publisering: én ekte TRAIN-rad,
12.12.2012 kl. 17:00 UTC, har BID=ASK. Ingen kryssede priser eller
float32-kollaps ble funnet på 652 552 TRAIN-/70 880 VAL-førstetilstander.
Entry-fill-eieren avviste likhet selv om de øvrige aktive kontraktene
aksepterer ASK>=BID. Minste rettelse er <= til <; positive, endelige priser
kreves fortsatt. 22 fokuserte tester består, inkludert ekte bridge-kode
med nullspread og fortsatt avvisning av kryssede/ugyldige priser.
FINAL_BINDINGS_001 og feilen er bevart. FINAL_BINDINGS_002 er bundet
etter rettelsen; base-/lifetime-fit gjenbrukes uendret. Ingen nye modellsteg.

Lifetime-normaliseringen er fullført med exit 0 og strict-load PASS.
4 026 919 samples fra alle 652 552 TRAIN-entryer gir 8 053 838 siderader.
TRAIN-/VAL-counts matcher tidligere kontrollert geometri eksakt; VAL fikk
ingen fit. Kilden var uendret, TEST-tilgangsforsøk = 0, modellsteg = 0.
29 fokuserte tester består. Base- og lifetime-statistikk skal nå gjenbrukes.
Sluttbindingens samme no-replace-publiseringsfeil er også rettet; 7 fokuserte
tester består. FINAL_BINDINGS_002 er bundet til samlet normalisering og
første M1-tilstand for hver Entry; dette åpner ingen fit eller modellkjøring.
Kostnadsdekning/broker-revalidering, indekser og mål gjenstår. Ingen edge.

Normaliseringskoden er nå bundet til de faktiske full-TRAIN-artefaktene før
statistikkfit. Feil vitne, byttede filer, feil MTF-sti og endrede bytes avvises.
Den eksisterende diskbaserte M1-innleseren erstatter store RAM-kopier;
historikkunion beregnes per sammenhengende intervall i stedet for per rad.
25 fokuserte tester består, inkludert eksakt normparitet og navnekollisjon.
BASE_NORMALIZATION_FIT_001 er fullført med exit 0 og strict-load PASS:
alle 254 signalfelt, kontekst og M5/M15/H1/H4/D1 er tilpasset på fryst TRAIN.
5 748 166 lokale feature-rader og 4 647 700 kontekstrader inngår; null VAL/TEST.
Alle MTF-utvalg er tilgjengelige før TRAIN-slutt. Ingen modellforward/optimizer.
Basefitten skal gjenbrukes; tillatelsen er brukt opp. Før indeksbygg gjenstår
lifetime-statistikk, no-cap-økonomibinding og samlet norm-/førstetilstandsbevis.
Native trening og TEST er stengt; ingen edge er dokumentert.


Hele TRAIN-sekvenskontrollen er fullført: alle 652 552 seq/snap-rader
matcher bundet M5-flate eksakt. Terminal exit 0, uendret kilde og null
VAL-/TEST-datasettilgang. Gjenbruk TRAIN_SEQUENCE_AUDIT_001; ikke relanser.
Normaliseringsforberedelsen gjenbruker nå dette beviset bare ved eksakt
fil-/hash-/populasjonsbinding; endrede bytes eller bevis avvises.
Kun M5-klokken lastes der signalverdier allerede er kontrollert.
19 fokuserte tester består. NORMALIZATION_POPULATION_001 er fullført med
exit 0: hele TRAINs 652 552 entryer gir 955 670 unike M5-kontekstrader og
3 995 148 unike observerbare M1-tilstander. M1-unionen matcher uavhengig
den tidligere kontrollerte geometrien eksakt. Denne populasjonskontrollen
tilpasset ingen statistikk; basefitten øverst er fullført senere.
Gjenbruk de tre publiserte inputartefaktene; ikke relanser produsenten.
Neste steg er tilstandsindekser og binding av normalisering/mål med separate
TRAIN-/kontrollkoordinater. Native launch, læring og økonomi gjenstår.

## Native v38 — komplett M1-featureflate kontrollert, læring gjenstår

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
13 fokuserte tester består. M1_FEATURE_REALIGNMENT_001 er nå fullført med
exit-kode 0. Den nye flaten har 5 523 147 rader og alle 254 ordnede felt,
eksakt komplett før-TEST-klokke etter kausal warmup og null manglende rader i
både TRAINs 4 884 638 og kontrollens 382 744 M1-rader. Originale beregninger,
registry-/squeeze-parametre og åtte familier er bevart. Ikke relanser bygget.
Den delte berikede kilden ble fullhashet og kausalt transformert, også med
senere inputrader; alle nye utdatarader er før TEST. Forseglet TEST-datasett
og -manifest fikk null tilgangsforsøk; ingen modell, målfit eller tuning.
Den verifiserte flaten er nå bundet i normaliseringspopulasjonen.
Tilstandsindekser, statistikkfit og målforberedelse gjenstår.

PC-en ble kontrollert omstartet etter fullført jobb og verifisert tomme
prosjekt-/GPU-køer 01.10 kl. 20:22 UTC (22:22 Oslo). WSL, GPU, cgroup-vakter,
feature-footer og kvitteringer er kontrollert etterpå. Bruk gx1-3090-lan nå;
Tailscale-ruten gx1-3090 svarte fortsatt ikke, selv om tjenesten kjørte.
Kontrollerte omstarter mellom ferdige, maskinfelles ledige kjøringer er
nå stående operatørinstruks. Aldri avbryt en aktiv jobb for periodisk omstart.
Gammel lærer fjerner dagens parameterfrie normalisering; ny fersk måling skal
bruke samme aktuelle funksjon for ONLINE og TARGET. Indekser, normalisering
og mål-/økonomibindinger gjenstår. Native trening er
fortsatt stengt; base-normaliseringsfitten øverst er fullført. Eldre neste-steg-tekst er historikk.

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

## Kodegjennomgang 01.10 — handover rettet og frakoblet kode fjernet

Handover-pekeren er rettet fra full B til det gjeldende native v38-sporet.
Tre ubrukte hjelpere er fjernet; gjenværende funksjoner og klasser i de berørte
kodefilene har identisk AST. 222 fokuserte tester består. Alle 589 sporede
Python-filer og 139 JSON-filer består strukturkontroll.
Detaljer og avgrensninger: [repo-gjennomgangen](docs/REPO_REVIEW_20260928.md).
V38-datasett, normalisering, læring og ny train/serve-paritet gjenstår.
Trening/TEST/handel er fortsatt stengt; ingen nye markedsmålinger er startet.

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

**Les først:** [GX1_RULES.md](GX1_RULES.md) (bindende regler), [AGENTS.md](AGENTS.md)
(arbeidsmåte), [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) (mål og vedtak) og
[VEIEN_VIDERE.md](VEIEN_VIDERE.md) (eksakt neste steg). `bash scripts/gx1_handover.sh --check`
overstyrer prosa.

## Avgrenset oppfølging 30.09: indikatorrevisjon og OANDA-prøve

Brukeren bestilte en egen skrivebeskyttet agentrevisjon av liquidity sweep,
order flow, anchored VWAP og volum. Revisjonen av HEAD 3947c778 fant aktiv sweep
og rullerende tickvektet VWAP; ingen aktiv hendelsesankret VWAP eller ekte
aggressor-/ordrebokflyt. Tre aktivitetsfelt varierer i de gjenbrukte 652552
TRAIN-snapshotene. Sju fokuserte syntetiske tester besto; dette er struktur- og
inputbevis, ikke lærings- eller lønnsomhetsbevis. Modell/kilde er ikke endret.
Audit: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/LIQUIDITY_VOLUME_AUDIT_20260930_001/AUDIT.json
(SHA256 7548b539559c6ce2e622acf9a425ce1c05de80063afae565ea44466c7b90433c).

Brukeren godkjente uttrykkelig én manifestbundet, skrivebeskyttet COT-forespørsel
mot OANDA practice med eksisterende demo-token. Manifestet
configs/research/TA_B_OANDA_COT_PROBE_20260930.json ble committet i 6c0402c5 før
henting. Den ene forespørselen ga HTTP 403 og en Cloudflare-blokkeringsside;
ingen COT-data ble mottatt. Dette avgjør ikke tokenens gyldighet, kontorettigheter
eller om den eldre COT-tjenesten fortsatt fungerer. Ingen automatisk retry eller
alternativ rute er startet. Tillatelsen omfattet kun denne prøven.
Resultat: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/B_OANDA_COT_PROBE_20260930_001/RESULT.json
(SHA256 51f8903bb88189dc63b219e6b4addf33290f8d438ecd30b8f2b4145ce1bf0da5).
Full B er fortsatt blokkert på kildebevis; ingen data er tatt inn som input,
og trening/TEST/handel er ikke åpnet.

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

## Historikk 29.09: A/B/C-planen avsluttet; ingen ny måling autorisert

[Samlet beslutning](docs/TA_RESEARCH_DECISION_20260929.md): A og C er INKONKLUSIV;
B er ikke målt grunnet konkret kildebegrensning. Den avtalte forskningsplanen er
avsluttet med dokumenterte resultater/grenser. Botens lønnsomhet er ikke etablert.
Ingen GO til ny native kontrakt/trening eller utførelsesforskning.
Neste mulige operatørvalg er dokumenterbar historisk GLD/COT-versjonsdata før
eventuell gjenåpning av full B. Ingen automatisk ny arm eller henting.



Brukeren vedtok [forskningsplanen](docs/TA_RESEARCH_PLAN_20260929.md); den er nå gjennomført. [Rotårsaksrapporten](docs/EDGE_ROOT_CAUSE_REVIEW_20260929.md) er publisert med
kontrollert dommetelling og spreadpresisering; originalen er bevart. Regel 1 åpner
navngitte, manifestbundne eksterne forskningsinputs med XAUUSD som eneste eksponering.

Dokumentasjonsbølgen er pushet som 52c8761e. Første instrumentrettelse er kontrollert:
ridge-gridet går nå til 1e7, og rapporten skiller ren ridge, konstantvalg og valg
på søkegrensen. Konstanten beholdes som synlig kausal referanse. Eierens fokuserte
tester besto gjennom capped audit, inkludert uavhengig sklearn-sammenligning.

[Økonomi- og inferensinstrumentene](docs/TA_RESEARCH_INSTRUMENTS_20260929.md) er
nå mekanisk kontrollert: sammenhengende beholdning med utførbare kostnader,
historisk rentekurve med signert SHORT-kreditt, åpne posisjoner, kausal
risikoskalering, Sharpe/drawdown, paret stasjonær bootstrap og felles max-t med
styrke/MDE og treveis effektbeslutning. De gamle registrerte kjøringene er bevart.
Bevis: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/ECONOMICS_INFERENCE_VERIFICATION.json.

[Måling A er fullført](docs/TA_A_RESULT_20260929.md): ridge og HGB er begge
INKONKLUSIV etter den fryste primære beslutningsregelen. 15 årsfolds per horisont,
4 003 prognoser og 4 002 porteføljeintervaller på gjenbrukt 2011–2025-historikk.
Ingen GO; ingen native treningsåpning. Ikke relanser eller tune A.

Etter historisk EFFR-finansiering gir h20 ridge +14,50 %, HGB +59,21 %
og samme risiko-LONG/kausale konstant +61,04 % samlet netto av initialkapital.
Primær meravkastning mot risiko-LONG er -0,67 / +0,018 bps per intervall,
med brede simultane intervaller. Positiv netto alene er ikke indikatorverdi.
Finansieringsproxy, varierende D1-kildedekning og gjenbrukt historie begrenser tolkningen.

Kildecommit 0dc4a04b; rapporten er pushet i c41ed952. A-resultat og alle 30
artefakter er verifisert.
Terminal: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_A_001/TERMINAL.json.
Verifikasjon: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_A_001_VERIFICATION.json.
Cached audit bekreftet de ytre fit-klokkene uten ny fit.
Direkte EFFR har 4 394 kontrollerte virkedager. Begge FRED-timeouter og
første EFFR-skjemaavvik er bevart; faktiske bytes ble gjenbrukt med bundet percentRate-felt.
[B er avsluttet som ikke målt](docs/TA_B_RESULT_20260929.md): de fire ALFRED-kallene
fikk timeout; Mac-kontroll av skjemaet lyktes. Avgjørende begrensning er manglende
bundet historisk publikasjon-/versjonsbevis for GLD/COT. Ingen B-fit, redusert
erstatning eller konklusjon om null makroeffekt. Den terminale kildeauditen bevares.

[C er fullført](docs/TA_C_RESULT_20260929.md) fra dab10749 og kontrollert:
1 649 muligheter /1 646 utførbare,1 291 passive berøringer. Alle53 artefakter
og alle berøringer er verifisert;24 kost-/porteføljeregnskap er avstemt.
Aktiv mid+1,465 bps blir netto−2,547 ved1 bps per utførelse og finansiering.
Passiv netto−1,397 bps per valgt mulighet,−14,91 % av initialkapital.
Berøringsutvalget har−0,751 bps mid, utførbare uten berøring+9,536:
gunstigere inngangspris gjenopprettet ikke det opprinnelige signalutvalget.
Primær passiv grense mot FLAT[−4,480;1,686], mot LONG[−3,332;5,179]:
INKONKLUSIV og ingen GO. Hele juni2025–juni2026 er gjenbrukt utviklingsevidens.
TERMINAL:/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_C_001/TERMINAL.json.
Etterkontroll:/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_C_001_CACHED_AUDIT.json.
Ingen åpne simulerte posisjoner ved slutt; ingen TEST-utfall eller native fit.
Ikke relanser A/C eller tune reglene på resultatene.
Mekaniske tester er ikke prognose- eller lønnsomhetsbevis.
Verifikasjon: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/RIDGE_REPAIR_VERIFICATION.json.
Planen alene er ikke en preregistrering.
Dokumentasjonsbevis:
 /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/DOCUMENTATION_VERIFICATION.json.
NEXT_RUN_POLICY.json og RUNNING_NATIVE_CALIBRATION.json binder samme nye status.

training_enabled=false: ingen native optimizer, native trening, full native VAL,
TEST-utfall, handel eller spending. Avgrensede forhåndsregistrerte CPU-fits og
navngitte forsknings-VAL-målinger er tillatt gjennom capped audit/producer.
V37-inputforberedelsen er fullført og skal gjenbrukes, ikke relanseres.

## Historisk operatørvedtak: ferdigstill inputs, revider hele repoet, ingen trening

Brukeren har autorisert [native-forberedelse og full repo-gjennomgang](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md)
før eventuell trening. Dette overstyrer tidligere forbud mot videre datasetbygging.
Den konsoliderte native modellen med de nye inputene er ikke epoch-trent.
Ridge/HGB-porten under er avsluttet forskning, ikke et bevist tak for native læring.
SMC-rettelsen og prosjektvis låsing er kontrollert. Tidligere kilde-/testgjennomgang
er dokumentert; fullført v37-inputbygging og resterende beslutningsgrense står nedenfor.
Trening forblir deaktivert. Eksakt scope står i native-forberedelsens `PLAN.json`.

## Fullført historikk: autorisert inputforberedelse og gjennomgang; trening er stengt

V37-datasettet er ferdig. Fersk etterkontroll og readiness besto 29.09 kl.
03:13 UTC / 05:13 norsk tid; lifecycle-eierens faktiske TRAIN/VAL-filadgang
besto kl. 03:53 UTC / 05:53 norsk tid. Alle prosesser i denne videreføringen
er avsluttet med terminalkvittering. Ingen bygging eller bestått audit skal gjentas.

Sluttrapport og maskinbevis:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/FINAL_PREPARATION_REVIEW_20260929/FINAL_PREPARATION_REPORT.json`.
Eksakt hash står i NEXT_RUN_POLICY.json. Datasettene ligger i den samme
`DATASET_REGISTRY_BINDING_RECOVERY_20260929`-roten. TRAIN har 652 552 rader;
VAL har 70 880. TEST er kun bundet gjennom completion og den uendrede forseglingen.
Original RED-kjede bevares: builder var fullført, mens den gamle etterkontrollen
feilet på fit-klokken. Ny readiness er en eksplisitt godkjent completion-recovery,
ikke en omskrevet GREEN-kjede eller ny datasetbygging.

Lifecycle-kontrollen brukte eksisterende `UnifiedExitLifecycleCorpus._require_file_admission`
med bare TRAIN/VAL. Den validerte faktiske input-/lifecycle-hasher, splitgrenser,
M1-kilde/featureflate og manifester. TEST-splitfilene ble verken åpnet eller statet;
ingen modell, expanded state replay eller utfallsøkonomi ble kjørt.

[Kompleksitetsvurderingen](docs/FEATURE_COMPLEXITY_REVIEW_20260928.md) er nå
fullført innen det bestilte omfanget. Alle 242 signalfelt er målt på alle
652 552 emitterte TRAIN-beslutninger: ingen globale konstanter eller eksakte
innbyrdes duplikater; 71 kontekstaliaser er eksakt like. Sterke korrelasjoner
og sjeldne hendelser er dokumentert uten automatisk featurefjerning.
Kildens parametertelling er 9 633 055 ved eksisterende referansedybde
(Ls=1, Lm=2); eksakt formel for annen eksplisitt dybde er rapportert.
De åtte hjelpeprojeksjonene har 6 450 parametre, 0,067 prosent av totalen.
Hjelpeoppgavenes merverdi er fortsatt ubevist; encoderne eier mest beregning.

Tidligere full repo-inventering/testtriage gjenbrukes. Nytt inventar binder
852 sporede filer og 49 endrede stier fra forrige inventarkilde; endret Python/JSON
består strukturkontroll. De fokuserte reparasjonstestene og faktisk fullført
bygg/readiness/lifecycle utgjør ny evidens. Ingen ny fullsuite er kjørt.
Se [repo-rapporten](docs/REPO_REVIEW_20260928.md) for presis dekning og begrensning.

Den tidligere forberedelsesgrensen krevde en egen native forskningsbeslutning.
Gjeldende neste steg er A/B/C-planen ovenfor; ingen native kjøring er vedtatt.
Det er ingen godkjent v37-treningsrecipe eller treningsstart. Fersk TRAIN-
normalisering, native konstruksjon og senere train/serve-paritet hører til den
bundet kjøringens eksisterende eiere og er ikke erklært utført her.
`training_enabled=false`; optimizer, full VAL, TEST-utfall, handel og spending
forblir stengt. Ingen features, familier eller heads er fjernet. Teknisk
konsistens er ikke dokumentert edge. Denne inputforberedelsens automatiske
oppfølging skal avsluttes etter commit/push og overleveringssynk.

## Historikk: fit-klokkereparasjonen i etterkontrollen

Kjøringen på `abd9c2c9` publiserte dataset-completion og TEST-forsegling 02:37 UTC.
Etterkontrollen stoppet 02:44 UTC fordi tre lesere bare anvendte den felles
M5-fit-klokken ved PRETEST. `ca6106a9` koblet alle tre til samme eier som
byggeren, uten endring av data, features, fit eller mål. 34 målrettede target-/
klokketester, 15 readiness-tester og faktisk TRAIN/VAL-manifest-/ECDF-kontroll
besto. Post-rebuild-kjøringen er fullført og skal ikke relanseres.

## Historikk: split-manifest binder den vedtatte tidlige kalibreringen

`CONTINUE_LOCKED_QUOTES_ROOT_20260929` lukket TRAIN-skriveren med 652 552
rader og passerte lifecycle-byggingen. Den stoppet 29.09 kl. 00:13 UTC /
02:13 norsk tid ved split-manifestet: en eldre kontroll krevde at registry-fit
skulle være lik hele modellens TRAIN-vindu. Den autoriserte kjeden binder
allerede tidligere fit: 2009-06-01 til 2013-01-01 22:00 UTC, med indre
fit-slutt 2012-04-11 22:00 UTC. Disse kalibrerte verdiene endres ikke.

Split-kontrollen bruker nå den uendrede registry-eieren, krever eksakt likhet
mellom alle frosne konstanter og den hashbundne cache-manifesten, og avviser
fortsatt fit etter modellens TRAIN-slutt. Målrettede manifest-/rebuild-/
kjedetester består, inkludert avvisning av VAL-lekkasje og endrede cachebevis.
Faktisk cache og denne kjøringens byggbevis består den eksakte split-kontrollen.
Builder-kode utenfor split-validatoren er AST-identisk; priser, features,
mål, splitgrenser og modell er uendret. Dette er teknisk evidens, ikke edge.

Ny engangsplan:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_REGISTRY_BINDING_20260929`.
Fersk outputrot:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928/DATASET_REGISTRY_BINDING_RECOVERY_20260929`.
Den nyeste røde terminalen og dens preflight er eksplisitt bundet. Ferdige
upstream-inputs i opprinnelig CHAIN gjenbrukes bare etter fysiske hasher og
kildekontroll; ny preflight er obligatorisk. Det ufullførte datasettet og alle
kvitteringer bevares. De har ingen ferdigmanifest og kan ikke godkjennes som
split-input ved å kopiere filer.

Normal commit-kontroll ble avvist av minnevakten: mindre enn 20 GiB ledig RAM.
Eksisterende engangskø venter hvert 15. minutt på uendret minnekrav, binder
nøyaktig staged diff/kilde/runtime, og gjør normal commit/push før dispatch.
Ingen hook omgås. QUEUE_BINDING/QUEUE_STATUS/QUEUE_TERMINAL viser dette trinnet.
Kilden og dokumentdiffen er frosset også mens køen venter.

Les PLAN/BINDING/WAITING/START/TERMINAL og samme runs prosesser. Ikke dupliser
konsumert plan. Kilde er frosset under venting/kjøring, og alle ressursvakter er
uendret. POST_BUILD følger bare grønn kjede; automatisk oppfølging består.
Readiness, lifecycle-bindinger og resterende kompleksitetsvurdering gjenstår.
Trening, optimizer, full VAL, TEST-utfall, handel og spending er fortsatt stengt.

## Historikk: lifecycle godtar samme låste priser som kanonisk tape

`CONTINUE_SUMMARY_MEMORY_20260928` lukket TRAIN-skriveren med 652 552 rader,
uten nytt minnedrap. Kjøringen stoppet 28.09 kl. 22:20 UTC / 29.09 kl. 00:20
norsk tid: lifecycle avviste M1-rader med BID lik ASK. Ingen ferdigmanifest
ble publisert; den lukkede TRAIN-parqueten er fortsatt ikke et godkjent datasett.

Kanonisk tape, Entry-utfall og optimal-stopping-eieren tillater allerede BID lik
ASK. Lifecycle-bygger, lifecycle-leser og offline replay bruker nå samme regel.
Kryssede priser, ugyldige tall og OHLC-brudd avvises fortsatt. Ingen priser,
features, mål, kostnader eller splitgrenser endres. 44 fokuserte tester består.
Faktisk TRAIN-vindu fra desember 2012: 2 732 rader og fire lifecycle-episoder
består tape-/bygger-/leserkontroll; alle priser er eksakt bevart. Kontrollen
brukte ikke TEST og måler ingen handelsverdi.

Ny engangsplan:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_LOCKED_QUOTES_ROOT_20260929`.
Ny outputrot:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928/DATASET_LOCKED_QUOTES_RECOVERY_20260929`.
Nyeste røde kjede og dens preflight er bundet som forelder; ferdige inputs
beholdes i den opprinnelige CHAIN-roten. Gjenbrukseieren kontrollerer eksakte
hashverdier og uendret upstream-kode. Bare de navngitte downstream-funksjonene
kan avvike; delte helpers/imports er AST-identiske. Ny preflight er obligatorisk.

Første oppstart (`CONTINUE_LOCKED_QUOTES_20260929`) ble avvist før kjeden
startet fordi den tomme outputmappen manglet. Mappen er nå opprettet, uten
data; nytt launcher-omfang er bundet ovenfor. Avbrutt oppstart bevares.

Les nye PLAN/BINDING/WAITING/START/TERMINAL og prosesser før videre handling.
Tidligere forsøk og delvise outputs bevares. Ny POST_BUILD venter på denne
launcherens grønne kjede. Kilden fryses under venting/kjøring. Minnegrenser og
alle vakter er uendret. Automatisk oppfølging hver 30. minutt består.
Trening, optimizer, full VAL og TEST-utfall er fortsatt stengt. Post-rebuild,
lifecycle-bindinger og gjenstående kompleksitetsvurdering gjenstår.

## Historikk: rettet minnevekst i datasettskriving

`CONTINUE_RESOURCE_WAIT_20260928` passerte de tidligere blokkeringene, men ble
OOM-drept i sin 10 GiB cgroup kl. 20:51 UTC / 22:51 norsk tid. Siste loggførte
TRAIN-flush var 496 640 rader. Delvis parquet, Group-A-checkpoints og alle røde
kvitteringer bevares. Ingen split-manifest/ferdigkvittering godkjenner dette datasettet.

Skriveren beholdt én dictionary med 97 identitets-/labelkolonner per rad. Den
beholder nå kolonnebaserte batcher. På 100 000 syntetiske rader med faktisk feltsett
falt beholdt summary-lagring fra 712 800 984 til 77 625 088 bytes (89,1 %).
Verdier, datatyper og rekkefølge er eksakt like. Dette måler representasjonen,
ikke toppminnet i en full rebuild. 22 fokuserte tester består. Parquet-skriving,
features, labels, splitgrenser og minnetak er uendret.

Ny engangsplan:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_SUMMARY_MEMORY_20260928`.
Ny outputrot:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928/DATASET_SUMMARY_RECOVERY_20260928`.
Eksisterende kjedeeier kan nå motta en separat, eksplisitt tidligere inputrot.
Den rehasher ferdige upstream-inputs, krever uendret produsentkode og tillater
bare endret downstream-datasettfunksjon/summary-hjelper. Delte helpers/imports
kontrolleres AST-identiske. Ny preflight kreves, og alle nye downstream-outputs
må være ferske. Gammel preflight er kun kildebevis; rødt blir aldri grønt ved kopiering.

Les PLAN/BINDING/WAITING/START/TERMINAL og prosesser i den nye runtime-roten.
Tidligere plan er konsumert. Ny POST_BUILD bindes til den nye launcheren og den
nye outputroten. Kilden fryses under kjøring. Automatisk oppfølging hver 30. minutt
består; trening, optimizer, full VAL og TEST-utfall er fortsatt stengt.

## Historikk for inputbyggingen nedenfor

## V37: ferdige featureflater; rettet preflight-leser og bundet videreføring

Kjøringen stoppet 28.09 kl. 13:27 UTC / 15:27 norsk tid etter ferdig M1/M5.
Alle 1 382 M1-chunks og featureflatene er ferdige: M1 5 570 522 rader, M5
1 152 859 rader, begge med 242 felt og PASS-manifest. Preflight har 29/29
beståtte kontroller og `READY_FOR_MODEL_NATIVE_SEQ513_REBUILD`.
Selve datasett-/lifecycle-rebuilden var ikke startet.

Den konkrete blokkeringen var kjedens filkontroll: publiseringseieren skriver
både JSON-hendelsen og obligatorisk `.json.order`-kvittering, men leseren krevde
én katalogoppføring. Leseren bruker nå eksisterende immutable-event-eier og
aksepterer bare én hendelse med gyldig kvittering. 14 kjedetester består,
inkludert ekte publisering og avvisning av endrede/manglende bevis, samt faktisk
kontroll av denne kjøringens preflight. Ingen feature-/modell-/dataprodusent er endret.

Første videreføring lå i
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_PREFLIGHT_ORDER_20260928`.
PLAN/BINDING/START/TERMINAL der eier ny kjøring. Eksplisitt preflight- og rød
forelder-hash kreves; kode under gx1 og øvrige scripts må være identisk med
forelderens commit. Alle gjenbrukte filer rehashes, eksakte perioder/stier
kontrolleres og senere outputs må fortsatt være ferske. Dette gjenbruker de
ferdige v37-inputene uten ny M1/M5-bygging. Gamle terminaler bevares.

Tidligere POST_BUILD stoppet korrekt på rød forelder og skal ikke gjenstartes.
Ny post-build-videreføring skal bindes til den nye launcherens PID/kilde og bare
kjøre eksisterende readiness etter grønn kjede. Automatisk oppfølging i samme
Codex-oppgave er opprettet hver 30. minutt (`gx1-v37-oppf-lging`): stabil drift
skal være stille; feil og ferdigstillelse følges opp innen samme autoriserte scope.
Det er ingen treningsautorisasjon. Under kjøring er kilden frosset.

## Ressursventing etter første videreføring — 28.09 kl. 19:48 UTC

`CONTINUE_PREFLIGHT_ORDER_20260928` besto den fysiske gjenbrukskontrollen,
men stoppet før model-source-identity på uendret krav om minst 20 GiB tilgjengelig
RAM. En separat EURUSD-produsent brukte omtrent 10 GiB; ingen av prosjektenes
vakter eller jobber endres. Datasett/lifecycle er fortsatt ikke bygget.

Ny engangsvidereføring ligger i
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_RESOURCE_WAIT_20260928`.
Launcheren avventer minnekravet lest fra eksisterende capped-run-eier, kontrollerer
hvert 15. minutt og starter deretter samme preflight-gjenbrukskommando én gang.
WAITING/RESOURCE_CHECKS/START/TERMINAL og prosess avgjør faktisk status.
Egen launcherlås avviser duplikater; kilde-/filbinding revalideres før start.
POST_BUILD bindes til denne launcheren. Også commit-kontrollen ble avvist av
minnevakten. En engangs runtime-kø binder den eksakte dokumentdiffen og venter
før normal commit/push, kilde-/filbinding og dispatch. `QUEUE_STATUS.json`
og `QUEUE_TERMINAL.json` viser dette fortrinnet. Ingen hook omgås.
Kilden og den bundne dokumentdiffen er frosset også under venting.
Tidligere røde terminaler og deres POST_BUILD bevares. Ingen ny tung jobb er
startet før START/prosess bekrefter det. Codex-oppfølging er aktiv hver 30. minutt;
den krever at Mac og Codex-appen kjører. Den eksterne launcheren venter på serveren.

## Opprinnelig v37-inputbygging — bevares som historikk

Forberedt kjøring: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928`.
`PLAN.json` avgrenser inputbyggingen; `BINDING.json` binder eksakt ren kilde
og inputhasher etter denne dokumentasjonscommiten. `START.json`, prosessene,
`progress.log` og `TERMINAL.json` avgjør om kjøringen faktisk er startet,
aktiv eller ferdig. Ikke start en kopi hvis planen allerede er konsumert.
Kilden fryses mens byggingen går; eventuell oppfølging skrives i runtime-mappen.

Faktiske registry-/squeeze-eiere har godkjent gjenbruk av frosne konstanter og
seks tidsrammers squeeze-parametre. Kalibrering er uendret 2009-06-01 til
2013-01-01 22:00 UTC, med indre fit-slutt 2012-04-11 22:00 UTC. Gamle feature-
matriser gjenbrukes ikke som v37-inputs. Nytt par, M5/M1, rangering og signal
bygges i `GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928`.
Historikkstart er målt 2010-06-14 22:00 UTC; TRAIN/VAL/TEST-grensene er uendret.
Eksisterende capped-run og prosjektlås gjelder, én tung jobb innen CURRENT.

Dette fullfører forutsetninger for en senere vurdert treningstest. Inputbygging
alene dokumenterer ikke beslutningsverdi eller edge. Post-rebuild/readiness,
lifecycle-bindinger og resterende kompleksitetsvurdering må fortsatt fullføres.
`training_enabled=false`; TEST bygges og forsegles uten utfallsevaluering.

## Kontrollert SMC-rettelse som den nye byggingen bruker

Den målte konflikten er rettet hos den delte SMC-eieren: posisjon og observert
intervallbredde er ett par. Positiv bredde beholder rå, uklippet posisjon; kjent
null bredde gir paret `(0, 0)`. Ukjent oppvarming beholder NaN i begge felt.
Lokalflaten får `smc_pivot_envelope_width_atr`; MTF bruker sitt eksisterende
`mtf_smc_range_width_atr`. Ingen pivotregel eller markedsrad endres.

Eksekverte eiere bekrefter signal v37 / SMC-primitiver v4: 242 signalfelt
(25 basis, 150 obligatoriske, 67 kandidater), fortsatt 71 kontinuerlige
kontekstfelt og 190 felt per høyere tidsramme. Ingen modellhoder er fjernet.
189 fokuserte tester og 92 integrasjonstester består. Faktisk M1-kontroll på
6 019 349 rader bekrefter bit-identiske posisjoner for alle 6 019 242 rader
med positiv bredde, 32 ærlige oppvarmingsrader og endelige koordinater etter
historikkgrensen. Alle sju berørte TRAIN-rader har bit-identisk lokal/MTF-
representasjon i sine eksakte pivotnabolag. Toppminnet var 3,12 GiB under
uendret 4 GiB-tak, uten cgroup-OOM. Dette er inputbevis, ikke læring eller edge.

De ferdige par-/M5-/rangerings-/signalartefaktene er **v36-bevis**, ikke ferdige
v37-inputs. De faktiske gamle signal- og M5-kontraktene avvises av v37-eierne.
Ingen manifest omskrives for å passere. De 1 382 ferdige Group-A-chunkene og
alle kvitteringer bevares. På dette historiske stoppunktet manglet M1-output;
v37-output er siden fullført som beskrevet øverst. Gjeldende videreføring
står øverst. Den gamle planen
skal ikke gjenstartes. Ingen rebuild eller trening ble startet i det tidligere
rettings-/oppryddingsarbeidet.

Bakgrunn: fullkjøringen fra `10c78d70` stoppet 02:59:28 UTC på sju TRAIN-rader
med fire like pivotpriser (2012 og 2019). Arrow-rettelsen var vellykket: RSS
9,52 → 6,85 GiB, alle Group-A-chunks fullført. Det første nye diagnoseforsøket
fikk SIGKILL under full MTF-materialisering; det er bevart som ufullført.
Den etterfølgende, avgrensede MTF-kontrollen besto med terminal exit 0.

## Parallelt prosjektarbeid etter brukerens vedtak 28.09

CURRENT har nå egen prosjektlås. Den gamle maskinfelles låsen og EURUSD-
prosjektet er urørt. Én tung jobb innen CURRENT og alle ressurs-/maskinvare-
vakter består. 294 vakttester og en faktisk kjøring mens den gamle låsen
var holdt bekrefter at separate prosjekter kan arbeide parallelt.

Detaljer, avgrensninger og kvitteringer: [repo-gjennomgang](docs/REPO_REVIEW_20260928.md).

## Fullført forskningsresultat — ikke gjeldende startinstruks

Codex har fullført tidlig kalibrering og én forhåndsbundet historisk sammenligning.
[Endelig resultat: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
Kalibrering slutter 2013-01-01 22:00 UTC; ti holdouts dekker juni 2015–juni 2025.
En separat råpriskontroll avdekket og rettet en eldre D1-foldgrensefeil; bare
modellsammenligningen ble beregnet på nytt, med samme artefakter og parametere.
54 tester og den separate kontrollen består. Bruk kun `walkforward_strict/`,
`DECISION_GATE_STRICT.json` og `VERIFICATION.json` under
`/home/andre2/GX1_RUNS/HISTORY2009W_EARLY_DECISION_20260927`.

HGB: +5,71 netto bps per beslutningsblokk (år likt vektet), +1,34 mot LONG,
men nedre grenser -5,12/-22,39 og bare 5/10 årsfordeler mot LONG. Ridge: -2,37.
Begge feilet porten. Ved dette historiske stoppunktet var videre bygging stengt;
brukerens senere vedtak ovenfor åpnet native-forberedelse, fortsatt uten trening.
Ikke gjenstart denne fullførte planen eller den avbrutte senkalibrerte C0.

Avsnittene under er historiske funn. Negativt resultat for målte oppsett beviser
ikke at all retning bare finnes på uker eller at et bestemt marked er ulærbart.

## Konsolidering (operatørvedtak 26.09)

Det fantes to spor: fra 19.09 ble det arbeidet i `/home/andre2/src/GX1_ENGINE` på en foreldet
08.09-base fordi rot-loaderen pekte dit. Nå er dette eneste kodebase; den arkiverte grenen er
tagget `archive/gx1-engine-audit-v9-20260926` og slått inn selektivt. Detaljer, hva som er tatt
og ikke tatt, og hvorfor: [docs/CONSOLIDATION_20260926.md](docs/CONSOLIDATION_20260926.md).
Rot-loaderen importerer nå denne kodebasens `CLAUDE.md`, som importerer `GX1_RULES.md` og
`AGENTS.md` — samme regler for Claude og Codex. Én agent om gangen.

## Hva vi vet

- **Tidligere måling av tidsskala (avgrenset evidens):** Trenden er ~1,6 % av bevegelsen
  per 95-minutters vindu; bekreftede M1-svingninger fortsetter med 49–51 %; retningsmålet hadde
  et hardt tak på 8 timer. På ukeshorisont tjente alltid-LONG etter all kost i 3 av 4 år, og
  ingenting (380 D1/H4-felt, trendregler) slo den — dette beskriver det undersøkte oppsettet og perioden 2021–26. Se
  [docs/DIRECTION_TIMESCALE_20260926.md](docs/DIRECTION_TIMESCALE_20260926.md).
- **Den direkte M1-hypotesen** (25.–26.09) bruker samme etiketter som knee-målet, og vent-målet
  velger side i ettertid (+13,7 bps skjevhet); den kjøres ikke videre.
- **Native lifecycle-v2-modellen:** Entry FLAT256/256 (alle side-Q negative etter kost), Exit
  slår umiddelbar lukking på gjenbrukt TRAIN, samlet læringsport ikke bestått
  ([docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md](docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md),
  [docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md](docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md)).
- **Retningsforskningen 23.–24.09** (walk-forward, mønstre, kryss-aktiva, HGB, direkte
  klassifikasjon) er slått inn under `docs/`, med Codex' forbehold 24.09; ingen robust
  retning på 95 min–8 t ble funnet.
- **Ukesretning, forhåndsregistrert (26.09): NO-GO.** 0 av 48 celler slår alltid-LONG i noe år
  2023–26; modellene kollapser til «vær long» eller taper når de avviker. 2021–26 er ett
  oksemarked — lengre historikk med fallende markeder er den avgjørende forutsetningen;
  operatøren vedtok 26.09 å hente fra 2005; native M5 + M1 fra 2006-03-19 er hentet 27.09
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md))
  ([docs/WEEKLY_DIRECTION_RESULT_20260926.md](docs/WEEKLY_DIRECTION_RESULT_20260926.md)).

- **Gjennomgang 27.09:** featureflaten og grunnmuren er bygd for scalping; flere felt er epokeklokker
  på 2011–2025, og med dagens finansiering tjener alltid-LONG ~0 netto 2011–25 (drift 10,43 mot
  finansiering 10,35 bps/uke). Anbefalt: modellfri TSMOM-måling på D1 før seq513-rebuild
  ([docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md](docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md)).

- **Modellfrie grunnlinjer 27.09: NO-GO** (0/62): ingen enkel scalp- eller swingregel gir retningsgevinst
  etter kost på 2011–2025 ([docs/MODEL_FREE_BASELINES_RESULT_20260927.md](docs/MODEL_FREE_BASELINES_RESULT_20260927.md)).

- **Makrohendelser 27.09: NO-GO** (0/18): FOMC-, NFP- og KPI-tidspunkt gir ingen handelbar retning
  etter kost ([docs/MACRO_EVENT_BASELINES_RESULT_20260927.md](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md)).

- **Intradag-mekanismer, bølge 1, 27.09: NO-GO** (0/61; operatørvedtak «kjør bølge 1 nå»): rundtall ($10/$50),
  de 36 oppsettene som speilede par over 2011–25, COMEX-momentum, LBMA-auksjonen og sesjoner/ORB på lokal klokke
  (sommertid). Ingen celle positiv netto ved policyens low-slippage (1 bps per utførelse); brutto ≈ −spread.
  Etterpåanalyse: fortsettelse etter brudd har +1–2,5 bps mid (beste $50-kryss, 1 t: +1,91, t 3,80, 11/13 år), mindre
  enn rundtur-spreaden på 2,3–2,7 bps ([docs/INTRADAY_MECHANISMS_RESULT_20260927.md](docs/INTRADAY_MECHANISMS_RESULT_20260927.md)).
  Walk-forward-eieren leser nå policyens navngitte slippage-scenarier (`load_slippage_scenarios`); primitiveieren
  har kolonnefilter (`build(..., keep_columns=)`). Neste: operatørvalg.

- **Rebuild v36 på 2009-tapene (operatørvedtak 27.09 «Ja»):** squeeze → C0 → par → squeeze(par) → seq513-kjeden,
  launcher `GX1_RUNS/HISTORY2009_REBUILD_20260927/`. Første forsøk stoppet i C0 på stillestående helgekvoter
  (2011); rettet med stengningskontrakt v2 og nytt parvedtak `OANDA_PAIR_PRETEST_2009_WEEKCLOSED_20260927`
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md)). Det daværende neste steget var v2-taper og tidlig historisk kontroll;
  disse er siden fullført. Dette er ikke en gjeldende startinstruks.

## Operativ tilstand

Gjeldende operativ status står øverst og i `NEXT_RUN_POLICY.json` /
`RUNNING_NATIVE_CALIBRATION.json`, med terminalkvitteringer fra samme navngitte run.
Gamle V9-/lifecycle-v2-checkpoints tilhører tidligere skjema og skal bevares som
historikk. De er ikke gyldige v37-modeller. TEST-resultater, optimizersteg,
full epoch/VAL og handel er stengt. Den nye v37-planens runtime-kvitteringer
avgjør nåværende byggestatus; gamle terminaler er historikk.
