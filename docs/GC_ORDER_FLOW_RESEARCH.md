# Avgrenset GC-order-flow-forskning

Operatørbestilling 06.10.2026: gjennomfør den avgrensede tilnærmingen fra
agentutkastet: kvalitetssjekk ekte GC-data, sammenlign baseline / GC-pris /
ekte flow, og vurder senere LOCATION × FLOW × STATE og kontrollerte ablasjoner.
Forsknings-ID: GC_ORDER_FLOW_RESEARCH_001.

## Aktivt mål og fire ferdigkriterier

Oppfølgende bestilling: «Flott, fullfør dette målet med å legge inn disse 4
punktene. Vi skal gjøre dette grundig». Alle fire arbeidspakker er lagt til i
den eksisterende protokollen, med eksplisitte avhengigheter og ferdigkriterier.
Aktiv status eies bare av NEXT_RUN_POLICY/current_work.gc_goal_progress.

| Trinn | Nødvendig leveranse før trinnet er fullført |
| --- | --- |
| 1. Kildekvalifisering | Ekte, lisensavklarte bytes/kvitteringer; datert identitetsmapping; aggressor-/BBO-/klokkesemantikk; målt dekning/kvalitetsbeslutning og OANDA-overlap. |
| 2. Kausale features | Frosne felt/formler/enheter; tilgjengelighet og rollover; prefix-/framtidsmutasjonskontroller; paritet målt på genuine kvalifiserte rader. |
| 3. Matched A/B/C | Frosset recipe/populasjon før fit; faktiske identiske rader/kostnader; kronologisk kontroll; parvis C−B/B−A med støtte, usikkerhet og kostnadsfølsomhet. |
| 4. Evidensbasert viderevalg | Resultat-/LFS-review og eksplisitt begrunnet valg av eksisterende flow-eier, foreslått niende spesialist eller ingen utvidelse; ingen automatisk native innføring. |

Målet er ikke fullført bare fordi planen eller auditen finnes. Ukjent tilgang,
ekstra edge eller dårlig støtte kan ikke fylles med syntetiske data eller
gunstige antakelser. En negativ eller inkonklusiv empirisk konklusjon skal
rapporteres ærlig; den er ingen grunn til å installere en niende familie.
Nødvendige videre tester bindes før gjennomføring og holdes avgrenset.

## Gjennomførbar grense nå

Kildekontroll er implementert som `audit-gc-source` i den eksisterende eieren
`gx1.scripts.research_ta_campaign_v1`. Det nye, avgrensede kontrakteierskapet er
`gx1/contracts/gc_order_flow_source_v1.py`. Planen ligger i
`configs/research/GC_ORDER_FLOW_RESEARCH_001.json`.

Arbeidskopien var ren på 75f20e07294ffa7fa16d46b33fb6b9e2a15c9e50 ved start.
Handover observerte ingen CURRENT-Python-jobb. Den tidligere oppryddingen er
committet. Kontrakteierne emitterte 254 signalfelt og åtte lærte spesialister;
10 registrerte feature-lag er en annen inndeling. v38 har fortsatt ingen fersk
initial-/læringsmåling. Native benchmarking/trening følger sin egen policy.

Ingen GC/DBN-kilde ble funnet i et avgrenset filnavnsøk i GX1_DATA/GX1_RUNS
uten TEST-stier; dette er ikke bevis for at alle lagringssteder er undersøkt.
Databento-klienten er ikke installert i CURRENTs venv. CSV-auditen bruker
standardbiblioteket og eksisterende research-eier, så ny SDK er ikke nødvendig.
`files=[]` er et eksplisitt manglende input, ikke et tomt godkjent datasett.
CLI-en returnerer exit 2 / BLOCKED_NO_BOUND_GC_FILES og lager ingen falsk
kvalitetsrapport eller outputmappe. Empirisk A/B/C er ennå ikke gjennomført.

Ved målregistreringen 12:46 UTC ble kilde-/tilgangsgrensen kontrollert igjen:
ingen matchende filnavn i det avgrensede DATA/RUNS-søket, ingen deklarert
DATABENTO_API_KEY i CURRENTs .env/Windows-prosessmiljø, ingen ikke-tom nøkkel
i WSL-prosessmiljø, ingen installert Databento-SDK og ingen tilgjengelig
markedsdatakobling i verktøyoversikten. Dette er tilgangsmetadata, ikke
markedsdatakvalifisering, og ikke bevis for alle kontoer eller lagringssteder.
Bare nøkkelnavn/tilstedeværelse ble lest ut. Ingen vendor-API, konto, lisens
eller kjøp ble opprettet. GC-files/brukbar leverandørtilgang er nå den konkrete
eksterne avhengigheten for første trinn.

## 1. Kilde og kvalitet før modellforsøk

Databento GLBX.MDP3 er en kandidat, ikke en valgt/kjøpt datatjeneste. Bestill
først en avgrenset historisk prøve når faktisk lisens, prisestimat, kontrakt,
periode og leveringsformat er kjent. Kjøps-/nedlastingsruten er ikke implementert
av denne bølgen. Ingen full historikk eller MBO følger automatisk.

Providerens dokumentasjon skiller gratis metadata/symbologi fra fakturert
timeseriedata og tilbyr get_cost før datakall. Den krever likevel en API-nøkkel.
En offentlig prøve-/kredittbeskrivelse beviser ikke at denne brukeren har
lisens, kredittramme eller gratis dekning for den ønskede perioden. Bind faktisk
konto-/lisensstatus og estimat, ikke et gjettet prisbeløp. Ingen konto opprettes,
vilkår aksepteres eller betalt/kredittbelastet nedlasting startes automatisk.

Minimum for delta er `trades` med dokumentert aggressor-side. TBBO legger til
BBO rett før hver handel. OFI trenger ordrebokendringene mellom handler;
`mbp-1` gir den relevante hendelsesflaten. BBO-1s/BBO-1m og TBBO alene kan ikke
bevise full ordrebok-OFI. Én MBP-1-fil inneholder også Trade-rader; de samme
handlene skal ikke legges til en gang til fra trades/TBBO.

Hver kildefil må bindes før lesing til provider, dataset, schema, eksakt
outright raw_symbol, instrument_id, publisher_id, start/end i ns, filstørrelse,
SHA-256, kildekvittering/SHA og eksplisitt max_records. Den lokale auditen
støtter vendor-CSV/gzip med pretty_px=false og pretty_ts=false. Fil og
kvittering skal ligge under `/home/andre2/GX1_DATA/research/GC_ORDER_FLOW_RESEARCH_001`,
uten symlinks. Ingen kildeintervall kan gå inn i TEST fra 01.07.2026.
En senere, immutabel kjøreplan binder faktiske files, kildefiler/hasher og en
ny outputidentitet. Dagens tomme kildeplan må ikke presenteres som slik binding.

Auditen rapporterer handlingstyper, ekte aggressorvolum B/A, ukjent N-volum,
tidsrekkefølge, sekvensendringer, største mottaksgap, snapshots, vendorflagg
for dårlig klokke/ordrebok og tilgjengelig/to-sidig/krysset/låst BBO.
Ukjent side blir aldri estimert fra prisretning. Delta er B-volum minus
A-volum for Trade-rader; unknown-volum rapporteres ved siden av.
Fill er passiv side og kan ikke telles som en ekstra aggressorhandel.
Sekvenssprang i instrumentfiltrert data er ikke alene bevis på datatap.
Strukturkontroll beviser verken kontinuerlig dekning, kontraktmapping,
kvitteringssemantikk, økonomisk verdi eller modelladgang.

Kvalifiseringen må i tillegg kontrollere vendor-metadata, periodedekning,
markedsstatus/scheduled closures, ukjent side, feedgaps og nøyaktig kobling
til OANDA. Rapportér n og volumandel og bind eventuell toleranse før utfall
leses. Uavklart dekning gir INKONKLUSIV, ikke nøytral/null flow.

Mottakstid er dataleverandørens capture-tid, ikke botens mottakstid. En
senere featurebygger skal bruke tilgjengelighetsklokke med erklært leveringslag
og gi tidsforsinkelsene samme behandling i alle forsøksarmer. Før 21.05.2017
setter denne leverandøren MDP2 ts_recv lik ts_event; slike rader må behandles
som egen semantikk og kan ikke bevise reell historisk feedforsinkelse.
Start derfor første prøve på MDP3-siden av dette skillet.

Velg outright kontrakter og kalenderperioder uten mål-/PnL-lesing. Rull ved
en forhåndsbundet kalenderregel eller sist fullførte tilgjengelige sesjonsvolum.
Ved volumvalg må begge kontrakters beslutningsgrunnlag bevares. Ikke velg
dagens mest omsatte kontrakt med samme dags framtidige sluttvolum. Ingen
backadjustert GC-pris skal brukes til spot/futures-basis. Reset CVD/profil/
ordrebok ved kontraktskifte; ikke bland kontrakter i samme footprint.

## 2. Kausale features og synkronisering mot OANDA

Første kandidater er signed delta/totalvolum, kjent aggressorvolumandel,
kort kausal CVD-endring og standard top-of-book OFI skalert med deklarert
book depth. GC-pris/basis bygges separat slik at ekstra prisinformasjon kan
kontrolleres i arm B. Feltnavn, ordning, enheter, dtype, aggregasjonsvindu og
normaliseringseier bindes i featurekontrakten før resultater leses.

OFI er pris-/størrelsesendringer i bid/ask, ikke handelsdelta. BBO før/etter
hendelsen og vendorens event-end/snapshot-flagg må tolkes eksplisitt.
Ingen observasjon uten kjent, komplett ordrebokprefiks gir oppfunnet null-OFI.
Ukjent aggressor-side forblir ukjent, med rapportert volumandel.
Book-/CVD-state resettes ved databrudd, session-/kontraktskifte etter bundne
regler; warmup eies av formelen. Tilgjengelighetsklokken er ikke ts_event alene.

Test mekanikk for prefix-invarians, framtidsmutasjon, lik mottakstid med
erklært rekkefølge, snapshots, gap, ukjent side, rollover og splitgrenser.
Bevis deretter featureparitet på eksakte genuine GC/OANDA-beslutningsrader.
M5 Entry/M1 Exit og åtte native familier endres ikke av forskningsbyggeren.
En prøve på toy-data alene fullfører ikke dette trinnet.

## 3. Hva A/B/C skal måle

| Arm | Inputs | Spørsmål |
| --- | --- | --- |
| A | Låst snapshot av dagens 254 signalfelt | Hva forklarer eksisterende signalflate? |
| B | A + GC-prisrepresentasjon og futures–spot-basis | Hjelper ekstra gullpris alene? |
| C | B + handelsdelta og hendelsesbasert OFI | Gir ekte flow merverdi utover GC-pris? |

Det primære inkrementelle sammenligningsparet er C−B; B−A er en egen kontroll.
Lås samme rad-IDer, dekning, warmup, M5-beslutningsklokke, OANDA-M1-fills,
utfall, perioder, treningsbudsjett, normalisering og økonomiregler i alle armer.
A må evalueres på de samme GC-dekkede radene som B/C. Rapporter også hvor mye
av den opprinnelige populasjonen disse radene representerer. Dekningsutvalg
kan ikke bestemmes fra framtidige labels, vinnere eller gunstige GC-perioder.

Første probe bruker én enkel ridge-eier fra eksisterende walk-forward-forskning.
Lås konkret ridge-oppsett, tidsfolder, målhorisont, minimumsstøtte, budsjett,
blokkmetode og beslutningsregel i kjøreplanen før utfallslesing/fit.
Ingen bred learner-/indikator-/terskelrunde. Ingen CPU-fit startes før denne
bindingsgrensen og datakvalifiseringen er fullført.

Snapshot-proben tester tilleggssignal; den gjenskaper ikke v38s sekvenser,
MTF-attention eller delte Entry/Exit-funksjon. Resultatet kan derfor ikke
omtales som en trent/native v38-baseline eller som ferdig botøkonomi.
En etterfølgende native A/B/C trenger fersk initialisering, eksisterende
læringsport og samme komplette baselinearkitektur/recipe i alle armer.

### Historisk TRAIN → VAL → TEST og backtest

Brukerens presisering: «trener med ALT og så kjører en val/test» og backtest
for å måle hva som fungerer. ALT betyr alle avtalte genuine featurefamilier
og inputs i den aktuelle armen, ikke alle tidsperioder i treningssettet.
TRAIN tilpasser modellen og nødvendig preprocessing. Senere utviklings-VAL
kontrollerer generalisering og dokumenterte valg. En uberørt TEST må først
åpnes særskilt etter frosset modell/recipe; den kan ikke brukes til tuning.
Juni 2026 er allerede gjenbrukt utviklings-VAL, ikke en uberørt slutt-test.

Backtest bruker bare informasjon som var tilgjengelig ved hver beslutning,
samme OANDA bid/ask-/kostnadskontrakter og alle valgte handler/åpne posisjoner.
Rapporter nettoavkastning, drawdown, handelsstøtte/expectancy, referanser,
regime-/sesjonssprik og usikkerhet; treffprosent eller TRAIN-fit alene er ikke
beslutningsgrunnlag. Separate TRAIN-fit-, senere VAL- og slutt-testresultater.

To bevisnivåer må holdes fra hverandre: først matched snapshot-/ridge-forskning
for inkrementell GC-informasjon, deretter en egen kildebundet native v38-
validering før innføring kan vurderes. Native bevis omfatter alle genuine
familier og samme lærte Entry/Exit-bundle, ikke en separat exit eller perfekt
framtidig exitfasit. Dagens mål åpner ikke native launch eller forseglet TEST.
Ingen modell-/preprocessing-fit skjer på VAL/TEST; manglende GC-historikk blir
ikke konstruert som null-flow, og baseline må måles på de samme GC-dekkede radene.

Bruk kronologisk TRAIN og senere utviklingskontroll; purge hvert måldomene
og all overlappende framtidig utfallsinformasjon. Alle scalere/parametre
tilpasses kun på TRAIN. Juni 2026 er gjenbrukt utviklings-VAL. TEST forblir
forseglet. Datokohorter med for få uavhengige tidsblokker gir INKONKLUSIV.

OANDA er gjennomførbar prisfasit. Observerte LONG/SHORT-utfall bruker
`entry_causal_m1_outcomes_v1` og eksakte M1 bid/ask. Finansiering, kapital,
slippage og kommisjon kommer fra de eksisterende økonomieierne og bundne
vilkår. Et beregningsvindu er en diagnostisk horisont, ikke maksimal holdetid.
Lærerens Q-estimat og hypotetisk optimal exit er ikke observert profitt.

Rapporter parvis C−B/B−A uten valg av gunstige perioder: TRAIN-fit separat,
senere feil/retning, usikkerhet med tidsblokker, sesjons-/regimesprik og
kostnadssensitivitet. Bind begge kontraster som samme testfamilie før fit.
Et positivt punktestimat alene åpner ingen native utvidelse. Økonomitest
må omfatte alle valgte handler og åpne posisjoner, samlet kapital og
forhåndsvalgt samme-risiko alltid-LONG/FLAT-referanse.

## 4. LOCATION × FLOW × STATE etter første signalbevis

LOCATION gjenbruker dagens nivå-/sweep-/retest-/geometrieiere. FLOW gir delta,
OFI og kausale forhold mellom prisrespons og aggressoraktivitet. STATE bruker
dagens trend, momentum, volatilitet og sesjon. Start med primitive
interaksjoner, uten en ny regelmotor.

Test etterpå absorption/exhaustion og failed-breakout. De er målbare hypoteser
om prisrespons og aktivitet, ikke observasjon av skjulte institusjonsordrer.
Profil × flow, acceptance/rejection og POC-migration kommer i neste avgrensede
trinn. POC/VAH/VAL må beregnes fra utført volum så langt i sesjonen eller fra
forrige fullførte sesjon; sluttprofilen kan ikke brukes tidligere samme dag.
Rader/bins, ties og value-area-metode må bindes før måling.
Stacked imbalances kommer senere. ADR/Opening Range er kausal kontekst.

Først ved støttet tilleggseffekt vurderes innføring gjennom dagens lærte
fusjon: eksisterende momentum/flow-eier eller en egen niende spesialist.
Dette valget er foreløpig åpent. Entry/Exit beholder én bundle og én
beslutningsautoritet. Indikatorablasjoner er et separat, senere vedtak;
ingen indikator eller genuine familie fjernes av denne testen.

## Kildegrunnlag og gjenværende leveranse

- [Databento MBP-1](https://databento.com/docs/schemas-and-data-formats/mbp-1)
- [Databento side, tidsstempler og flags](https://databento.com/docs/standards-and-conventions/common-fields-enums-types)
- [CME-feedens historiske semantikk](https://databento.com/docs/knowledge-base/datasets)
- [TradingView footprint-metode](https://www.tradingview.com/support/solutions/43000726164-volume-footprint-charts-a-complete-guide/)
- [Cont/Kukanov/Stoikov: OFI](https://arxiv.org/abs/1011.6402)
- [Databento historisk autentisering, metadata og get_cost](https://databento.com/docs/api-reference-historical/metadata/metadata-get-cost)
- [Databento tilgang og API-nøkkel](https://databento.com/docs/quickstart)

Leverandørbeskrivelsene beviser feltsemantikk, ikke lønnsomhet i GX1.
OFI-studien gjelder aksjer og er en motivasjon, ikke GC-effektbevis.
TradingViews standard-kategorisering bruker intrabar prisretning og er
ikke den ønskede GC-aggressorfasiten.

Fullføres når genuine GC-bytes er tilgjengelige: kildekvittering og mapping,
kausal klokke/rollover/OANDA-overlap, eksakt frosset A/B/C-kjøreplan,
primitiver/featureparitet og parvise resultater med usikkerhet/kostnader.
Det er ikke målt GC-kvalitet, ekstra edge, v38-læring eller lønnsomhet nå.

## Verifisering av denne leveransen

81 fokuserte GC-/research-tester og 44 status-/handover-tester bestod under
eksisterende capped audit (4 GiB minne, 512 MiB swap, CPU 0–7, én numerisk tråd).
Dette er syntetisk mekanikk-/regresjonsbevis, ikke en ekte GC-kvalifisering.
Endrede Python-filer er syntakskontrollert; kildebindinger, policy og
gjeldende handover er kontrollert. Native launch/trening/TEST forblir stengt.

Målregistreringen la til én fokusert protokollkontroll. 78 GC-/handover-tester
bestod etter registreringen av de fire trinnene; de eksisterende 48 research-
testene fra første bølge gjenbrukes fordi audit-/research-koden er byteuendret.
Dette fullfører målplanens mekanikk, ikke noen av de fire empiriske trinnene.
