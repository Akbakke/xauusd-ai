# GC M5 / MTF feature recipe, 17.10.2026

Brukervedtak: «Vi holder oss til M5 intradag (med mtf så klart).» Dette
konkretiserer bestillingen om GC-datasett, volum/order blocks og ny trening.
Plan GC_M5_FEATURE_BUILD_20261017_005 bygger bare featureverdier på de
innrømmede maskene fra plan 004. Den relanserer ingen gammel plan og åpner
ikke fit, normalisering, TEST, økonomi eller native serving.

## Eiere og klokker

`gx1/features/gc_order_flow_v1.py` er ny, avgrenset eier for observerte GC
Trade-laddere/profiler; slik eier fantes ikke. `gc_continuous_source_v1`
fortsetter alene å validere rå records, T/F, identitet, klokke, kvalitetsflagg
og heltallsvolum. Den nye eieren får en valgfri callback etter validering;
den gamle parserens resultater endres ikke uten callback. Alle utvalgte
minutttellere må være identiske med den fullførte 004-maskfilen.

M5-beslutning = native radens start + 300 sekunder, strengt før 01.07.2026.
GC grupperes på ts_recv; et M1-vindu er tilgjengelig når minuttet slutter
(004-klokken). Dette er capture-tid, ingen bevist faktisk bot-leveringstid.
M5/M15/H1/H4/D1 bruker `htf_features.multi_tf_bar_label` og eksisterende
origin/fase (H4/D1 fra kl. 22 UTC); bare siste helt lukkede vindu kobles til
native M5. Ugyldig siste vindu erstattes aldri med et eldre gyldig vindu.
Eksisterende native 315-felts input, MTF-sekvenser, kontekst, familier og M1
Exit beholdes som refererte inputs; denne sidefilen registrerer ikke nye
native modellfelt eller gamle D1-labels som intradagmål.

## Masker, enheter og formler

Hvert TF-vindu krever alle sine M1-minutter, positiv innrømmet observert
T-aktivitet i hvert minutt og samme valgte outright-kontrakt. Ingen hullfylling.
Volum/profil bruker 004-volummasken; footprint/delta/CVD krever dessuten
004-signert maske i alle minutter. Kjent-sideandel og ukjent kontraktsvolum
rapporteres også i gyldige usignerte vinduer. Manglende/ugyldige verdier er
Parquet NULL, aldri nøytral flow. Separate volume_valid/signed_valid følger
hver TF. Rene GC T-laddere er ikke komplett historisk børsvolum.

Priser er rå, ujusterte outright-priser i USD (rå tick × min_price_increment /
1e9). Mengder er observerte kontrakter, trade count er antall T-records.
Rå og aggregerte laddere summeres i uint64. Publiserte modellkandidater er
float64 med NULL; byggingen avviser mengder over 2^53 før konvertering.
Ingen normalisering tilpasses i featurebyggeren.

| Feltgruppe per TF | Eksakt betydning |
| --- | --- |
| open/high/low/close_usd | Første/maks/min/siste innrømmede T-pris i receive-rekkefølge. |
| volume/trades/unknown | Sum observerte kontrakter / T-records / ukjent aggressorvolum. |
| known_aggressor_fraction | (B-volum + A-volum) / totalvolum; ukjent side estimeres ikke. |
| vwap_usd | Sum tickpris × kontrakter / kontrakter, omregnet til USD. |
| poc_usd | Tick med størst observert volum; likhet velger laveste pris. |
| q25/median/q75_usd | Laveste tick der kumulativt volum når henholdsvis 1/4, 1/2, 3/4. Dette er kvartiler, ingen påstand om børsens value area. |
| price_levels | Antall observerte ticks med positivt volum i vinduet. |
| buy/sell/delta_contracts | B-aggressorvolum / A-aggressorvolum / B minus A; bare fullt kjent aggressor-side. |
| max/min_level_delta_contracts | Største/minste B minus A på samme observerte tick. |
| buy/sell_poc_usd | POC separat for B/A; NULL når den siden ikke har volum. |
| cvd_segment_* | Delta, totalvolum og minutter siden siste UTC-datoskifte, rull eller ugyldige/silent M1. Kort relativ CVD; ingen komplett sesjonspåstand. |
| spot_basis_bps | 1e4 × (GC-close / OANDA-mid-close − 1), samme siste lukkede M1-minutt; bid/ask fra eksisterende hash-bundne OANDA-kilde. Ingen backadjustering. |

Profil/footprint er statiske fordelinger i hvert helt lukket TF-vindu, uten
kumulativ profil-state over vinduer. CVD-state resettes separat som i 004.
Høyere TF kan derfor ha svært lav/ingen gyldig flowdekning; rapporten skal
vise dette på native M5-radene før noe velges til trening.

## Order blocks og sweeps

`smc_v1.replay_smc_causal_structure_v2` er fortsatt eneste pivot-/BOS-/sweep-
eier. Ny `replay_volume_order_block_proxies_v1` utvider den eieren for
sammenhengende gyldige GC-prisvinduer, med reset ved brudd/rull:

- BOS opp fryser full low/high-sone fra siste strengt tidligere bearish
  candle; BOS ned bruker siste tidligere bullish candle. Sonens opprinnelige
  observerte T-volum følger med, uten vilkårlig volumterskel.
- Sonen oppstår først på BOS-baren, aldri på den historiske opprinnelsesbaren.
  Bull-sone ugyldiggjøres ved close under low, bear-sone ved close over high.
- Første senere wick som overlapper sonen gir én first_retouch. Aktivitet,
  opprettelse, alder og grenser er observasjoner, ingen handel/exitregel.
- BOS/sweep/bull/bear-masker krever eierens bekreftede relevante pivot, med
  eksisterende SWING_LOOKBACK. Full swing-state-tilgjengelighet rapporteres
  separat; fire-pivot-klassifisering er ikke nødvendig for én bekreftet BOS.

Dette er kausale prisproksier med volumstøtte. Institusjonelle ordre og sweeps
på flere boknivåer er fortsatt REJECT. Event-book OFI er ikke innrømmet av
Trade-maskene og bygges ikke uten egen book-state-kvalifisering.

## Verifikasjon og grenser

Frosne første komplette usignerte/signerte M5-vinduer per kontrakt/måned
velges fra masker før prislesing. Deres faktiske rå T-records summeres med en
uavhengig Python-dict-reduksjon mot OHLC/profil/footprint fra Arrow-ladderen.
Alle utvalgte M1-tellere avstemmes eksakt mot 004; alle fil-recordtall mot 002.
Reelle sammenhengende segmenter kontrolleres med prefix og framtidsmutasjon;
CVD-prefix, canonical MTF-label, faktisk native rad-ID og Parquet readback
kontrolleres. Native train/serve-paritet og prediktiv verdi er fortsatt ubevist.

Planen publiserer en kompakt sidefil med native split/radnummer og M5/MTF-
features, aggregerte laddere og bevis. Den kopierer ikke det store native
sekvensdatasettet. Videre trening krever en separat konkret intradag-mål-/
recipebinding; tidligere 20-handelsdagersresultater brukes ikke som M5-bevis.

## Teknisk gjenopptakelse 006

005 fullførte råpassering, avstemming og featureberegning, men feilet før
publisering: NumPy-int64 i prefix-rapporten kunne ikke skrives som JSON.
Ingen ferdig featurefil ble publisert. Originalfeil/logg/partiell rapport
beholdes. 006 gjenbruker de 15 ferdige sparse ladder-filene (hash-bundet,
55,9 MB samlet). M1 OHLC-receive-rekkefølge var bare i minnet og gjenoppbygges
fra samme kildemottak med eksisterende parser; ingen ny ladderkopi lages.
Alle minutttellere/rå recordtall, gjeldende og gjenbrukte hasher, profil- og
råreferanseparitet kontrolleres på nytt. Kun rapportheltall og gjenbruk er
rettet; formler/masker/kohorter er uendret. 005 relanseres aldri. En reell
TF-prefix med bare én bar har ingen framtidig suffix og merkes eksplisitt.
