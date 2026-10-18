# CURRENT HANDOVER — XAUUSD M5 og full MBP-1-støtte

Autoritativt arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md`; aktuell arbeidsstatus og hashbindinger er i `NEXT_RUN_POLICY.json`.
Flere agenter kan ha adskilte oppgaver; én integrator og én capped tung jobb.
Ingen kilde-/policy-/dokumentendringer under kjøring.

## Mål og arbeidsrekkefølge

Autonom XAUUSD-bot er prosjektmålet. Arbeidet nå er offline data-, indikator-
og modellkvalifisering. M5 Entry bruker lukket M15/H1/H4/D1 som kontekst.
M30 vurderes bare gjennom målt tilført verdi. Åtte spesialistfamilier og
niende D1/makrokontekst beholdes. Hver genuine indikator skal følges fra
formel og kildeklokke til alle relevante TF og faktisk modellkonsum.

Operatørens eksplisitte18.10-svar: M5-prisforløp og liten MAE før MFE er
hovedmål; 1/5/20d er støtte. Beregningshorisont er ingen maksimal holdetid.
Entry lærer uten Exit-targetavhengighet; inaktive gradienter/vekter og EMA
skal kontrolleres. Hindsight-MFE er supervisjon, ikke realiserbar exitgevinst.

Bruk eksisterende26MBP-måneder før utvidelse til hele OHLC-historikken.
TRAIN er [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
retrospektiv utviklingskontroll [2026-04-01,2026-07-01). Alle26måneder er
gjenbrukt utvikling. TEST fra juli er forseglet. Én full native epoke er
allerede bestilt; datakjede, normalisering og fersk modellkvalifisering må
bestå først. Ingen broker/live/paper/ordre/spending/promotion.

## Gjort og målt

- Rå001 er publisert og konsumert:66filer,892100712records og9173893
  M1-prisnivårader med eksakt buy/sell/unknown-konservering.15ufiltrerte
  LADDER- og15M1-reduksjoner er bevart med publiserte hashbindinger.
-019 er COMPLETE,953.109s, uendret kilde og TEST=false. Ett eventpass
  tilførte spread-/event-/klokkestatistikk; opprinnelige handels-/bokreduksjoner
  og prisnivårader ble kontrollert eksakt.019s eventstatistikk ble maskert før
  publisering og kan ikke alene gjenopprette senere utvidet kildeadmission.
-020 er COMPLETE,23.033s, uendret kilde og TEST=false. H4/D1-SMC ble rettet
  fra M1-gapreset til observerte samme-kontrakt-candles. Hele manglende
  TF-candles sensurerer OB-state. Alle935kandidater hadde TRAINobservasjoner,
  og232ikke-proxykolonner perTF var eksakt uendret.160fokuserte tester bestod.
-020s foreløpige1130registry ble aldri aktivert og er erstattet av025.
-253genuine primitive TF-felt er implementert perTF, inklusive ekte BID/ASK.
  Entry får også M5.217sammeformelaliaser har eksplisitt paritetskontroll.
  Bred GC-kontekst kobles på i eksisterende skrivechunks, uten full bred
  flerårsmatrise. 022 har nå publisert fersk253TFcache og M5/model-source; M1, fysisk
  normalisering og aktuelt modellkonsum gjenstår.
- Tidligere fysisk016/017 med109507TRAIN/15993VAL var gammel735/552-generasjon.
  Disse kvitteringene er historiske, ingen adgang for dagens modellfunksjon.

## Konkret kildeblokkering og rettelse

Signaturidentitet var feilaktig brukt som avvisningsgrunn for hele minutter.
134296Tposter er identiske over20felt, uten signaturkollisjoner. Dette alene
avviste78322ellers gyldige M1 med25064729observerte handelskontrakter.
Databento bekrefter at identiske Tposter kan være korrekte. Ingen T dedupliseres
på dette grunnlaget; alle andre flagg-/pris-/kontraktvakter beholdes.

Den bundne leverandørhendelsen17.09.2025 13:34:28Z–13:50:40Z sensurerer alle
17overlappende M1-bøtter. Bok/event-state sensureres også gjennom første
friske etterfølgende anker. Ingen ukjent timestamp-offset korrigeres ved gjetning.
Historiske non-T-actionlabels er ikke kvalifisert som kansellerings-/endringsflow.
BBO-state, T-volum og observerte ordreantall har egne dokumenterte domener.
Kilder og eksplisitt behandling er i
`configs/research/GC_VENDOR_SOURCE_INCIDENTS_20261018_001.json`.

024er COMPLETE/konsumert etter1202.084s, exit0, uendret kilde og TEST=false.
Alle66råfiler/892100712records ble behandlet. Samme30hashbundne råreduksjoner
ble brukt; gamle handels-/bokreduksjoner, alle9173893originale prisnivåside-rader
og019eventstats på opprinnelig gyldig populasjon er eksakt kontrollert.
Ingen T deduplisert. PTPsensur ga17overlappende volumbøtter og18bok/eventbøtter.
Netto etter sensur:78305M1 og25059620kontrakter gjenopprettet. Strict komplett
volum/profil M5/M15/H1/H4/D1 er nå151773/50074/12145/2445/0bøtter.

Nye T-sidetall/sidemiddel og total størrelsesstandardavvik/min/maks er fysisk
publisert med eksakt uint64antall/volumparitet før minuttsensur.025er COMPLETE/
konsumert,29.254s, uendret kilde og TEST=false. Alle995nye felt,199perTF, og
alle210gamle felt har genuine TRAINobservasjoner. Endelig registry har1205
verdier+1205observasjonsmasker,30observerte konstanter,0uobserverte felt og
0formelaliaser. Alle gamle kilder er rebindet til samme reparerte024tabeller.
Hele109507TRAINtime+5min er brukt, uten labels/VAL/TESTfit. Den nye M5populasjonen
skal måles fysisk og få fersk wholeTRAINregistrytilpasning før inputadmission;
025s109507klokke gjenbrukes ikke som bevis for endret emisjon.

Registryen er aktivert i eksisterende nativekilde med SHA d2210dae945c9a43dfdf2aff15f4362563b566ba3c7ebaf3f29d9817201dd3ca.
Målt kanonisk import gir2725signal-/2542ctx-cont-felt; dette er ikke en fysisk
modelforward. 75unike forbruker-/maske-/familie-/batching-/normaliseringskontroller er bestått.
Ny fysisk003-generasjon og ekte aktuell modelforward/paritet gjenstår.
Full rå semantisk oversikt er i024/AUDIT.

558unike fokuserte tester er nå bestått etter målrettet feilretting. Alle
opprinnelige feilrapporter består; siste resultat per case er sammenstilt i024s
FOCUSED_SOURCE_QUALIFICATION_FINAL.json. JSON-forfall konverteres korrekt ved
rollkontroll. Énepokerutens checkpointpolicy/session/progress er eksplisitt bundet
til faktisk M5-populasjon, mens gamle ruter beholder sine kontrakter. M5-
normalisering bruker eksisterende diskbaserte loader med300s-klokke; eksakt
in-memory-paritet, kildehash, fremtidsisolasjon og publiseringsvern er testet.
Dette er kildekontroller, ingen fysisk modell- eller alpha-kvalifisering.
Ny kildefeil er også målt og rettet: normaliseringsstatistikk for binære GC-
verdier med egen mask fikk asinh(1) i stedet for1 som transformert maksimum.
Ekstremer bevarer nå samme binær/kategori-identitet som faktisk runtime; numeriske
verdier beholder asinh og manglende verdier hard0. Før-repro og originalfeil er
bevart.75unike forbrukercaser er bestått;002etterkontrollen bestod62/62.
Bevis:022/QUALIFICATION/FINAL_CONSUMER_QUALIFICATION.json. Ingen guard svekket.
024/025/022 er konsumert.022 avsluttet13:08:50UTC etter2554.675s, exit0,
uendret kilde og TEST=false. Ny003M5 har1140982rader og2576kolonner;
model-source bekrefter2725signal/2542ctx-cont/1ctx-cat og samme featurekontrakt.
Terminal og seks faktiske artefakthasher er bundet i policy. Trening er lukket.

Konkret neste blokkering: både sekvensrekonstruksjonsauditen og trainerens
sekvenskildeleser allokerer hele den historiske M5signalmatrisen i RAM. Ved
1140982×2725float32 er signal alene omtrent11.6GiB. Minste rettelse gjenbruker
eksisterende diskbaserte M5loader og beholder hele populasjonen/verdikontrollen.
Kilden var fryst til022terminal. Rettelsen bestod39fokuserte caser med
uendret kilde; alle verdier/klokker og disklevetid er kontrollert.

Tidlig klokkesammenligning viste118706M5kildetider mot109507gamle fysiske
TRAINrader; alle gamle finnes i kilden. Kildeklokke er ikke emisjonsklokke.
Undersøkelsen avdekket en konkret målfeil: fysisk emisjon krevde fremdeles
legacyM1labels og alle37auxmål, inkludert inaktive96bars-hoder. M5kildens
95minprimary ble dermed indirekte filtrert av8timersauxkompletthet. Fire
aktive M5forecasts K1/5/12/24 hadde dessuten ubetingettrue-masker.

Ny eksplisitt m5_primary_observed-emisjon bruker ekteM5valid, split og
inputwarmup. Legacydefault bevares. Hver av fireM5forecasts og seks1/5/20d
støtteceller har egen observasjonsmaske; uobserverte lagres0 uten falsk
observasjon. Ny fase-/målbinding må bevise faktiske parquetmasker og
forby legacy/jointbruk av nyeM5inputs. 496unike fokuserte caser er bestått etter tre små fixture-rettelser
(str-sti/ns-enhet/import); produksjonskilden var uendret gjennom etterkontrollene.
Originale feilrapporter er bevart. Fysisk nyeM5data/normalisering/modell gjenstår.

## Neste steg

1.496unike sourcecaser er bestått, inkludert faktisk Arrow→Dataset→batch og
   fysisk-maskebinding på små Parquet-fixtures. Gjeldende shared-featurekontrakt
   er eksakt lik022s publiserte.026 er claimet for én ren kildebundet M1jobb.
2. Kjør én fersk026M1-enriched på003, gjenbruk publisert022253TFcache.
   Producer20G/512Mswap,4096skrivechunks, ny checkpointsti. Ingen009reuse.
   Faktisk022terminal og alle seks artefakthasher er fylt inn i026planen;
  026 er claimet; binding skrives først fra ren commit. Ingen kjøring før
   denne bindingen.026 er dataarbeid, ingen trening.
3. Bind faktisk022/026 og bygg nytt26mM5-pathdatasett fysisk. Wrapperens
   eksplisitte --physical-build-only stopper før inputadmission. Mål faktisk
   nyTRAINklokke, tilpass ferskTRAINregistry og publiser gjeldende immutable
   signal-/datasetbindinger før liveness/pretrain. Gamle025hash/klokke kan
   ikke utgis som nye. Gjenbruk featurebytes bare etter eksakt dokumentert
   verdifunksjon/kilde/maske/rolle-ekvivalens; aldri011wideM1read.
4. Normaliser hele observerte fysiske TRAIN én gang. Mål ny initial ONLINE=
   TARGET=EMA, fysisk batchparitet og Entry-only optimizer/resume.
   Maskinvid idle og trygg fysisk Windowsreboot må bevises før GPUadmission.
5. Bind faktisk unik M5-populasjon/N/ordre og ceil(N/16), behold siste delbatch,
   tren nøyaktig én epoke med eksisterende varige checkpoints. Stopp før VAL/
   lærerrefresh/ny epoke. TRAIN256/CONTROL256 er måleprober, ikke treningsbudsjett.

## Ærlige grenser

M5/cache-delen er fysisk publisert. Hele inputkjeden, normalisering, Transformer-læring,
generalisering eller nettoøkonomi er fullført. MBP-1 viser beste bid/ask,
ikke dypere bok eller individuell ordreidentitet. Full prisgrid beholdes;
modellen konsumerer deklarerte avledede felt, ikke alle råbane-/prisbininteraksjoner.
Makroens fullstendige historiske originalvintager er ikke bevist.
Ingen Pythonfil er bevist slettbar etter referanse/import/CLI/test/eierkontroll.
DATA/RUNS ryddes kun av retention-eieren. Tidligere generelle negative
prosjektslutninger brukes ikke; faktiske originalmålinger bevares.

Detaljert inventar/formel-/TF-eierskap og avgrensninger: `docs/FEATURE_AUDIT.md`.
Eksakte terminaler, plantilstand og neste arbeid: `NEXT_RUN_POLICY.json`.
