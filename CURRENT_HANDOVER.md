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
  flerårsmatrise. Ny fysisk cache/normalisering/modellkonsum gjenstår.
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
Hele109507TRAINtime+5min er brukt, uten labels/VAL/TESTfit. Nye003TRAINklokken
må fortsatt bevise eksakt likhet før fysisk inputadmission.

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
024og025er konsumert.022er claimet for én ny003M5featurekjøring; trening er lukket.

## Neste steg

1. Bind og kjør claimet022 på ren commit: først ny pair, så M5-enriched/
   253TFcache og model-source. Audit4G/producer20G/10G,512Mswap, ett barn om gangen.
   M5innlasting/GCvedlegg er bounded; eksisterende vakter og checkpoints beholdes.
2. Kontroller022terminal og faktiske dimensjoner/GCkolonner; planlegg ny M1-
   etterfølger og nytt datasett fra faktiske manifester. Ingen gamle checkpoints.
3. Bygg ny003M5/M1/TF-cache og M5-path-datasett med aktivert registry.
  022M5er claimet og execution_ready=true; kun fersk binding/ren kilde kan starte.
   Gjenbruk eksisterende bounded liveness-/rekonstruksjonseiere; kopier ikke
   den gamle011auditens fulle pandasinnlasting av bredM1flate.
4. Normaliser hele observerte fysiske TRAIN én gang. Mål ny initial ONLINE=
   TARGET=EMA, fysisk batch-/serveparitet og Entry-only optimizer/resume.
   Maskinvid idle og trygg fysisk Windowsreboot må bevises før GPUadmission.
5. Bind faktisk unik M5-populasjon/N/ordre og ceil(N/16), behold siste delbatch,
   tren nøyaktig én epoke med eksisterende varige checkpoints. Stopp før VAL/
   lærerrefresh/ny epoke. TRAIN256/CONTROL256 er måleprober, ikke treningsbudsjett.

## Ærlige grenser

Ingen ny fysisk nativeinputkjede, normalisering, Transformer-læring,
generalisering eller nettoøkonomi er fullført. MBP-1 viser beste bid/ask,
ikke dypere bok eller individuell ordreidentitet. Full prisgrid beholdes;
modellen konsumerer deklarerte avledede felt, ikke alle råbane-/prisbininteraksjoner.
Makroens fullstendige historiske originalvintager er ikke bevist.
Ingen Pythonfil er bevist slettbar etter referanse/import/CLI/test/eierkontroll.
DATA/RUNS ryddes kun av retention-eieren. Tidligere generelle negative
prosjektslutninger brukes ikke; faktiske originalmålinger bevares.

Detaljert inventar/formel-/TF-eierskap og avgrensninger: `docs/FEATURE_AUDIT.md`.
Eksakte terminaler, plantilstand og neste arbeid: `NEXT_RUN_POLICY.json`.
