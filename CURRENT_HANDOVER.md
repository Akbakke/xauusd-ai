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
-020s foreløpige registry med1130verdier+1130masker er IKKE aktivert:
  det bygger på gammel kildeadmission. Endelig antall må måles etter024.
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
publisert med eksakt uint64antall/volumparitet før minuttsensur. Alle995nye
kandidater,199perTF, er deklarert i publiserte024tabeller; endelig TRAINstøtte
og felttall bestemmes i025. Alle210gamle registrydeklarasjoner er rebindet til
samme reparerte kilder. Full rå semantisk oversikt er i024/AUDIT.
025er en ny engangsplan for TRAIN-only registry, med faktisk024terminal og
alle tabellhashene bundet. Hele109507TRAINtime+5min brukes, ingen labels/VAL/TEST.

558unike fokuserte tester er nå bestått etter målrettet feilretting. Alle
opprinnelige feilrapporter består; siste resultat per case er sammenstilt i024s
FOCUSED_SOURCE_QUALIFICATION_FINAL.json. JSON-forfall konverteres korrekt ved
rollkontroll. Énepokerutens checkpointpolicy/session/progress er eksplisitt bundet
til faktisk M5-populasjon, mens gamle ruter beholder sine kontrakter. M5-
normalisering bruker eksisterende diskbaserte loader med300s-klokke; eksakt
in-memory-paritet, kildehash, fremtidsisolasjon og publiseringsvern er testet.
Dette er kildekontroller, ingen fysisk modell- eller alpha-kvalifisering.
024er konsumert.025er claimet for én capped TRAIN-only registryfit; trening er lukket.

## Neste steg

1. Bind og kjør den claimede025registryfitten etter ren commit, audit4G/512M.
   Ingen råpass eller trening.024relanseres ikke.
2. Bygg endelig TRAIN-only registry på hele
   beslutningsklokken, uten labels/VAL/TEST. Bruk024s reparerte retainedbindinger.
   Ingen hardkodet1130grense. Ny fysisk TRAIN må matche klokke, ellers refit.
3. Aktiver endelig registry, bygg ny003M5/M1/TF-cache og M5-path-datasett.
  022M5draft er uclaimet og execution_ready=false til faktiske kilder er bundet.
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
