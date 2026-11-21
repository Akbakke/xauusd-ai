# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

## Gjeldende status 21.11.2026

Main003 er COMPLETE_CONSUMED/exit0: nøyaktig én full kronologisk TRAIN-epoke og
full utviklings-VAL er ferdig.5540updates/7062forwards,1modell,0autograd og0nye
norm-/priorfits. Alle treningsporter er stengt; ingen ekstraepoke er innrømmet.
Varighet7281,263s (~2t1min).23checkpointkvitteringer bevarer cursor0→5540.
Stabil drift ble fulgt opp hvert30.minutt med kontinuerlige maskinvarevakter.
Uavhengig integritetsreview består, og læringsreview er ferdig. Nattmålets ene
epoke og praktiske gjenopptakelse er fullført; ingen videre trening er autorisert.

GPUvaktens5682samples: topp67Ccore/74Cmem/231,6W/7488MiB. Ingen vaktavslag
under main. CPUwatt/temperatur er fortsatt ikke målt. Kilde er uendret og TEST
urørt etter produsentens kilde-/IO-brannmur. Teknisk fullføring er ikke edge.

## Målt læring og neste beslutning

Frossen TRAIN-prior (identisk med INITIAL) → sluttmodell, alle faktiske predictive
celler; kjent-open-diagnostikk teller aldri som prediktiv kvalitet:

| Omfang | Log-loss | Brier |
|---|---:|---:|
| TRAIN, alle10heads | 1,082061→0,980241 | 0,655284→0,591833 |
| Utviklings-VAL, alle10heads | 1,085805→0,993167 | 0,657712→0,599772 |
| TRAIN, primær M5close | 1,091087→1,082956 | 0,661521→0,655787 |
| Utviklings-VAL, primær M5close | 1,086871→1,077720 | 0,658601→0,652040 |

Alle10TRAINheads og8/10VALheads forbedrer begge proper scores. D1open/-close
forverres på VAL. M5close velger bare105DOWN på TRAIN og32på VAL; DOWNrecall
0,031%/0,117%. Dette beviser ikke balansert retningskvalitet eller kalibrering.
Fullhistorikken har423/44genuine predictiveD1open-celler; de er med i poolen,
mens kjent-open er utelatt. Gjentatte høy-TF-targetbars er ikke uavhengige prøver.

Én initial/sluttmåling viser læring utover prior, ingen konvergenskurve eller beste
epoke. Ingen konkret kode-/mål-/optimizerfeil er bevist. Ingen automatisk femserie
eller earlystop/CONTROL-utvalg er innrømmet. Neste avgrensede beslutning bør bruke
lagrede bevis til å undersøke M5DOWN-dekning før en separat kvalifisert fortsettelse;
ingen taps-/terskelendring begrunnes av argmaxtelling alene. Utviklings-VAL er
gjenbrukt, uavhengig generalisering og økonomisk edge er fortsatt ubevist.

`OVERNIGHT_GOAL_COMPLETION001.json` binder fullføring, integritets-/læringsreview,
389gjenbrukte fokustester på uendrede runtimekilder og faktiske ressursutfall.
53kildeeiere er fersk SHA-/syntakskontrollert. Integritetsreview kontrollerer alle
23JSONkvitteringer/5540steg/prefikser. Mainens24PT-filer tar2230690101bytes;
sluttreview gjenbruker produsentens PT-SHA/stateproof og leser bare PT-metadata.
Checkpoint1 fra restartprøven er derimot helhashet før/etter reboot.

### Retningsskjevhet: faktisk opp/ned målt

Ny diagnose av lagrede INITIAL/FINAL-captures og frosset TRAIN-prior:
0modell/forwards/updates, ingen fit/terskel-/klassekvoteendring. Alle54capturefelt,
kilde-/tidsakser og sannhetsarrays er kontrollert. Originale nativeINITIAL- og
FINAL-scorer reproduseres eksakt på begge splitter. AUC-matematikk gjenbrukes fra
eksisterende ren eier; orientering/ties/konstantprior/tom/enklasse er selvtestet.

Betinget retning er P(UP)/(P(DOWN)+P(UP)), skåret bare der framtidig fasit faktisk
er DOWN/UP. Dette er diagnostikk, ikke en handelsregel eller et kjent-livefilter.
M5closeAUC TRAIN0,5201/VAL0,5294. Brier TRAIN0,249962prior→0,250125modell
(forverring); VAL0,249982→0,249567 (svak forbedring). Alle6VALmåneder har
AUC0,5243–0,5379; juni har likevel dårligere Brier enn prior. Omtrent95%av
M5VALs forbedrede treklasse-logloss kommer fra aktivitet/nøytral, bare5%fra side.

På VAL er fasit DOWN/NEUTRAL/UP=10238/14270/10425, modellargmax32/26894/8007.
Gjennomsnittlig sannsynlighetsmasse er28,49%DOWN/40,29%nøytral/31,22%UP;
DOWN har altså masse selv om den svært sjelden vinner. Betinget UP-probabilitet
er52,31% mot faktisk50,45%: UP-skjevhet og svak separasjon, ikke en svært
UP-tung fasit. Ingen klasseindeks-/maskefeil er påvist.

### Høyere TF: tidligere sterkt samlet tall krever innsnevring

Fasit er nesteTF-bars slutt relativt til forrigeTF-slutt. Når prognosen utstedes
inne i målbaren, kan deler av denne nettoendringen allerede være observert.
Dette er tillatt innen dagens kontrakt, men samletAUC er ikke bevis for prognose
av en helt ukjent bar eller ny prisbevegelse fra utstedelsen.

| Close-retning på utviklings-VAL | Alle utstedelser AUC | En hel nominell TF fram AUC | 5min før slutt AUC |
|---|---:|---:|---:|
| M5 | 0,5294 | 0,5300 (n20549) | 0,5300 |
| M15 | 0,7230 | 0,4743 (n6621) | 0,8944 |
| H1 | 0,8013 | 0,4776 (n1557) | 0,8899 |
| H4 | 0,7330 | 0,4718 (n368) | 0,8064 |

TRAIN viser samme tydelige tidsavhengighet. Full-TF-Brier på M15/H1/H4 er også
verre enn frosset prior. Alle faktiske ledetider er bokført, ikke valgt bort;
nominell TF-varighet hentes fra native målkontrakt. Native gap/delvise kilder
beholdes. D1fullTF har bare24retningsutfall på VAL (AUC0,5), for lite til sterk dom.

Det finnes et svakt M5-rangeringssignal og klart bedre høyereTF-score nær
målbarens slutt. Dette isolerer ikke hva modellen har lært eller om delvis
observert tilstand er årsaken. Robust retningsverdi er ikke bevist. Hovedspørsmålet før flere
epoker/tapsendringer er en kausal kjent-pris-/persistensreferanse og eksplisitt
kontroll av ukjent bevegelse fra utstedelsen. Disse er ikke målt av denne analysen.
Ingen automatisk nyfit, målomskriving, kontrollutvalg eller TEST-adgang åpnes.

Bevis: `DIRECTION_BIAS_REVIEW001.json`, `DIRECTION_LEAD_REVIEW001.json` og
`DIRECTION_FINDINGS001.json` i samme kvitteringsrot, eksakte bindinger i policy.
CPUdiagnostikkene tok10,41s/3,50s, maksRSS1113056/633376KiB. Ingen original
modell-/mål-/treningskilde eller gamle kvitteringer er endret.

### Samarbeid mellom tidsrammer og M5

Kildesporingen bekrefter H1/H4/M15/D1-inputs og nivåinformasjon helt fram til
M5-utgangen via familie-/TF-attention og felles representasjon. Nivåregistre
omfatter avstand, berøringer, alder, reaksjoner, brudd og retester. Bare lukkede
TF-vinduer med tilgjengelighet senest ved utstedelsen konsumeres. H1/H4-output-
sannsynligheter mates ikke tilbake som stemmer; det er feature-/representasjons-
samarbeid. Om støtte+H1/H4-vending faktisk forbedrer M5 er ennå ikke isolert målt.

Den bundne main003 bruker `m5_close_only_backbone_v1`: bare close.M5 gir direkte
tapsgradient til fellesnettverket. Alle øvrige9utgangshoder lærer egne parametre
på detached fellesvektorer. Dette er ikke full gjensidig multitasktrening, selv
om alle TF-inputs er tilgjengelige og kan læres brukt gjennom M5-tapet.
H1/H4s egne prognoseskårer må derfor ikke tolkes som målt verdi av H1/H4-kontekst
for M5. En test av hele H1-baren framover svarer heller ikke på det spørsmålet.

«Tydelig læring av delvis observerte tilstander» var sterkere enn målingen:
vi har målt bedre skår nær barens slutt, ingen isolert årsaks-/persistensanalyse.
Eksempel: ved10:55 kan M5-historikken vise55min av10–11-barens bevegelse,
selv om H1-input fortsatt er siste lukkede H1-bar. Sluttkl11 sammenlignes med
sluttkl10; mye av målbevegelsen kan derfor allerede være kjent. Dette er
kausal kontekst innen valgt kontrakt, og er forskjellig fra å bevise helt ny
framtidig bevegelse. Bevar høyereTF-konteksten og mål dens bidrag til M5.

Bevis: `MTF_M5_WIRING_REVIEW001.json`, kilde-/planbundet; ingen modell/data/TEST
lest eller endret i denne kildesporingen. Ingen ny treningsadmisjon.

## Praktisk gjenopptakelse og lagring

Fryst runtimekilde060fe55f.389målrettede testtilfeller består. Modellens numeriske
kjerne/fit/score/restore/schedule er uendret; throughput004 og mekanikk003 er
COMPLETE_CONSUMED/exit0. Mekanikken matcher original modell/AdamW/RNG eksakt.
Et ferskt uavbrutt WINDOW002 utførte2updates; separat WINDOW001 utførte1update.
Deretter ble maskinen kontrollert ledig, fil-/katalog-fsync og filesystemsync utført,
og Windows ble restartet668→669 med ny WSLboot. Alle102kritiske filer og10arkiverte
hjelpere matchet etter boot. Ny engangsCONTINUE001 fra prefix1 utførte1ny update.
Modell, AdamW, RNG, schedule/bindinger og stegrapporter er bitidentiske med
uavbrutt2; eksakt uavhengig bevis er bundet i policy/readiness. Ingen levendeparent.
Main bruker fersk initialmodell, aldri vekter fra kvalifiseringene.

Checkpoints publiseres bare etter fullført optimizersteg, hver256updates (~22PT,
~2,1GB per fullpass), med immutable SHA/state/cursor/komplett stegprefiks-kvitteringer.
Nye CONTINUE-planer bruker siste komplette receipt og egen engangsadmisjon;
konsumert plan relanseres aldri. Tilstand etter siste komplette receipt ved hardkill
forblir UNKNOWN. Trygg tidsgrensepause bevarer prefix; avbrutt sluttscoring må gjøres
på nytt uten flereupdates. Originale checkpoints og terminaler beholdes.

Etter boot manglet aktiv telemetritransport igjen. Samme etablerte private
portproxy38128→127.0.0.1:38127 ble reetablert, signert query bestod og ingen
cert/brannmur/vaktgrense ble endret. Første wrapperforsøk ble avvist før CLAIM;
faktisk CONTINUE utførte deretter eksakt én admisjon. Windowskommandoer kjøres
med stdin=DEVNULL for å unngå at de konsumerer resten av SSH-scriptet.

Bevisrot: `/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001/CRASH_RECOVERY_QUALIFICATION_001`.
Readiness: `MAIN003_READINESS001.json`; fullføring: `OVERNIGHT_GOAL_COMPLETION001.json`.
Aktuelle eksakte SHA/referanser står i policy.

## Gjeldende datasett, plan og effektivitet

`NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json`, SHA
20c964020752318c507957ec0b71645e419e023defd7275637cc85fd48b2b5fd.
TRAIN2021–2025:354550issues, B64/5540updates, én kronologisk pass.
Utviklings-VAL2026H1:34934issues. B256/equalTF-sluttscoring:1522forwards;
samlet7062forwards.10800s ytre hardgrense, trygg pause før intern10770sgrense.
Gjenopptakelsesruten gjelder en avbrutt epoke med ny engangsplan; denne epoken
er ferdig. Ekstraepoker krever separat kvalifikasjon og ny eksakt plan.
Native TEST fra01.07.2026 er forseglet og ikke lest. M1 er utelukket også indirekte.
Alle genuine M5/lukketM15/H1/H4/D1-features, åttefamilier og D1/makro beholdes.
GC/MBP er bevart separat. Ingen live/paper/broker/ordre/spending/promotion.

Målt tidligere par: B64forward/backward uten AdamW1,12137→1,05918s; eval256rader
2,11099→0,53442s. B64bitidentisk og B256 innen forhåndsbundet FP32-toleranse,
eksakte tellinger/confusion/ties. Ingen robust full-AdamW-ETA eller globalt optimum
hevdes. Fullinitial6086forwards gjenbrukes med autentisert kildebro, ingen nyfit.

CPU i9-10900K har10fysiske kjerner/20logiske tråder; WSL eksponerer19vCPUer.
Treneraffinity0–18, numeriskpool8,20GiBram/512MiBswap/128tasks. CPUboost0 beholdes
som vedtatt stabilitetsprofil etter tidligere HyperV-krasj. CPU-packagewatt/temperatur
kan ikke måles via etablert sensorrute; ingen CPU-termisk/wattstop hevdes.
GPU RTX3090 har250Wfysiskcap. Uendrede GPUvakter:85Ccore/80Cmem,300Wcapgrense,
310Wdrawgrense,12288MiBVRAMbruk/12GiBallocator. Ingen driver/boost/powerendring.

## Målt kode-, input- og testdekning

- Fullauditens snapshot av CURRENT:1402Pythonfiler,20251definisjoner,311888AST-kall,19693importsetninger.
  Fullt originalinventar og komplette utskiftninger av sju endrede moduler er
  bevart. Historiske/stored kildekopier er også inventert. Ingen manglende intern
  import-/symbolkandidat ble funnet; installerte direkte avhengigheter og pipcheck
  består. Optional/deferred OpenBB er ikke del av den aktive ruten og er ikke installert.
- `FULL_NATIVE_NUMERIC001`: alle256felt på femTF og lokale ikke-makrofelt,
  1609råfeltposter; alle27arrayheaders/bytebindinger og faktisk normaliserings-
  aritmetikk/domener. Exit0,5078,675s,peakRSS2551732KiB.
- `FULL_MACRO_NUMERIC001`: alle61makroformler i signal/kontekst,122feltposter/
  47528638celler. Samlet1731råfeltposter er uavhengig kontrollert. Offentlige
  kildecontainere ble eksplisitt helhashet også etter forseglingen; bare tidligere
  kvalifiserte observasjoner/tilgjengelighet brukes til makroverdier. Ingen nativeTEST-fil.
- `FROZEN_TRAIN_STATS003`: hele input-TRAIN/vindusunioner før målmasker, kvartiler/
  skala, domener, kategorier og132aliaser består.1729centerbits er eksakte; ett
  navngitt D1-center er numerisk likt +0/-0, dokumentert i `D1_STATS_DIAG001`.
  Alle skala-/float64-ekstrembits er eksakte ved uendret faktisk lagret center.
  Exit0,20,582s,peakRSS748472KiB. Ingen fit-API eller nye parametre.
- Fersk initialbaseline med én modell/6086forwards og kvalifisert kald mekanikk
  gjenbrukes. Ingen endret ONLINE-funksjon krever ny baseline. De tre uavhengige
  input-/statistikkauditene konstruerte ingen modell/optimizer eller dekodet targets.
- Målrettet testkvalifikasjon:544tilfeller,527uendredePASS og alle17rettede feil
  består ved eksakt etterkontroll.170retention-tester består. Obligatoriske
  pre-commit Entry-kontrakter består. Ingen unødvendig fullsuite-repetisjon.
- Fullauditens testsamling:15058tilfeller samlet/importert med0feil. Ingen testkropper
  i denne innsamlingen. Normale modell-/optimizer-/genuine-data-/nettverkskall var
  sperret.18nektede operasjoner var egen kildeintrospeksjon/cachemkdir. Vakter er
  prosesslokale trusted-code hooks, ikke OS-sandbox. Ikke kall dette15058kjørte tester.

## Rettelser og opprydding

Retention stanser nå nye dokumentreferanser også etter planlegging og før staged
sletting. `docs/*.json` inngår i dokumentvernet; sikker gjenoppretting er bevart.
Fire konsumerte GC-dokumenter pensjonerte eksplisitt scratch-lagringskrav med
originale Git-/RUNS-bevis intakte; feilaktig PENDING-prosa er rettet.

Fem frakoblede historiske workers og gammel fysisk V38-designkopi er fjernet
fra CURRENT med Git-gjenoppretting. Ti ekte designlesinger i fem testmoduler bruker
nå syntetiske fixtures i eksisterende eier. Gammel teacher-test fjernet feilaktig
alle familienormer; den er korrekt avgrenset til hovedencoderens norm med streng
Entry/Exit/state/RNG-paritet. Ingen produksjonsmodell eller toleranse er endret.

Eieren fullførte eksakt plan→godkjenning→fersk dry-run→execute:8mål/22filer,
35728523861logiske bytes,35728601088allokerte payloadbytes (+4096katalogbytes).
WSL tilgjengelig plass økte35728494592bytes ved umiddelbar etterkontroll.
Owner `DELETE_COMPLETE` SHA:
`50ecbbf128efcf314a06c52b73784e54b640e07d0171606cf6d97f0a777a3635`.
Beholdte TRAIN/VAL-kopier matcher forventet SHA også etter slettingen; nødvendig
signalfil er lesbar/stabil. Rådata, øvrig GC/MBP og originale checkpoints beholdes.

Ingen root-FD-tabell ble hoppet over. Bare slettetreets prosesser fikk midlertidig
SYS_PTRACE+DAC_READ_SEARCH med alleUID-felt1000 før uendret capped-wrapper.
Samme FD9-lås dev78/inode35,4GiB/512MiB/1tråd/64tasks og alle writer-/byte-/
autoritets-/dokumentvakter bestod. Ingen filcap/kernel-/permissionendring; vanlig
sesjon har effektive/ambient capabilities0. Windows-VHD er ikke komprimert.

## Diskkontroll og ubeviste grenser

Førsteinventar:217026oppføringer/403488735232unike allokerte filbytes over21røtter.
Sluttinventar:217422oppføringer/368725549056bytes,0metadatafeil,ingen symlink-
følging. Nye revisjonsbevis forklarer at nettoreduksjonen er mindre enn slettede
payloadbytes; sluttinventarets egne arkivkopier/sluttkvitteringer ble skrevet etter
snapshot. Alle22manglende DATA/RUNS-filer er eksakt eieravstemt; øvrige6manglende
kilder er de Git-gjenopprettbare pensjoneringene. VHD og WSL skal ikke dobbelttelles.

`PRESEAL_BYTE_AUDIT002` bestod34441fysiske filer/94198463861logiske bytes på255,225s.
81513799539bytes matcher opprinnelig forventet SHA;12684664322bytes har kun fersk
stabil checksum/lesbarhet.0bytefeil/arraydekodere/modeller/fit/nativeTEST-lesinger.
ALLE opprinnelige filstier og nye tilføyelser er metadataført; positive fysiske
aliaser er kvalifisert sammen. UNKNOWN/MIXED/SEALED/øvrige ukvalifiserte payloads
ble ikke åpnet. Regel106/764/812 tillater heller ikke blindhash av nativeTEST.
Dermed er ikke alle403GB eller antatte670GB innholdsgodkjent.

AST/ruting er ikke full atferdsgaranti for alle historiske funksjoner. Full
vendor-/revisjonsledger, per-felt gradient/nytte og blanket numerikk for utsatt GC/
historiske flater er ikke bevist.0,3ATR er nettoendringsproxy, ikke bevist rolig
prisbane/økonomisk FLAT. Juni2026 er gjenbrukt utvikling. TRAIN-fit, senere
generalisering, referanseverdi og realisert økonomi holdes adskilt. Edge/profitt er
fortsatt ubevist. Teknisk beredskap betyr ikke bedre handelsbeslutninger.
