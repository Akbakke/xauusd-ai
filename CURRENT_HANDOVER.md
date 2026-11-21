# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

## Gjeldende status 21.11.2026

Den bestilte fullprosjektrevisjonen og ekstra checkpoint/resume-sikringen er ferdige
innen dokumentert omfang. Main003 er ADMITTED_ONE_USE, venter på første launch:
CLAIM=0, remaining=1, outputrot finnes ikke. Bare dens egne training/native-porter
er åpnet; alle øvrige porter er stengt. Kilde og policy fryses gjennom kjøringen.
Aktivt nattmål skal nå innrømme og kjøre nøyaktig én full kronologisk TRAIN-epoke,
deretter vurdere resultatet mot frosset TRAIN-prior og separat utviklings-VAL.
Bruker valgte én epoke først; ingen automatisk ekstraepoke eller femepoke-earlystop.
Stabil langkjøring kontrolleres hvert30.minutt; maskinvarevaktene går kontinuerlig.

## Praktisk gjenopptakelse og lagring

Fryst runtimekilde060fe55f.389målrettede testtilfeller består. Modellens numeriske
kjerne/fit/score/restore/schedule er uendret; throughput004 og mekanikk003 er
COMPLETE_CONSUMED/exit0. Mekanikken matcher original modell/AdamW/RNG eksakt.
Et ferskt uavbrutt WINDOW002 utførte2updates; separat WINDOW001 utførte1update.
Deretter ble maskinen kontrollert ledig, fil-/katalog-fsync og filesystemsync utført,
og Windows ble restartet668→669 med ny WSLboot. Alle102kritiske filer og10arkiverte
hjelpere matchet etter boot. Ny engangsCONTINUE001 fra prefix1 utførte1ny update.
Modell, AdamW, RNG, schedule/bindinger og stegrapporter skal være bitidentiske med
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
Readiness: `MAIN003_READINESS001.json`; aktuelle eksakte SHA/referanser i policy.

## Gjeldende datasett, plan og effektivitet

`NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json`, SHA
20c964020752318c507957ec0b71645e419e023defd7275637cc85fd48b2b5fd.
TRAIN2021–2025:354550issues, B64/5540updates, én kronologisk pass.
Utviklings-VAL2026H1:34934issues. B256/equalTF-sluttscoring:1522forwards;
samlet7062forwards.10800s ytre hardgrense, trygg pause før intern10770sgrense.
Eventuell fortsettelse er samme epoke og krever ny engangsplan; ingen ny initialfit.
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
