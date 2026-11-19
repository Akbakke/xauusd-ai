# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`.
`GX1_RULES.md` er bindende. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus;
prosess og eksakte terminalkvitteringer overstyrer prosa. Tidligere bølger finnes i Git.

## Gjeldende status 19.11.2026

Brukeren har endret første hovedrunde til et mindre datasett og omtrent
30minutters kjøretid. Oppfølging er hver30.minutt (avløser timeønsket).
Den lange SHORT_FIT001 er stoppet under INITIAL-skåring før første update.
Original CLAIM/TERMINAL og separat OPERATOR_STOP_REQUEST/OBSERVATION er bevart.
Native TimeoutError skyldes vårt styrte stoppsignal, ikke utløpt3timersfrist.
Ingen INITIAL_SCORE/INITIAL_STATE finnes; kildebundet kontrollflyt beviser
0optimizersteg. Ingen ferdig prognosebaseline eller mainresultat foreligger.

Alle trenings-/kjøringsporter er stengt. Én ny, mindre og nøyaktig30minutters
plan skal klargjøres; main001 er konsumert og relanseres aldri. Ferdige
native data, stats og genuine modell-/coldresume-kontroller gjenbrukes.
M1 er aldri retningsgrunnlag. TEST, Entry/Exit, broker og handel er stengt.

### Full historikkfasit

18 originale M5-årsfiler fra 2009–2026 ga 1 215 514 råbarer. Baren som
slutter nøyaktig på TEST-grensen 01.07.2026 er utelatt; faktatapen har
1 215 513 M5-barer. Ingen forseglet TEST-periode er brukt.

| TF | Observerte barer | Første uten forgjenger |
|---|---:|---:|
| M5 | 1 215 513 | 1 |
| M15 | 406 109 | 1 |
| H1 | 102 354 | 1 |
| H4 | 26 427 | 1 |
| D1 | 4 417 | 1 |

Åpning mot forrige observerte åpning og slutt mot forrige observerte slutt
har separate fortegn -1/0/+1. Likhet og blandede fortegn er kjente fakta.
Hver TF har 17 bevarte årsskifter. De 5 558 M5-kildegapene annoteres uten å
fjerne kjente fakta; ingen tomme barer fabrikkeres. Bare første bar mangler
forgjenger. Framtidsmål har separate modningsmasker, og allerede kjent
åpning gis ingen prognosekreditt.

Juni 2026 har 6 024 utviklings-VAL-rader til faktakontroll, ingen fitting.
Historikk003 fullførte med exit 0 og uendret kilde på 334,87 sekunder.
Uavhengig råaggregering gjenskapte alle 76 arrays eksakt.

### Indikatorer og konkrete rettelser

M1 er aldri retningsgrunnlag: heller ikke via klargjøring, akser, mål,
masker, priorer, normalisering, metadata eller warmstart. En kald prosessvakt
avviser M1-artefakter og det gamle 049-registeret før kilde-/modellimport.
M1 tilhører bare senere oppfølging av en allerede åpnet handel.

Native TA har 318 signalfelt, 132 kontekstfelt, 256 felt per TF og 61
D1/makrofelt. Alle åtte spesialistfamilier beholdes. Native squeezeparametre
for fem TF og V29 gjenbrukes uten refit. Feil i tomme helgebarers aggregering,
binære/signed feltdomener, normaliseringsenheter og koblingen mellom
feature-ID og fasittid er rettet. Opprinnelige ID-er og kildegap beholdes.

Pipeline006 kvalifiserte alle 27 eksisterende inputarrays uten ny
råindikatorberegning. De sju faktiske konsumpopulasjonene fikk fersk
normalisering; én fersk 16-raders CPU-initialbatch og identisk strict reload
bestod. M5-apexfeltets målte mangel på estimerbar spredning har en avgrenset,
kildebegrunnet enhetsregel. Ingen generell konstant-/skalafallback er åpnet.
Originale feilkvitteringer fra 001–005 bevares og planene relanseres aldri.

GC går direkte fra rå MBP-hendelser til 300-sekunders grunnlag og videre til
M5/M15/H1/H4/D1. Ingen M1-reduksjon brukes. Alle 66 originalfiler,
892 100 712 hendelser, 23 definisjoner og 15 kontrakter er behandlet.
Råhashbookends og mengde-/stige-konservering består per fil.
RAW_COMPONENTS001 fullførte på 935,924 sekunder uten modell eller råreplay.

ASSEMBLY001 gjenbrukte alle komponentene og fullførte på 537,195 sekunder.
Den fysiske pakken har 1 215 baseverdikonsepter og fem støttefelt.
Ti genuine kjøps-/salgs-maksfelt manglet tidligere i registeret og er nå
koblet inn. De 210 boolske verdidomenene er eksplisitte, inkludert 59 som
tidligere manglet. Invers kontroll fanger uregistrerte primitiveier-felt.

Pakken har 227 808 kalenderbøtter, 152 999 volume-admitterte og 23 430 helt
signerte. De 288 første-dagsbøttene uten kausalt kontraktsvalg beholdes ukjente.
Kjent delvis B/S/N-flow bevares separat. Dokumenterte klokkehendelser,
manglende historikk og svak observasjonsdekning har eksplisitte masker.
At alle kildekolonner finnes betyr ikke at alle verdier er observerte.

### Bestått samlet kontroll på ekte data

JOINT_QUALIFICATION001 er COMPLETE_CONSUMED_GENUINE_JOINT_TECHNICAL_ONLY:
exit 0, uendret kilde, 235,244 sekunder. Ingen ny råreduksjon eller parameterfit.

- Alle 29 964 opprinnelige M5-utstedelser er inputklare. Ingen fjernes fordi
  GC mangler. Normalisering bruker alle 29 963 issues med modne framtidsmål.
- Tolv ferske normaliseringsflater og ferske vekter; ingen gamle statistikker
  eller vekter lastet. Den faktiske modellen mottar 14 inputflater:
  signalbredde 2 748, kontekstbredde 2 562, TA-historikk og egne GC-historikker.
- Åtte spesialistfamilier, 61 makrofelt og ti åpning-/slutthoder er koblet.
  Alle 25 GC-familie/TF-ruter har målte gradienter gjennom observerte
  kontinuerlige markedsverdier og projeksjonsvekter.
- Den kildevalgte kontrollgruppen har 16 rader. Åtte hoder har kvalifiserte
  tapsceller og gradient; to åpninghoder har ingen slike celler på akkurat
  disse radene og korrekt null gradient. Ukjente markedsverdier har null gradient.
- Tre forwards og én autogradberegning, null optimizersteg. Én kontrollert
  endring av en kildekoordinat gir målbar logitendring gjennom produksjonsruten.
  Kontrollvarianten er bare DIAGNOSTIC; originalpakke, binding og normalisering
  er uendret. Strict reload gir identisk initial model-state.
- Null forbudte M1/049-IO-forsøk. Ingen TEST, broker, ordre eller spending.

Fokuserte regresjoner, faktisk kald fullmodell/backward og uavhengig
agentrevisjon er gjennomført. Sluttartefaktenes hasher er kontrollert mot
RESULT/TERMINAL og bundet i policy. Ingen bestått suite eller modelljobb skal
gjentas uten et konkret nytt behov.

### Gjenstår / aktivt mål

Brukeren har opprettet mål om å kvalifisere og gjennomføre én kort offline
retnings-fit. TRAIN låses til 01.05.–01.09.2024 og senere CONTROL til
01.09.–01.10.2024, begge fra den eksisterende kvalifiserte inputpakken.
Dette krever fersk TRAIN-only normalisering: gammel mai–september-statistikk
har sett CONTROL og kan ikke brukes. Rådata, TA-arrays, squeeze/V29-parametre,
historikkmål og fysisk GC-pakke gjenbrukes uten ny produksjon.

Tre avgrensede agentoppgaver er integrert: eksisterende driver har ny
eksakt fit-/resume-rute, fast Laplace1-treklasse-TRAIN-prior og separate
populasjoner. 45 fokuserte capped4G-testfall består. Hovedrunden skal skåre
alle 24 175 TRAIN- og 5 789 CONTROL-inputrader før/etter, med korte haler og
ukjent siste kontrollrad. De to apex-formlene er separat observert på de
faktiske TRAIN-unionene; eksisterende normaliseringskode kan gjenbrukes.

FIT_MECHANICS_20261119_001 er COMPLETE_CONSUMED, exit0/uendret kilde,
450,045s etter trygg fysisk Windows-boot666→667. Ekte CUDA-kjøring ga
2 modeller, 7 forwards og 3 AdamW-steg (to kanoniske, ett kaldt repetert).
Ny prosess gjenskapte neste modell-/optimizer-/RNG-/cursor-/logitstate
bit-identisk. Begge prosesser har null M1/049-IO. Fersk TRAIN-only
12-flate-normalisering og Laplace1-prior er lagret; ingen gamle vekter lastet.
Dette er teknisk kvalifikasjon, ikke full prognosebaseline eller læring.

En konkret ny runtimeblokkering er målt: to B16-skåringer tok35,912/36,605s,
og hvert treningssteg18,886/18,944s. Hele før-/ettermålingen trenger3746
forwards før treningssteg; dette er ikke kvalifisert innen10800s-vakten.
Ren CPU4G-metadataprofil er nå fullført med null modeller/forwards/fit:
full normaliseringskontroll tok9,436s under cProfile;218 kall til
_native_gc_specs brukte8,130s og464 katalogbygginger8,037s.
Den eksisterende validatorløkken rettes til lokal gjenbruk innen samme kall,
med alle kontroller og verdier bevart. Ingen persistent cache innføres.
Ny negativ-/kompleksitetstest består. Samme metadata-etterprofil002 har
fullnorm2,417s, history0,752s, provider2,062s og profilecompile1,211s;
original TRAIN-normalisering er uendret og null M1/modeller/updates er brukt.
Faktisk ny GPU-throughput og minne skal kvalifiseres i egen mechanics002.
Den gjenbruker bare original001 TRAIN-normalisering/prior/akse og starter
ferske vekter med samme initialSHA. Ingen stats refittes eller kopieres.
Scoring deklareres B64, trening beholder B16. Alle 24 175/5 789 inputrader
beholdes: 378/91 batcher per hel skåring, siste haler47/29, totalt938 forwards
for før/etter. Ny002port har8fokuserte tilfeller PASS; driveren27 PASS.
Normaliseringsløkken har egen negativ-/kompleksitetstest PASS. Mechanics002 er nå COMPLETE_CONSUMED med opprinnelig plan-SHA
79cb674b0dc62e3071b2a9f175fedb121774b3277de07a788aac6195a503439e.
Ekte GPU-kjøring fullførte179,343s/exit0/uendret kilde,2modeller/7forwards/
3updates. Kald resume er bit-identisk og begge prosesser har0M1/049-IO.
Initial og final model-state er identiske med001;0norm-/priorfit og ingen
gamle vekter er lest. B64-to-batchskåringer tok12,478/12,826s; B16updates
6,889/6,771s. Målt peakCUDA-allokering1,891GB ligger under eksisterende vakt.

SHORT_FIT001 er OPERATOR_ABORTED_CONSUMED_BEFORE_FIRST_UPDATE.
Den tidligere store24 175/5 789-før-/ettermålingen er avløst av brukerens
nye mindre scope, ikke fullført. Native0updates og kvitteringer bevares.
Nytt scope er1536FIT-rader (96spredte B16blokker) og512senereCONTROL
(8spredteB64blokker). Begge inkluderer første/siste poolrad, også ukjent
sisteCONTROL. Pool/rank-hasher fryses før labels/prognoser. Originalnormen
fra24175TRAIN-poolrader gjenbrukes, mens matchet Laplace1-prior fittes én
gang på nøyaktig1536FIT-rader.96updates+64scoreforwards=160forwards.
Konservativ20prosentprojeksjon er1502s≈25min innen1770/1800s.
32driver- og11nye/berørte guardtilfeller består capped4G. Ingen modelljobb.

Den kvalifiserte mekanikk002 gjenbrukes via verifisert Gitblob/fullmodul-AST
for uendret modellmatematikk, optimizer, RNG og checkpoint. Bare navngitt
kohort-/prior-/planorkestrering og fire eksplisitt hashbundne filer kan
avvike; alle andre kildeeiere må være eksakt uendret. Nye små kohorter
krever ingen ny GPU-mekanikkrunde. SHORT_FIT002 er PREPARED_NOT_ADMITTED med plan-SHA
bb841bfced12176a16bd9a0d12c2777427f9475871cbab6f305dbd7553891daf.
32drivertester,11nye/berørte guardtilfeller, AST/diff og uavhengig
kilderevisjon består. Eksakt CLI-førkontroll og engangsadmisjon gjenstår. Ingen
utfallsstyrt seleksjon, ny råproduksjon eller relansering av konsumerte planer.

### Kilderevisjon før M1-presiseringen18.11 (historisk)

- Statisk sjekk av 2 106 inngående sporede filer og 995 187 tekstlinjer.
  Syntaks: 1 379 Python-, 636 JSON-, 9 shell- og 9 PowerShell-filer
  samt commit-hooken. Ingen dupliserte JSON-nøkler eller manglende
  direkte gx1-modulimporter. Tre avgrensede
  agenter sporet data/indikatorer, modell/fusjon og fasit/tap/skåring.
  Dette dekker statisk inventar og målrettet semantisk kontroll.
- MTF-fusjon blandet familieverdier innen en TF ved å dividere en global
  softmax med en klampet TF-masse. En syntetisk felles +100-forskyvning for
  én TF endret 605/640 lokale tokenverdier i den faktiske ruten. Direkte
  softmax innen hver TF bevarer de lokale verdiene i vitnet og beholder
  global vekt. Forekomst i gamle ekte forwards og retningsvirkning er ukjent.
- Lukket M1-prisbane krever siste observerte M1-sluttpris som anker.
  11 FIT-rader og 0 senere rader hadde tidligere minutter, men manglet
  ankeret. M1-grenen er nå identitet der, også med trent bias. Fysisk tape
  bevarer observasjonsmasken. Endret modellkontrakt avviser gammel identitet.
- M5-vending krever riktig destinasjon. Samtidig flip til motsatt side
  telles separat. Sammenligning på tvers av TF ved samme utstedelse
  oppgir sine faktiske native horisonter. Fjernet ubrukt Fold2 FIT-akse
  fra en scorer med kun SCORE-visninger.
- Fire urefererte familie-rapporter fra 31.10 er fjernet fra repo etter
  kontroll av referanser, hashbinding og eierskap. De tolv original-
  referansene er fortsatt hashbundet i policy. Originalfiler og DATA/RUNS
  ble ikke endret. Fjernet ubrukt tomt overstyringsregister i
  feature-eieren. Samme 10 lag og 165 utgående felt, feltrekkefølge,
  aliaser og kontrakthasher.

Kontroller: 54 integrerte regresjons-/bindingstester og 53 fokuserte
featuretester bestod. Syntaks, stale-path-scan, gamle plan-/resultat-/
terminalhasher og `git diff --check` bestod.

Den kjørte TA-visningen ruter alle åtte familier og 61 D1/makrofelter.
Dedikert S/F per TF og den eksisterende GC-ruten har forskjellige
komposisjoner. Ingen full joint GC-kjøring, feltvis retningsnytte eller
effekt av rettelsene er målt. På tidligere FIT har M5 fast sidevekt per
kjent rad. De fire andre TF bruker batchvis støtte, mens skåringen vekter
radene likt. For eksempel endret dette den konstante native F-D1-
referansen fra 66,69 % OPP radvis til 64,18 % batchvis. Virkning på
rangering er ukjent. Den konsumerte planen har heller ingen direkte
hashbinding av enkelte faktiske S/F-, taps- og AdamW-eiere. De undersøkte
blobene var uendret fra planens opprinnelige commit til revisjonsstart.

Før en ny fit kreves fersk kildebundet baseline for endret forward,
hashbinding av faktiske kilde-/inputeiere og én falsifiserbar
retningshypotese på fast FIT og full senere utvikling. Ingen modelljobb eller ny plan er innrømmet.
Trening, Full GC, TEST, ONLINE, broker og handel er stengt.

Historisk kontekst følger. Arbeidsstatus og eksakt gjeldende terminal
eies av `NEXT_RUN_POLICY.json`.

## Fullrevisjon 17.11.2026

Brukeren ba om hele-prosjekt-sanity check, avgrensede agenter og sletting av
bevist foreldede filer. 2 069 sporede filer / 984 774 tekstlinjer er maskinelt
skannet; alle 1 353 Python-, 625 JSON- og 9 shellfiler parser. Dette er ikke
manuell semantisk godkjenning av hver linje. Avgrensede agenter reviderte feature/modell, mål/tap/score, tester,
status/kontrakter og kjøring/retention. Ingen trening eller TEST.

Konkrete retention-funn: eieren ignorerte `PermissionError` i `/proc/*/fd` og
`fdinfo` og kunne derfor feilaktig godkjenne sletting uten fullt
åpen-skriver-bevis. Minste fail-closed rettelse og to negative tester er gjort;
162 fokuserte tester besto. Fysisk normal bruker får nektet 27/31 fd-tabeller;
root kan lese alle. Root-capped audit med 4G/512M er verifisert og en ytre
`flock` på andre2s eksisterende prosjektlås blokkerer parallell jobb på
samme inode. Retention-eieren slettet eksakt én ureferert gammel run,
`HISTORY2009W_EARLY_DECISION_20260927`: 470 filer, 17 mapper og
2 600 617 478 bytes. Plan SHA
`3e3150233e10c4e19413f0e2f8f82cf4a0d951a8c5e2a9ba412c2a81df7a1bc7`,
godkjenning SHA `3509d43f1a71f094cf6b2b4408d1f518da53baf6ef1f85d0bec71e2343419a4b`,
DELETE_COMPLETE-rapport SHA `160406465b2e4588f1aab45b43082b25db558b02890760195a30fb26b8dc92f3`.
Sti og karantene er borte. Eksakte kvitteringsstier og byteantall står i
`NEXT_RUN_POLICY.json`.

Videre revisjon fant at retention-eierens gjeldende-dokument-sjekk overså
relative `GX1_RUNS/...` og `GX1_DATA/...`-navn. Minste fail-closed rettelse
normaliserer dem til eksakte hjemmebaner og bevarer absolutte referanser.
162 fokuserte tester besto; den gamle V12-topprooten avvises nå korrekt fordi
et gjeldende prereg-dokument navngir en manifeststi under den. Eieren
planla, godkjente og slettet bare tre interne V12-undermapper: 35 filer,
4 mapper, 904 784 968 bytes. Plan SHA
`99ae8a2546c16e5c92cbf8e22b9f6915e0220f04fbe0a8234ddf2ff225aa2e04`,
godkjenning SHA `b52190a8a83c79cd638c6671442031799c0361c25927e840f65eee60bdfdc418`,
DELETE_COMPLETE-rapport SHA `0daa22ef9182cf38f0e5e94a2a5141a4bd7fcca6ede0968085f679e5d1d23b67`.
Alle tre mål og karantener er borte; walkforward-mappen og V12-roten står.
Eksakte stier ligger i `NEXT_RUN_POLICY.json`. Ingen andre GX1 DATA/RUNS er rørt.

Mål/tap-revisjonen fant en målbar side-skala-feil: `side_pair_loss` brukte
kvadratsummen av separate LONG/SHORT-standardavvik og ignorerte negativ
kovarians. Faktisk FIT-standardavvik for kvalitetsforskjellen er 1,315–1,383
ganger større på alle seks horisonter. Koden bruker nå direkte parvis
FIT-skala med ny v2-kontrakt og tester. Rettelsen er ikke trent eller skåret;
gamle plan-, modell- og resultatkvitteringer er uendret.

Feature-revisjonen fant at den tidligere «eksakt duplikat»-begrunnelsen for
å fjerne fem-bars EMA50/200-spreadendring var feil etter at EMA200-slope ble
målt over 20 barer. Indikatoren er nå gjeninnført på lokal og alle fem
TF-baner, med faktisk signalbredde 318 og per-TF-bredde 256. Eldre cold
feature-view, normalisering, modellmatriser og checkpoint er historiske og
kvalifiserer ikke den nye profilen. Ny materialisering og eksakt FIT/later-
kvalifisering trengs før noen fremtidig samlet plan; ingen slik jobb er åpnet.

Statusrevisjonen rettet foreldet `active_stage`, `next_action`,
`current_work` og `learning_gate.current_evidence_scope`: gjeldende evidens er
den konsumerte kalde M5-runden, ikke den eldre Fold2-runden. 729 testfiler og
681 dokument-/konfigurasjonsfiler er maskinelt skannet. Gamle planers fysiske
fil-/SHA-referanser er fortsatt intakte. To fokuserte historiske inputtestfiler
ga 92 PASS, 47 eksplisitte SKIP for gamle hashbundne 317/255-kvitteringer og
0 feil; aktive syntetiske klokke-/akse-/normaliseringsvakter besto. Dette er
ikke full manuell semantisk lesning av alle nær én million kodelinjer.

Diskrevisjonen målte 669,64 GiB brukt på Windows C:. Ubuntu-VHDX brukte
456,52 GiB av dette; inne i WSL var om lag 412 GiB brukt, hvor GX1_DATA og
GX1_RUNS sto for rundt 335 GiB. På C: ble 274 gamle, eksklusivt lesbare
Temp-GUID/`swap.vhdx`-filer re-sjekket og slettet: 117 214 019 584 bytes.
Én aktiv låst swapfil ble bevart. Deretter ble 222 gamle `wct*.tmp` og én
gammel installer, 1 331 148 525 bytes, slettet etter ny alders-/låskontroll;
åtte nyere `wct` og nye/aktive RdClient-logger ble beholdt. Samlet direkte
C:-sletting var 118 545 168 109 bytes (110,40 GiB); siste målte frie plass
var 397 166 620 672 bytes mot 278 856 753 152 før første sletting.
To gamle, urefererte `/var/tmp/gx1-*-next-batch.pt`-filer på 536 371 599
bytes ble slettet under prosjektlås etter eier-/alder-/åpen-fil-kontroll.
GX1 DATA/RUNS-kartleggingen fant at de største datasettene og kjøringene
er bundet til gjeldende kvitteringer eller dokumenter. Ubuntu-VHDX er ikke
sparse; komprimering krever separat avsluttet WSL og er ikke utført.

## Lagringsstatus og komprimering 17.11.2026

Ny måling: Windows C: har 601 409 798 144 bytes brukt (560,11 GiB) og
397 167 644 672 bytes ledig. Ubuntu-VHDX er 490 187 259 904 bytes
(456,52 GiB), ikke sparse; ext4 inni bruker 438 184 099 840 bytes
(408,09 GiB). GX1_DATA + GX1_RUNS + GX1_CURRENT utgjør om lag 340 GiB,
ikke 500 GiB. Ubuntu-filen rommer også separat EURUSD, Linux-system,
virtuelle miljøer og slack. Av cirka 166 GiB prebuilt er en 37 GiB-mappe
fysisk kilde for den konsumerte M5-featurevisningen. Store øvrige mapper er
navngitt i gjeldende policy/dokumenter; urefererte toppmapper utgjør bare
om lag 0,45 GiB i RUNS og 1 MB i prebuilt før retention-grafkontroll.
Å komme ned mot 200 GiB krever avvikling eller ekstern arkivering av minst
om lag 140 GiB navngitt historisk evidens, ikke enkel cache-sletting.

Kapslet `fstrim -v /` fullførte og rapporterte 630 501 879 808 trimmete
bytes; dette er ledige blokkmarkeringer, ikke frigjort plass på C:. VHDX-
lengden og C:-plassen var uendret etterpå. Grov øvre størrelseforskjell
mellom VHDX og brukt ext4 er 52 003 160 064 bytes (48,43 GiB), ikke en
forventet komprimeringsgevinst. To interaktive WSL-shell er åpne; full lokal
backup av VHDX får ikke plass på C:. Operatøren valgte deretter å droppe
komprimeringen. Ingen `wsl --shutdown`, DiskPart eller VHDX-komprimering er
utført. `NEXT_RUN_POLICY.json` binder denne beslutningen.

Skjult hjemmelagring ble skilt fra GX1: eldre Cursor-runtimeversjoner,
Cursor-snapshots, Codex-arkiver, Python-miljø og papirkurv. Fire gamle
Cursor-runtimeversjoner uten åpne filer/referanser ble slettet under
prosjektlås: 1 767 837 696 allokerte bytes. Nyeste runtime står.
Papirkurven (~1,30 GB) inneholder gjenopprettingskopier av prosjektdata og
checkpoints og er bevart. Cursor-snapshots (~13,79 GB), Codex-økter
(~6,33 GB) og Python-pakker (~6,00 GB) er også bevart. Denne slettingen
reduserer ext4-bruk, men ikke den uendrede ikke-sparse VHDX-lengden på C:.
`fstrim` ble kjørt før Cursor-slettingen og må gjentas før en senere
komprimering.

## Gjort og målt 17.11.2026

- Regelen om direkte M5-edge, aktiv mismatch-kontroll, sparsom tokenbruk og
  ingen arbeid for å fylle ventetid står i `GX1_RULES.md`. Avgrensede agenter
  reviderte kilde → input → fasit → tap → prognose → valg → kostet utfall,
  featurekonsum og opprydding. Målte kald-rute-feil ble rettet før fit:
  senere dekning, Fold1-prior, uobserverte enheters profil og SCORE-scope.
- Eksakt kaldt view fikk fysisk Oct–Dec-dekning på 17 489 rader. Rå FIT-/SCORE-
  gather, fersk normalisert modellinitial og kosttap besto før én samlet plan.
  Prefiksnormaliseringen fra 29 574 tidligere FIT-rader ble gjenbrukt; 275
  gaprader ble ekskludert. Den gamle 4 096-radersdekningen tilhører et annet
  view og ble ikke brukt. Ingen inputvakt ble omgått.
- `TA_M5_COLD_JOINT_FIT_20261117_001`, plan SHA
  `f6d2d368ebc48280dd688a35c58182a63401384a01a90a5eb0722f04037697b6`,
  kjørte én gang på fysisk boot 664 under kapslet trainer-vakt. Kildefilenes
  innhold var uendret gjennom kjøringen. 29 574 FIT-rader, 1 849/1 849
  optimizersteg, 1 849 treningsforwards og 2 × 1 094 senere scoreforwards;
  null hoppede batcher. Kjøretid 6 395,58 sekunder. Terminal `COMPLETE`, exit 0.
  RESULT SHA `716fc82d6e2e13499dd2dce38b0274c00d4ab11e7d303952069b5d5c07d5e9e7`;
  TERMINAL SHA `b9c8fff72e3213d074c8498091750384503f19568268206b9fbc26f9835760bc`.
  Checkpoint SHA `deaf8d615198c7203e71ea9a486cb8187f12cdde33d29207034365a82f071c14`;
  SCORE-capture SHA `72eb98b91642dc4b73f713d57952f566ce6c495e3f0beff3250763eea7ab3f7f`.
  Terminalreferanser, antall og checkpoint-state-hash er kontrollert uavhengig.
- Fersk initial valgte 17 489 FLAT. Trenet modell valgte 8 289 LONG,
  0 SHORT og 9 200 FLAT. 7 365 valgte terminaler er kjente: kostet netto
  −6,622 bps i snitt; 924 er ukjente. Av 8 289 valg har 6 112 komplett
  prisbane og 2 177 ukjent bane. På komplett valgte baner var predikert
  kvalitet (netto minus pre-MFE MAE) +3,159 bps mot faktisk −22,173 bps.
  Predikert netto var +17,990 mot faktisk −5,593 bps; predikert MAE
  14,831 mot faktisk 16,580 bps. Kvalitetsgapet er −25,332 bps.
- På 11 318 komplette 480-minuttersbaner rangerte modellen LONG best
  i 97,06 %, fasiten i 53,45 %. Side-treff var 52,78 % og korrelasjon
  mellom predikert og faktisk sideforskjell 0,035. Korrelasjon for LONG-
  kvalitet var 0,043. Kvalitetssnitt for valgte komplette baner var
  −2,045 bps i oktober, −22,787 i november og −26,203 i desember.
  Forhåndsvalgt alltid-LONG på alle komplette h480-baner ga −8,216 bps
  netto markout; dette er ingen posisjonsført handelsreferanse.
- Den eldre varme utviklingsmodellen valgte 0 LONG og hadde +3,246 bps
  predikert mot −16,167 faktisk kvalitet på 4 031 komplette valgte baner.
  Den kalde modellen endte med ensidig motsatt sidevalg. Forskjellig
  FIT-periode og view gjør at årsaken ikke kan tilskrives tapsendringen alene;
  den skapte ikke dokumentert side- eller prisbaneedge. Gjenbrukt historisk
  utvikling kan ikke bevise uavhengig edge.
- Oppryddingsrevisjonen slettet én bevist ureferert gammel RUNS-mappe
  gjennom retention-eieren; eksakte kvitteringer står over. Ingen repo-kode
  eller andre DATA/RUNS er slettet uten tilsvarende eierskapsbevis.
  Gammel handover-prosa ligger i Git.

## Tilstand etter kald runde 17.11 (historisk)

Ingen ytterligere fit, TEST-lesning, forward-journal, ONLINE-baseline, broker eller
handel er åpnet. Posisjonsført PnL, alle åpne posisjoner og uavhengig
økonomisk generalisering er ikke målt. Ikke bruk positive terminalsnitt,
FLAT eller tekniske PASS som edge.

Ikke start flere modellrunder fra dette resultatet. For ny 318/256-
featureprofil må faktisk full FIT/later input, normalisering, kausalitet og
alle hasher kvalifiseres på nytt før en eventuell ny samlet treningsplan kan
vurderes. Gjenbruk gamle pris-/mål- og sidecarverdier bare når deres klokke og
bindinger fortsatt består. Ingen ny plan eller trening er admittert; TEST
forblir forseglet. M5-forward-journal krever separat vedtak.

## Aktiv 318/256-inputkvalifisering 17.11.2026

Den gamle 317/255-kilden kan ikke relabeles som ny featureprofil. Den gamle
kalde rågatheren kontrollerte fire eksempelbatcher, ikke samtlige FIT/later-
verdier. Nye kildekontroller bygger kun EMA50/200-spreadendringen fra kausale
prisbarer før 31.12.2024 og legger seks hashbundne verdiarkiver oppå gamle
pris- og spanverdier. Gammel enhets-/normaliseringsidentitet bevares; den nye
318/256-identiteten trenger egne FIT-statistikker for det nye feltet.

Engangsplan `TA_M5_EMA_DELTA_SIDECARS_20261117_001`, SHA
`e2e9f9233e27223099f8905a003e05f637cfa0c7e0325485d1d55b2f89223b08`,
besto ren preflight og eksakt kilde-SHA, men den kapslede jobben stoppet på
`TA_EMA_DELTA_OLD_SLOPE_PARITY:M5`. CLAIM ble opprettet, ingen RESULT ble
publisert. Terminal er `FAILED_CONSUMED`, exit 1, kilde uendret; terminal-SHA
`5eac4f13ad8f9d0c768b58cd7db006d9744dc77b5c3d2ecc988fe112fd304eee`.
Engangsplanen er lukket i `NEXT_RUN_POLICY.json` og skal aldri relanseres.
Dette er en konkret gammel/ny M5-verdiparitetsfeil, ikke kvalifisert ny input.
Neste steg er å måle årsaken til paritetsavviket og rette bare denne sperren
før en eventuell ny plan. Ny normalisering, view, full rå/normalisert gather og
modellinitial er ikke gjort. Alle treningsflagg er stengt; ingen modellforward,
trening eller TEST fant sted.

Årsaken til `_001`-feilen er målt: gammel MTF-cache er beregnet fra
manifestbundet native M5-historikk som starter i 2009 og deretter beskåret til
M5_ENRICHED fra 2010. Den første nye jobben startet MTF-beregningen på den
beskårne filen. Med full native-kilde og opprinnelig prisoppløsning stemte begge
gamle EMA-helningene bitlikt på 550 kontrollerte M5-rader; 840 kontrollerte
2024-pris-/klokkerader stemte eksakt mellom native og M5_ENRICHED.

Korrigert engangsplan `TA_M5_EMA_DELTA_SIDECARS_20261117_002`, SHA
`a575667f96460b1ed282137e1069fba1d0842f64748cc306089e055798463c42`,
er fullført og konsumert, terminal-SHA
`2b697d3c86a61a134de2e41dee7ff7f0f21b70b85e9a37da257a8c3b44f0dbb5`.
Seks fysiske sidecars er verifisert mot filhash og gammel EMA-helning bitlikt
på samtlige rader: lokal 47 614, M5 47 534, M15 15 903, H1 4 058, H4 1 134,
D1 425. Materialiseringen dekker 29 574 FIT og 17 489 senere valgte rader;
modellforward, optimeringssteg og TEST-lesning var null. Dette kvalifiserer
verdiene, ikke full modellinput eller edge. Neste steg er FIT-only statistikk,
ny 318/256-visning og full rå/normalisert FIT/later gather. Trening er stengt.

FIT-only normaliseringsplan `TA_M5_EMA_DELTA_FIT_NORMALIZATION_20261118_001`
(SHA `68f987910976de6ec0d24c1b5787e8773c01abf1672c1fe304561b5619bc1fc2`) er innrømmet som én auditkjøring: seks nye
statistikkfitter på gammel Fold1-prefix, null modellforward og ingen senere
statistikk. Planen er fullført og konsumert; terminal-SHA
`47eeacd82df0a18f5da5693382131aa62c15561e67c542572ad59f7ec2d260d8`.
Seks nye felt ble normalisert bare på gammel Fold1-FIT-prefix, 29 574 valgte
FIT-rader; gamle felt ble kopiert uten ny fit. Ny TA-projeksjon har 318/256
felter, uten modellforward eller TEST. Den nye visningen og full FIT/later
rå/normalisert gather gjenstår før eventuell modellinitial.

Ny 318/256-pakke og visning er publisert som metadata uten modell eller
trening. Visningsfil-SHA `638b3706ebc4566439599e27826cd00322d8e0156166333e507a4707bd1bac25`; RESULT-SHA
`0372b18d1fd80e7d8ae5efd0fadd1d56413ff18d406a5dc1a96c85756b6c2139`. Alle seks faktiske EMA-arkiver besto direkte
inputvakt. Første bygging av full-gather-plan avdekket én konkret kontraktfeil:
constructorens 2 728 råfelter ble forvekslet med 318 TA-projiserte felter.
Feilen er rettet med egen regresjonstest; ingen gammel jobb ble claimet.
Ny engangsplan `TA_M5_EMA_DELTA_FULL_GATHER_20261118_001`, SHA
`a8d1a39819402ca8fe80bb53e459f8431308fc7ea3153ac0a6e7ef65f2ff4864`, er innrømmet som targetfri
FIT/later inputaudit: 47 063 valgte rader, 2 943 B16/tail-batcher, alle åtte
rå og normaliserte flater. Ingen modell, optimering eller TEST er åpnet.
Kapslet audit er fullført og konsumert på 1 544,39 s. RESULT-SHA
`f1589bff55919ac9a1bcb0c9ed99ae7848c51e5dd34bf453d9bd6d92d34b8f2a`, COVERAGE-SHA
`d3c95a7a16a3a63c3276ba5f90dfa3b99a93957edea302cf990d521c3c20325c`, TERMINAL-SHA
`57ff529479000918ac382268e857d518c8d89f7ebe8111bad16aedb4a6149ce3`. FIT: 29 574 rader/1 849 batcher;
senere: 17 489 rader/1 094 batcher. Åtte flater har rå og normalisert
fullhash i begge perioder, alle 23 544 normaliseringskall fullført. Målarrays
dekodet 0, modellforward 0, normaliseringsfit 0, optimeringssteg 0, TEST 0.
Kilde uendret. Dette kvalifiserer input, ikke læring eller edge. Neste steg er
en fersk 318/256-initialmodell med samme FIT/later aksene; treningsflagg er
fortsatt stengt. Fysisk Windows-boot er fortsatt 664; ny fit krever >664.

Gammel Fold1-endepunktprior `loss_contract_sha256=63cb7337…` avvises av dagens
318/256 F-objektiv (`2bcdacf1…`). Også gammel FIT-statistikkhash `284b…`
er foreldet; v2-statistikk på den uendrede 29 574-raders FIT-målfilen er målt
til `1e1278c2eced0a2344f2a7162fa8d54f5e5cb49d92640c59edf3edd38976fd30`.
Ny target-only engangsplan `TA_M5_EMA_DELTA_FOLD1_PRIORS_20261118_001`
(SHA `d249c24d449324a3591b0af8426256e05a97618edf7959a879bee9dc48118148`) er innrømmet med null
modellforward og null senere mål i priorfit. Ren preflight og én kapslet audit besto. Ny prior-fil SHA
`1fdd70246c6dbb295819ec0dcfe45ab5c6260d8767ca8e4cdbd516591609f009`, RESULT-SHA
`55a9cdd4a54cf612434f6e3ce8a650b49f8514fc04b26f7f0decfceafde28267`, TERMINAL-SHA
`a4a948fc6c49989eca42f49b7841363a8dc5e471ac3c55b275b866f12a118965`. Dagens Fold1/F-objektiv godtar prioren;
29 574 modne FIT-rader, null modell og ingen senere mål i priorfit. Planen er
konsumert. Neste steg er fersk initialbaseline; fit-plan er ikke innrømmet.

Fersk initial-engangsplan `TA_M5_EMA_DELTA_INITIAL_PREFLIGHT_20261118_001`
(SHA `eec69cb41408f43485a047ee764c4cf5810a920584068b2964de0a1c3c3d8f4f`) er innrømmet: én ny
318/256-modell, én FIT B16 v2-loss og én senere B16 score, null optimizer.
Planen binder full gather, ny Fold1-prior, dagens FIT-statistikkhash og uendret
måltape. Ren preflight og én kapslet CPU-initialaudit er fullført og konsumert.
RESULT-SHA `8fe95ca667c1ba814a6c0b9785218891cf9bae5e4e0d2fb01cbddbb2a3972adb`; TERMINAL-SHA
`5b121cf92eab315db9102cec583b632641cf9d72bbc8734c840f9f577922e344`. Faktisk ny modelltilstand
`b00f694b982ba6e46742e51e7de6b9c6376b109ae8b987c25f9dbc3e90e57c40` og kjerne-/beslutningskontrakter
er publisert. To forwards, null optimizer. I begge B16-batcher var største
avvik mellom modellens normalisering og referanse null på alle åtte flater.
Dette er mekanisk baseline, ikke læring/edge. Neste steg er én eksakt fit-plan;
Windows-boot 664 tilfredsstiller ennå ikke kravet om en senere fysisk boot.

Eksakt én-epoke kandidat `TA_M5_EMA_DELTA_JOINT_FIT_20261118_001` er
forberedt, **ikke innrømmet**, plan-SHA `0ce0eecd763c3b9f55105f4ee6eb1d9e3ffbee0d1b1a1635577b647d5c4f912c`.
Den binder de nye fullgather-, prior- og initialkvitteringene, 29 574 FIT,
17 489 senere, 1 849 maksimale oppdateringer og 4 037 planlagte forwards.
TRAIN/TEST/ONLINE/broker er fortsatt stengt. Ny fysisk Windows-boot >664
kreves før eventuell engangsadmisjon og capped trainer-kjøring. Ingen ny
modellfit, checkpoint eller edge-bevis finnes fra dette viewet ennå.
Direkte preflight på ren commit `003adda6` besto, outputrot er fraværende og
CLAIM er ikke opprettet. Den rapporterte `fresh_physical_boot=false` på
Windows-boot 664. Engangsfit forblir PREPARED_NOT_ADMITTED; neste steg krever
operatørens valg om fysisk omstart, deretter ny boot-/kilde-/maskinvarekontroll
før eventuell admittering.

Windows ble fysisk startet om fra boot 664 til 665 etter ledig jobb-/GPU-/lås-
kontroll. Ny preflight på ren commit `226507f4` besto; plan-SHA er uendret
`0ce0eecd763c3b9f55105f4ee6eb1d9e3ffbee0d1b1a1635577b647d5c4f912c`, outputrot fraværende og CLAIM ikke opprettet.
Eksakt én `TA_M5_EMA_DELTA_JOINT_FIT_20261118_001` er nå innrømmet i policy
for capped trainer 20G/512M, 29 574 FIT, 17 489 senere, høyst 1 849
optimizersteg og 4 037 forwards. Bare de fire nødvendige treningsflaggene
er åpnet for denne engangsjobben; TEST, Full GC, ONLINE, broker og handel
forblir stengt. Kilde og plan skal fryses fra launch til terminal.

## EMA 318/256 én-epoke FIT — terminal og vurdering 18.11.2026

Etter ledig jobb-/GPU-/lås-kontroll ble Windows fysisk startet fra boot 664
til 665. Ren kilde `7c7a8bd9` innrømmet eksakt én plan
`TA_M5_EMA_DELTA_JOINT_FIT_20261118_001`, SHA
`0ce0eecd763c3b9f55105f4ee6eb1d9e3ffbee0d1b1a1635577b647d5c4f912c`.
En første wrapper-kommando med relativ Python-sti ble avvist før CLAIM;
samme uclaimede plan startet deretter med absolutt sti gjennom kapslet
trainer 20G/512M. Ingen annen fit ble startet. 29 574 FIT-rader,
1 849/1 849 optimizersteg, 1 849 treningsforwards og 2 × 1 094 senere
scoreforwards, null hoppede batcher. Terminal `COMPLETE`, exit 0,
6 117,67 sekunder; kilde uendret og TEST ikke lest. RESULT-SHA
`6cd1d9176765ec73ff4c6230b79490c2459bd0d79d5f13d657f3511e2143aa0c`,
TERMINAL-SHA `33871aa1f5e5e84d37eee9efea7a391659022dcb60ba1b597f7c86f60d3fde00`,
checkpoint-SHA `5e370c6220e7e16236a0941933919541d181cd0638956394e0a60d288009c2d5`,
SCORE-capture-SHA `11ae0f94c82e6e8471175859c28349a8f883bf84e831b5da6ae2a3cb0a7916a9`.
Kvitteringsreferanser og resultatantall er verifisert; SCORE-capture er
uavhengig rekalkulert mot den uendrede faste målfilen.

Fersk initial valgte alle 17 489 FLAT. Trenet modell valgte 7 450 LONG,
0 SHORT og 10 039 FLAT. Av 7 450 valg hadde 6 512 kjent terminal og
938 ukjent terminal; kostet netto på kjente terminaler var −11,073 bps.
5 288 valgte baner var komplette, 2 162 ukjente. På komplette valgte
baner var predikert kvalitet +5,545 bps mot faktisk −27,486 bps;
predikert netto +20,784 mot faktisk −10,610 bps. Dette er ikke edge.

På de samme 11 318 komplette h480-banene som forrige kalde modell rangerte
den nye modellen LONG best i 99,93 %, mens fasiten gjorde det i 53,45 %.
Side-treff var 53,38 %, men predikert/faktisk sideforskjell korrelerte
bare 0,0073 (forrige modell 0,0349). M5 F-retnings-AUC var 0,509 og S
0,512; alle fem F-sidetaps Brier-tall på senere utvikling var dårligere enn
FIT-prioren. Av 1 301 faktiske M5-vendinger fanget den 92 og varslet 78
falske. Struktur-headet lærte high/low (AUC 0,784/0,781), men dette ga
ikke dokumentert framtidsretning eller gode kostjusterte valg.

Et direkte målskifte er målt på h480: I 18 901 komplette FIT-baner var
LONG minus SHORT faktisk kvalitet +10,724 bps i snitt; i 11 318 komplette
senere baner var den −5,303 bps. Oktober/november/desember hadde faktisk
sideforskjell +7,070/−16,613/−7,910 bps, mens modellens predikerte
sideforskjell var +32,484/+50,424/+51,606 bps. Samme lengre horisontskifte
fra positiv FIT-sideforskjell til negativ senere sideforskjell sees også ved
60 og 240 minutter. Dette viser en konkret FIT→senere mismatch og at modellen
forsterket LONG-biasen; det isolerer ikke alene om feilen ligger i signal,
modell, mål eller markedsskifte. På det faste h480-panelet var predikert/faktisk
kvalitet −2,796/−25,463 bps for LONG og −47,088/−20,160 bps for SHORT:
begge sider er feilkalibrert, ikke bare seleksjonen. 7 380 av 7 450 valg
lå på h480. På det samme faste panelet ga modellens foretrukne h480-side
(8 SHORT av 11 318) kostet netto −8,324 bps og kvalitet −25,566 bps, mot
forhåndsvalgt alltid-LONG −8,216 bps netto og −25,463 bps kvalitet. Dette er
overlappende markout, ikke posisjonsført handels-PnL.

Resultatet gjelder gjenbrukt oktober–desember-utvikling, ikke uavhengig TEST,
handels-PnL eller posisjonsførte åpne handler. Den nye EMA-indikatoren og
rettet side-skala ga ingen målbar forbedring i sidevalg; endringene kan ikke
isoleres kausalt fra hverandre i denne ene fit. Engangsplanen er konsumert,
treningsflaggene er lukket, og ingen ny plan er innrømmet. En fokusert
test som antok fravær av resultatrot etter kjøringen, er isolert til en
temporær no-claim-mappe; modell og historiske kvitteringer er uendret. Neste
direkte arbeid var diagnose av side-rangering og kvalitetskalibrering med de
fryste FIT/later-kvitteringene; resultatet følger nedenfor. Full GC, TEST,
ONLINE, broker og handel forblir stengt.

## Diagnose med fryste FIT/later-kvitteringer 18.11.2026

Tre avgrensede lesere reviderte mål/tap/valg, indikator- og TF-konsum samt
treningsveien. Ingen ny modellforward, optimizeroppdatering, TEST-lesning eller
kildeendring. I den reviderte ruten var LONG/SHORT-orden og klokker like fra
bid/ask-mål gjennom tap, lagret prognose og scorer; alle 17 489 senere
radkoblinger og handlingsantall ble uavhengig gjenskapt. Begge beslutningssider
har ikke-null trente vekter; 0 SHORT er ikke en død eller frakoblet utgang.
Den nye EMA-endringen når lokal input og alle fem kausale TF-baner. Lokal/M5-
verdien er ikke konstant og bare 0,02 % av senere verdier ligger utenfor
observert FIT-spenn etter normalisering. Dens enkle korrelasjon med h480-
sidekvalitet var +0,027 på FIT og −0,043 senere; dette beviser verken
prediktiv nytte eller mangel på ikke-lineært samspill. Gjeldende TA-profil
utelater GC/MBP-1 med vilje, så denne fitten tester ikke full featureflaten.

På komplette h480-baner kom FIT→senere forskjellen hovedsakelig fra netto
LONG−SHORT: +9,901 til −4,688 bps. Forskjellen i pre-MFE MAE var
−0,823 til +0,614 bps. Kvalitetsforskjellen +10,724 til −5,303 bps er et
observert utvalgsskifte. Ukeblokk-bootstrap med 20 000 trekk, frø 20261118 og uavhengig
resampling av ISO-ukers summer/antall på 22 FIT- og 14 senere uker ga
95 % intervall [−36,994, +4,636] bps for endringen;
null er inkludert, så dette er ikke et statistisk etablert regimeskifte.

Den aktive markedstilstandsfasen prioriterer S/F-retning før Entry. Denne
EMA-runden trente kostjustert M5-prisbane som hovedtap, med S/F og struktur
som hjelpetap multiplisert med 0,10. Hjelpetapet sto for omtrent 37–43 % av
registrert skalar tapsverdi; gradientandel er ikke målt. Dette er et faktisk
målprioriteringsskille, ikke bevis for at vekten alene forårsaket svikten.
I h480-sideparets eksakte FIT-skala (111,040 bps) er konstant minimizer for
Huber-leddet på de senere 11 318 komplette banene +4,683 bps LONG−SHORT,
mens aritmetisk gjennomsnittlig kostet kvalitet er −5,303 bps. Beslutningen
bruker prognosen som forventet kvalitet. Dette er en matematisk mål-/verdi-
mismatch på denne fordelingen: et perfekt robust konstantestimat kan velge
motsatt side av gjennomsnittlig økonomisk verdi. FIT hadde både positivt
Huber-sentrum og positivt gjennomsnitt; mismatchen alene forklarer ikke
modellens senere +44,292 bps predikerte sideforskjell eller nær-null rangering.

Et eldre, separat LightGBM-kontrollforsøk på lukkede indikatorøyeblikksbilder
hadde M5 S/F-AUC 0,788/0,769 på FIT, men 0,510/0,509 senere. Det var et
annet view og ingen native sekvensmodell; det viser at høy FIT-score i en
enkel kontroll heller ikke dokumenterer overføring. Denne EMA-kjøringen
lagret kun initial/sluttprognoser på senere rader og kronologiske FIT-
tapsblokker, ikke sluttmodellens FIT-prognoser. Derfor er det fortsatt
ubevist om dagens modell lærer sidebetingelsen på FIT og så mister den,
eller ikke lærer den der. En ny epoke eller liten tapsjustering avklares
ikke av denne evidensen.

Neste falsifiserbare, direkte kontroll er én separat innrømmet, kapslet og
oppdateringsfri scoring av den fryste EMA-checkpointen på eksakt 29 574 FIT-
rader, sammenlignet med allerede lagret senere score på h480 og S/F.
Ingen slik plan er laget eller innrømmet. Fram til FIT-vs-senere er skilt,
forblir trening, Full GC, TEST, ONLINE, broker og handel stengt.

## Fryst FIT-scoring klargjort 18.11.2026

En separat, oppdateringsfri scorer er lagt til for den eksakte fryste
EMA-sluttmodellen og de 29 574 Fold1 FIT-radene. Den binder 31 kvalifiserte
inputreferanser, konsumert checkpoint, mål-/featureakser, kildefiler og en
engangsport med global trening fortsatt stengt. Scoreren rapporterer
forhåndsvalgt komplett h480-sidepanel og S/F-score mot FIT-prioren; et
uavhengig kontrollgrunnlag lagres. Kapslet CUDA-produsentrute er eksakt
avgrenset til denne planen, 20G/512M og maksimalt 3 600 sekunder. To
scorertester og 14 rutetester, syntaks og diffkontroll består. Ingen
modellscoring eller trening er startet; plan og innrømmelse følger separat.

Plan `TA_M5_EMA_FINAL_FIT_SCORE_20261118_001` er nå klargjort mot
kildecommit `b0eecca7` og 31 inputreferanser; policy står
`PREPARED_NOT_ADMITTED` med null tillatte kall. Neste steg er eksakt
preflight og kontroll av kapsling; dette er ingen treningsåpning.

Eksakt kapslet preflight bestod uten CLAIM, modellforward eller resultatrot.
Denne separate scoringsporten er deretter innrømmet én gang, med
`training_enabled=false`, null optimizeroppdateringer og fortsatt stengt
TEST/Full GC/ONLINE/broker/handel. Kilde og policy fryses under kjøringen.

## Fryst FIT-sluttmodell kontrollert 18.11.2026

Engangsscoringen fullførte på 2 380 sekunder med 1 849 forwards, null
optimizeroppdateringer/backward/normaliseringsfit og uendret modell-digest.
Resultat, checkpoint, policy ved CLAIM, mål-/featureakser og lagret capture
er bundet; uavhengig beregning fra capture og fryste mål gjenskapte h480 og
M5 S/F-tallene. Planen er konsumert og stengt.

På 18 901 komplette FIT-h480-baner hadde sideforskjellen LONG−SHORT
AUC 0,570 og korrelasjon 0,143, men modellen foretrakk LONG på 99,38 %.
Valgt-side kvalitet var −18,165 bps mot alltid-LONG −18,189 bps. På
11 318 senere komplette baner var AUC 0,473, korrelasjon 0,007 og
valgt-side kvalitet −25,566 bps mot alltid-LONG −25,463 bps. Alle er
overlappende markout, ikke posisjonsført PnL eller uavhengig TEST.

Enda viktigere for aktiv fase: M5 native S/F retnings-AUC på FIT var
0,528/0,517; på senere utvikling 0,512/0,509. FIT Brier for betinget
M5-side var S 0,25108 mot FIT-prior 0,24965 og F 0,25307 mot 0,24988.
Av alle fem TF hadde S ingen bedre betinget side-Brier enn FIT-prioren,
og F bare D1 svakt bedre (0,21993 mot 0,22214). Senere var F dårligere
enn prioren på alle fem TF. Sluttmodellen har altså svak S/F-læring allerede
på eget FIT-utvalg; dette er ikke bare en senere overføringssvikt. Den
aktive fasens markedstilstand var kun hjelpetap 0,10 i denne treningen.

En ny, oppdateringsfri lesning av de samme hashbundne FIT/senere-capturene
viser systematisk OPP-skjevhet på kjente native F-sider. H4 hadde
predikert/faktisk OPP-andel 0,697/0,567 på FIT (n=10 134) og
0,712/0,543 senere (n=6 053); retnings-AUC falt 0,542→0,472.
D1 hadde 0,735/0,667 på FIT (n=8 926) og 0,737/0,567 senere
(n=6 063); AUC falt 0,602→0,505. Også M5 lå over faktisk andel
på både FIT (0,556/0,511) og senere (0,562/0,513). Dette er
betinget OPP-sannsynlighet på kjente sider, ikke andel av alle rader.
En ren terskelendring kan ikke reparere svak rangering. FIT-kalibrering
svikter før senere regimeskifte; materialet isolerer ikke om årsaken er
representasjon, mål, optimizer eller samspillet i trening. Ingen ny fit
eller parameterendring følger av dette alene.

Et konkret neste kontrollspørsmål er om eksisterende fullverdig S/F- og
struktur-tap som eneste mål kan slå FIT-prioren uten å skade senere score
på de kvalifiserte EMA-inputene. Gjennomgå tidligere tilsvarende fit først;
ingen ny treningsplan er laget eller innrømmet. Huber-verdiavviket for
Entry-kvalitet står åpent for senere, eget middelverdikonsistent mål.
Full GC, TEST, ONLINE, broker og handel forblir stengt.

## Hvorfor ny epoke eller S/F-omvekting ikke er neste steg

Nærmeste tidligere native S/F-hovedmål på 317/255-visningen fullførte 2 959
oppdateringer, men på fast TRAIN4096 steg M5 F-side-Brier fra 0,25045 til
0,25521, mot FIT-prior 0,24988. S/F-side-Brier var verre enn prioren på alle
ti S/F×TF-celler. Dette panelet er ikke hele det eldre FIT-utvalget. En eldre
topass-kontroll på samme 29 574 foreldre gjorde senere M5 F-side-Brier
0,2573 etter første pass og 0,2587 etter andre, mot prior 0,2497; den brukte
et annet view. Batchrekkefølgetest påvirket sannsynlighetsnivå, men ga ikke
målt bedre retningsrangering. Fallende samlet treningstap over forskjellige
kronologiske batcher motsier ikke disse faste, dårlige retningstallene.

På den aktuelle EMA-visningen er både betinget M5 S- og F-log-tap på fullt
FIT også dårligere enn prioren. Masker, UP/DOWN-koding, akser og native
query samsvarer eksakt i FIT og senere capture. Dermed er det ingen grunn
til å gjenta det samme tapet, stokke batcher eller legge på epoker som
«rettelse». Tidligere gradientgjennomgang fant ingen bred S/F-konflikt ved
M5-grensen, men målte ikke den nye beslutningshovedveiens effekt. Neste
minste falsifiserbare måling er en oppdateringsfri gradientpartisjon på
forhåndsvalgte FIT-batcher av den fryste EMA-modellen: om kostet primærtap
motvirker native F-side ved den delte representasjonen. En ny fit er ikke
innrømmet. Økonomisk edge er fortsatt ubevist.

Kilde for den nye gradientproben er nå klargjort: 20 faste hele B16-batcher
spredt over Fold1 FIT; 21 forwards (ett ekstra paritetskall), 80 autograd-
kall kun til løsrevet delt TF-representasjon, null parametergradient,
optimizer eller oppdatering. En eksakt 1 800-sekunders kapslet rute er
lagt til. Fire probetester og 12 rutetester består. Plan og opptak følger;
ingen modelljobb er startet.

Plan `TA_M5_EMA_LOSS_GRADIENT_PROBE_20261118_001` er klargjort mot
kildecommit `57e7eca8` og de samme 31 kvalifiserte inputreferansene.
Policy er `PREPARED_NOT_ADMITTED`: null tillatte kall, trening fortsatt
stengt. Preflight og deretter eventuell eksakt engangsinnrømmelse gjenstår.

Gradientprobens eksakte kapslede preflight bestod uten CLAIM, forward eller
resultatrot. Kun denne separate 20-batch-proben er innrømmet én gang.
`training_enabled=false`, null optimizeroppdateringer og stengt TEST/Full
GC/ONLINE/broker/handel består. Kilde og policy fryses under kjøringen.

## EMA-gradientpartisjon fullført 18.11.2026

Den eksakt innrømmede proben fullførte 20 faste FIT-batcher på 69,6 sekunder:
21 forwards, 80 autograd-kall kun ved løsrevet delt TF-representasjon, null
parametergradienter, optimizer eller oppdateringer. Modell-digest var uendret,
forward-paritet og tapspartisjon bestod; 19 batcher hadde til sammen 117
kjente native M5 F-sider. Resultat/CLAIM/TERMINAL er hashbundet. Uavhengig
summering av de lagrede batchverdiene gjenskapte dot, kosinus, norm og
støttetelling.

Det forhåndsbundne rene fortegnsvilkåret slo ut: primært beslutningstap hadde
negativt dot-produkt med isolert F-side i 13 av 19 støttede batcher og i
summen. Størrelsen er imidlertid nær null: samlet kosinus −0,00235.
Beslutningsgradientens norm var 5,39 ganger den isolerte F-sidegradienten,
men nesten ortogonal. Hele faktisk tap hadde positiv F-side-kosinus +0,117,
og S/F-hjelpetapet +0,173. Dette viser en svak lokal fortegnskonflikt,
**ikke** en materiell gradientmotstand som forklarer svak F-læring. Proben
måler bare grensen til fused TF-representasjonen på 20 batcher, ikke hele
upstream-parameteroppdateringen eller historisk optimizerforløp. Ingen ny
fit, tapsomvekting eller epoke følger av dette resultatet.

Arbeidsblokkeringen er nå konkret: full FIT-retning svikter FIT-prior;
senere kostjustert sidevalg og S/F-generaliserering svikter; gammel S/F-
primærtrening og ekstra epoke hjalp ikke, og gjeldende tapskonflikt er
for liten på målt grense til å gi en begrunnet vektjustering. En neste
modellfit krever en **distinkt** forhåndsdefinert informasjons- eller
målkontrakt med fast FIT/senere kontroll, ikke flere like runder. Huber-
utgangen for Entry kan fremdeles ikke kalles forventet økonomisk verdi.
Full GC, TEST, ONLINE, broker og handel forblir stengt.

## Ny, avgrenset informasjonskontroll klargjort 18.11.2026

S/F-fasiten er fortsatt relevant som framtidsorakel: på h480-komplette
baner med kjent F-H4-side var oracle-AUC mot kostjustert beste side
0,926/0,919 på FIT/senere (7 592/4 888 rader). Fryst modell-F-H4
rangerte den faktiske kostsiden bare 0,530/0,497 på alle komplette baner.
En ren fortegnkontroll ga senere valgt kvalitet −24,452 mot alltid-LONG
−25,463 bps på 11 318 baner, men forskjellen +1,011 bps har ukeblokk-
intervall [−0,033,+2,378], bruker bare 225 SHORT og gjenbrukt utvikling.
Dette er ikke Edge. Eksakte target-/captureakser ble kontrollregnet.

Kildegjennomgangen fant én konkret manglende relasjon: gjeldende M5-pris
sammenlignes ikke eksplisitt med siste kausalt lukkede H4-trendlinje
projisert til samme M5-tid. En egen, lesebasert forskningsmåling ble
implementert og kjørt én gang uten modellkall. Den bruker bare komplette
48×M5 H4-barer,
sammenhengende M5-klokke, to aktive cachelinjer og ett forhåndsbundet
M5-sluttprisbrudd. Fraksjonell H4-linjeprojeksjon er en **ny** konvensjon;
integer-linjefeltene støttes av tidligere fysisk H4-orakel. Preflight
sjekker kilde, klokke og linjearitmetikk før F-/h480-utfall åpnes.
Fem modul- og 11 rutetester, syntaks og diffkontroll består. Målingen er
nå fullført én gang; ingen modelltrening er startet.

For Huber-mismatchen er framtidig middelverdikonsistent Entry-mål spesifisert
i `docs/NATIVE_LEARNING.md`. Den konsumerte v2-koden er ikke mutert:
aktiv fase er S/F før Entry, og ny Entry-fit er ikke begrunnet.

Plan `TA_M5_PROJECTED_H4_LINE_20261118_001` ble kvalifisert mot
kildecommit `ab03b7c4`, 31 EMA-referanser og fysisk H4-orakel. Kapslet
preflight bestod på cirka 15 sekunder uten CLAIM eller utmappe. Én
innrømmet lesemåling fullførte med exit 0, uendret kilde, null
modellforward/optimizer og ingen TEST. CLAIM, RESULT, TERMINAL og lagret
raduttrekk er hashkontrollert; mål og kvalitet er kontrollregnet fra
raduttrekket uavhengig av rapporten. Engangskallet er konsumert.

På 2 081 FIT- og 1 196 senere prisbrudd var kjent F-H4-retning bare
51,0 % riktig på 723 FIT-rader og 49,1 % på 491 senere rader. For
komplette h480-baner var hendelsesvalgt kvalitet mot alltid-LONG
−0,771 bps på 1 186 FIT-rader og +11,408 bps på 717 senere rader.
Ukeblokk-bootstrap med 20 000 trekk (frø 20261118) ga 95 %
intervall [−9,047, +7,623] og [−9,898, +34,945] bps. Seneregevinst
kommer fra gjenbrukt utvikling og varierer mellom uker/måneder; den
beviser verken stabil prediksjon, handels-PnL eller Edge. Dette er
M5-sluttprisbrudd, ikke intrabarberøring. 23 FIT-rader ligger utenfor
fysisk H4-orakel, og hendelsen dekker bare en liten del av alle rader.

Denne H4-linjeproxyen gir ikke grunnlag for S/F-refit eller en ny epoke.
Neste mulige fit trenger en **annen**, forhåndsbundet informasjons- eller
målkontrakt, kvalifiserte komplette senere inputs og én samlet voktet plan.
Entry-Q v3 er fortsatt bare framtidig design. `training_enabled=false`;
Full GC, TEST, ONLINE, broker og handel er stengt.

## Eksakt S/F-hovedmål-kontroll forberedt 18.11.2026

Den konsumerte EMA-modellens rapporterte senere initial- og sluttmåling
bruker samme 17 489 rader. Betinget side-Brier ble dårligere etter
trening i alle ti native S/F×TF-celler; F-H4 gikk 0,249780→0,286405,
F-D1 0,250652→0,275951. Original RESULT/TERMINAL, checkpoint og
sluttcapture er hashkontrollert. I oktober forbedret den trente modellen
F-H4/D1, men i november/desember ble både kalibrering og H4-rangering
svakere. Sluttmodellens F-D1 OPP-andel lå også langt over faktisk andel
i FIT-mai; dette er ikke bare et senere regimeskifte.

Dette gir ett nytt, direkte kontrollspørsmål: Kan S/F som eneste hovedmål
på **eksakt samme** EMA-input og friske startvekter slå fast FIT-prior
på hele FIT, og bære forbedringen til full senere utvikling? Den eldre
S/F-hovedtreningen brukte et annet view/panel; M5-gradientproben dekker
ikke alle TF eller parameteroppdateringer. Et negativt FIT-svar avviser
målprioritet som tilstrekkelig rettelse; FIT-gevinst uten senere gevinst
peker på overføring/regime. Én kontrollpassering, ikke epokejakt.

Ny kilde og eksakt 10 800-sekunders kapslet rute er klargjort. De binder
29 574 Fold1 FIT-rader, 17 489 senere rader, 1 849 B16-oppdateringer,
full FIT/senere S/F-skår og F-H4-fortegn som fast markout-proxy mot
alltid-LONG. Samme initiale EMA-wrapper-hash er rekonstruert på CPU uten
forward; core-hash er `b73dd0c23c0e62fb20bcb79bbb7a87dd84794580ebdcea814559fab508f63826`.
Sju nye fokuserte tester, syntaks og kapselkontroll bestod. Plan
`TA_M5_EMA_SF_ONLY_CONTROL_20261118_001` var bundet til kildecommit
`3ff90f57`, SHA `9a0a54b087d1089869a513492bee7b5cc4d3974d5ffa862dbb25b3217e2980b4`
og 31 kvalifiserte inputreferanser. Kapslet preflight bestod uten CLAIM,
resultatmappe, modellforward eller oppdatering og gjenskapte samme core-hash.

Fysisk Windows-boot 666 oppfylte fersk-boot-vakten. Umiddelbart etter
omstart var siste lokale Git-commit, to blobber og ett tre null byte.
Arbeidsfiler og indeks matchet eksakte Git-hasher; blobene og treet ble
gjenskapt fra uendrede filer og indeks. Det eksakte treet fikk gyldig
commit `a2ee53d6` med forrige gyldige commit som forelder.
Ødelagte nullobjekter og nulltegn i to reflogger ble fjernet. Git-kjedens
tilkobling, ren arbeidskopi og innrømmet kilde-/boot-vakt bestod på nytt;
commitene ble pushet før trening. Ingen modellkilde eller planinnhold ble
endret i reparasjonen.

Den eksakt innrømmede S/F-kontrollen fullførte på boot 666 med exit 0:
1 849/1 849 oppdateringer, 2 943 sluttscore-forwards, 7 254 sekunder og
uendret kilde. CLAIM, RESULT, TERMINAL, begge radcapturer og checkpoint
er hashbundet. Gjeldende trening, Full GC, TEST, ONLINE, broker og handel
er stengt, og engangskallet er konsumert. Checkpoint-digest, gammel
policy ved CLAIM, FIT/senere-akser, alle ti S/F proper scores og h480-
kvalitet er kontrollregnet uavhengig fra lagrede rader og faste mål.

**Hovedkontrollen feilet allerede på FIT.** Native F-M5 betinget side-Brier
var 0,26266 mot fast FIT-prior 0,24988 på 11 909 kjente FIT-sider, og
0,26578 mot 0,24983 på 7 058 senere sider. Begge var også verre enn
beslutningsmodellens 0,25307/0,25428. Alle ti native S/F×TF-celler var
verre enn FIT-prior på både full FIT og full senere utvikling; log-tap
for F-M5 var også verre. Retnings-AUC F-M5 var 0,520/0,512 FIT/senere,
F-H4 0,553/0,486. Målprioriteten alene var altså ingen tilstrekkelig
rettelse av denne modellens S/F-læring. Å legge til epoker på samme
kontrakt er ikke begrunnet av dette resultatet.

Forhåndsbundet F-H4-fortegn som sideproxy på komplette h480-baner ga
+1,149 bps mot alltid-LONG på FIT (18 901 rader) og +1,853 bps senere
(11 318). ISO-ukeblokk-bootstrap med 20 000 trekk, frø 20261118, ga
95 % intervall [−1,062, +3,239] og [+0,153, +3,693] bps. Senere var
586 SHORT-valg, hvorav 488 i oktober; kontinuerlig kostside-AUC var
0,503, og 6 171 senere h480-baner var ufullstendige. Dette er en liten,
forhåndsvalgt utviklingsobservasjon på overlappende markout, ikke
posisjonsført PnL, uavhengig generalisering eller Edge. Den kan ikke
brukes til å åpne TEST eller handel.

Neste fit trenger en faktisk ny, kausal informasjons- eller målkontrakt,
full senere inputkvalifikasjon og ny engangsinnrømmelse. Den konsumerte
Entry-v2-kvalitetsutgangen er fortsatt ikke en forventningsverdi;
middelverdikonsistent Entry-v3 er spesifisert som senere design, ikke
den aktive S/F-fasens treningsmål. Ingen ny plan er innrømmet.

En samlet, oppdateringsfri dekomponering av de samme FIT/senere-capturene
skiller hva kjernen faktisk lærer: aktivitet-Brier slo fast FIT-prior i
8/10 native S/F×TF-celler på både FIT og senere; betinget side-Brier i
0/10. Struktur-high/low hadde AUC 0,808/0,799 på FIT og 0,802/0,794
senere. Dette er spesifikk svikt i retningssiden, ikke generell manglende
modellforward, dataflyt eller optimizeroppdatering.

På full FIT var F-M5 gjennomsnittlig predikert OPP-sannsynlighet 0,091
over faktisk andel, F-H4 0,203 og F-D1 0,161. Den eksisterende
side-BCE-en har kjente F-sider i 1 844/1 849 M5-batcher, 1 143 H4-
og 705 D1-batcher. F-H4 og F-D1 har egne lineære sidebiaser; regnet fra
sluttcapturen er gjennomsnittlig direkte tapgradient på disse biasene
+0,0253 og +0,0140 per FIT-batch etter faktisk TF-divisor. En lokal
senkning av bias ville derfor redusert fast FIT-side-BCE. Modellen
avsluttet ikke ved et full-FIT-stasjonært sidepunkt. Få kjente H4/D1-
batcher kan bidra til oppgavenes ubalanse, men M5 er kjent i nesten alle
batcher og feiler også. Dette isolerer ikke om årsaken er én passering,
samspill mellom oppgaver, representasjon eller målstøy. En ren biasretting
ville uansett ikke bedre AUC eller bevise senere Edge; flere epoker er
ikke innrømmet av dette funnet.

## EMA 318/256: målrettet input-, fasit- og modellrevisjon 18.11.2026

Tre avgrensede, parallelle leserevisjoner og én oppdateringsfri analyse av
allerede lagrede rader fant ingen konkret kode-, klokke-, maske- eller
skåringsfeil som forklarer S/F-sidekollapsen. Ingen modellforward,
optimizersteg, TEST-lesning eller ny innrømmelse ble gjort.

Det nye EMA50/200-spreadendringsfeltet kobles på alle fem TF etter fysisk
rad-/tidparitet. Tilgjengelighet krever barstart + TF-varighet ≤ M5-issue;
M1-kontekstens fem barer er også lukket, og strukturfasit går bare til tap.
FIT-only asinh-normalisering er invertibel uten klipping. Det normaliserte
M5-feltet har standardavvik 0,623 på FIT og 0,630 senere, nullandel 0;
modellens EMA-projeksjonskolonne har ikke-null vekt. Dette beviser input-
kontakt, ikke output-sensitivitet eller prediktiv verdi.

En direkte, ujustert rangering av **kun det nye råfeltet** mot kjent native
F-side, med samme lukkede-bar-kobling og de hashbundne FIT/senere-capturene,
ga følgende OPP-AUC (FIT/senere): lokal/M5 0,4937/0,4915
(n=11 909/7 058), M15 0,4905/0,4937, H1 0,4914/0,5169,
H4 0,4677/0,5249 og D1 0,3452/0,4086. D1 har bare 109/65 ulike
lukkede barverdier på disse overlappende issueradene. Ingen fortegn,
terskel eller modell er tilpasset fra senere data. M5/M15 viser ingen tydelig
standalone retningsrangering fra tilleggsfeltet; dette utelukker ikke
ikke-lineært samspill med øvrige indikatorer og er ikke Edge.

Samtlige 29 574 FIT- og 17 489 senere rader matchet de bundne
allowed/eligible/mature-fasitene for sampled S, native S og native F i
alle fem TF; alle ti native side-Brier-tall og FIT-priorer matchet RESULT.
Alle ti dedikerte S/F×TF-sidehoder avvek i sluttcheckpointet fra
fellesradene de ble klonet fra ved init (vekt-L2 0,045–0,123). Fellesradenes
output overstyres før tapet, så dette er sterk kilde- og checkpoint-evidens
for oppdaterte sidehoder, ikke en direkte initial→final tensorserie.
Sideprediksjonene er likevel ikke kalibrert godt nok. Backbone
har dropout 0,05, ingen BatchNorm; ingen lagret train-mode sluttscore eller
per-oppgave gradientbane isolerer dropout, klipping eller oppgavekonkurranse
som årsak. Ingen sidehead-ruting eller optimizerfeil er bevist.

S og F har ulike aktivitetsbegreper: S beskriver lokal range, mens F
bruker terminal nettoendring. På felles informative M5-rader var
S-AKTIV/F-FLAT_NET 52,0 % på FIT og 53,0 % senere. De har egne hoder,
så dette er et målt semantisk målskille, ikke en feilmerket side. Kjente
M5 S/F-sider var enige i 97,1 % begge perioder. M5 aktivitet-Brier-gevinsten
mot FIT-prior var liten: S 0,00297/0,00255 og F 0,00143/0,00091
(FIT/senere). For F-H4 var fasitsiden enig med kostet beste h480-side i
92,7/92,0 % av kjente, komplette baner, men dette er et delutvalg.

Revisjonen avgrenser feilen til manglende læring/kalibrering av betinget
framtidsretning på allerede kjent FIT, ikke bortfall av det nye EMA-feltet
eller en feil i radfasit/skår. Den isolerer ikke om gjenværende årsak er
svakt kausalt retningssignal, delt representasjon, én kronologisk passering,
klipping eller måloppgavenes samspill. Ingen kildeendring eller ny fit er
begrunnet. Konsumert Entry-v2 forblir en robust proxy, ikke forventet
økonomisk verdi. Den senere Entry-v3-kontrakten er nå matematisk
bundet i `docs/NATIVE_LEARNING.md` til direkte Q=netto−pre-MFE-MAE på samme
komplette banemaske, kvadratisk tap og identisk Q ved valg; ingen kjørbar
v3-modell eller ny fit er åpnet. Policy og alle lukkede porter består.
