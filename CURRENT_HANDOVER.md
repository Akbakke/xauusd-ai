# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`.
`GX1_RULES.md` er bindende. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus;
prosess og eksakte terminalkvitteringer overstyrer prosa. Tidligere bølger finnes i Git.

## Gjeldende status

`PAUSED_M5_COLD_FIT_CONSUMED_EDGE_UNPROVEN`. Ingen GX1-jobb kjører. Den ene kalde
M5-fit-planen er fullført og konsumert. `training_enabled`,
`native_training_enabled`, `full_epoch_training_allowed` og
`native_launch_authorized` er `false`. Full GC, TEST, ONLINE, broker og
handel er stengt. Ingen økonomisk edge eller uavhengig generalisering er bevist.

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
backup av VHDX får ikke plass på C:. Operatøren valgte å vente på backupplass
og lukkede terminaler. Ingen `wsl --shutdown`, DiskPart eller VHDX-
komprimering er utført. `NEXT_RUN_POLICY.json` binder denne beslutningen.

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

## Ikke gjort og neste direkte steg

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
