# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md`, kjør `bash scripts/gx1_handover.sh --check`, og les bare
aktuelle status-/scopefelt i `NEXT_RUN_POLICY.json`. Policy eier arbeidsstatus;
faktiske prosesser og terminaler overstyrer prosa. Én integrator og én capped
jobb; fryst kilde/policy under kjøring. Konsumerte planer relanseres aldri.

## Gjeldende mål og tidsrammer

Entry tar beslutninger på lukkede **M5**-barer, med lukket M15/H1/H4/D1.
M1-data brukes nå bare som finere etterfølgende prisfasit. Prognose-/
observasjonshorisontene 1–480 minutter er ikke Entry-tidsrammen, Exit-logikk
eller en fast holdetid. **M1-Exit tilhører neste bølge etter Entry-edge.**
108s512M5-proberader hadde beste prognosescore ved1minutt for begge sidekandidater;
diagnostiske handlinger var116SHORT/396FLAT/0LONG, ingen native handelsautoritet.
Reelle141/144-vitner har300sekunders Entry-barer og korrekt lukket tilgjengelighet.

Lær LONG/SHORT/FLAT fra kausale inputs. Retningsretur, signert MFE og
konservativ MAE før første MFE har separate framtidsmasker; dager er støtte.
Kvalitet og dekning gjelder både valgte og avståtte muligheter. Alltid-FLAT
er ingen suksess, og framtidsfasit/masker brukes aldri som beslutningsinputs.

Første profil: `compact_mbp_smc_ema_atr_clock_macro_entry_v1`, SHA
2ac4fb4ddfac84f588f65363ac8959544233ace286d9648bfebf60b97153f125.
MBP-1, SMC/struktur, EMA/ATR og nødvendig klokke/makro beholdes i kompaktprofilen;
utvalgte candle/chart-/overlappende felt er parkert. Fullbiblioteket med åtte
familier og niende D1/makrokontekst bevares. Ingen felt er erklært støy uten måling.
Separate modeller/stacking og M30 krever kronologisk OOF-merverdi.

TRAIN [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingsdata [2026-04-01,2026-07-01). Alle26måneder er gjenbrukt utvikling;
TEST fra juli er forseglet. `training_enabled=false`, fullVAL/HPO/Exit/broker/
live/paper/ordre/spending/promotion er stengt. De to practice-leseoppslagene er konsumert.

## Siste native inputpåvirkningsmåling —144

144 er COMPLETE/PASS, exit0,160,399sekunder.229kilder/policy ble bevart.
Faktisk218forwards/13input-VJP/3445gyldige counterfactual-responser ble kjørt
på115optimizer0ONLINE, tre genuine TRAIN-vitner og13uscorede kopier.
480framtidige minutter, alle9outputnøkler og B16-bitparitet bestod.
Vekter, normalisering, parametergrads, RNG, moduser og sessionfiler ble bevart;
0optimizersteg/refits, ingen Exit/TEST eller ny epoke.

Alle3445gyldige inngrep endret kandidatmarginen og alle fem komponentmaksima.
Alle3407numeriske/alias-ruter hadde minstett observert currentvitne og
ikke-null VJP i hver av13projeksjoner der observerte. Ingen observert rute var
helt uten respons. Dette er funksjonell sensitivitet på tre vitner, ikke
lært indikatornytte, Entry-kvalitet, generalisering eller edge. Finite deltaer
er maksima over side/horisont, ikke bevis på at alle480punkter/begge sider endret seg.

Én D1-session_change_flag er kildekonstant og ble korrekt ikke intervenert.
Ti GC-H1-aliaser mangler observert historikk blant vitnene; de er ikke bevist
inaktive.96snapshotruter har ingen historikkakse.786numeriske og2515alias-
historieruter hadde ikke-null historikkmaksimum. Alle tre diagnosevalg er FLAT:
begge predikerte sidekvaliteter er under analytisk FLAT0, med unikt argmax.
LONGs beste prognosehorisont er480min, SHORTs1min på disse tre optimizer0-vitnene;
ingen etterfølgende utfall eller vellykket avståelse er bevist av det.

Bevisroot: `/home/andre2/GX1_RUNS/NATIVE_M5_DYNAMIC_COMPACT_INPUT_INFLUENCE_20261021_144`.
- `NATIVE_INPUT_AUDIT_TERMINAL.json`: dd6e6bfebf99e96035244e44ffe821d876387e8042320682a40c88cd1adc91e4
- `ACTUAL_REVIEW.json`:7665527f3944bb73fa07b2fae4a30b57564c45188af984f8cba016a9b92d6bff
- `ACTUAL_PEER_REVIEW.json`:acaab9c0bc00ec665d178972ed3e3b6cb59af8a27c3813b0bbd62e5e3d02db6e
- `ACTUAL_INFLUENCE_NUMERIC_REVIEW.json`:61dbac1477439406b681b8ddca11e80d41feb489c9d6664b8e547a7709d9b935
Original34,37MB-resultat og alle bindinger er bevart.144 er konsumert/stengt.

## Neste konkrete arbeid

Neste steg er operatørens beslutning om det konkrete150-forslaget nedenfor.
Ingen trening er åpnet.

Fullkurvene er nå lagt til i eksisterende `_dynamic_m5_prefix_observations`
og gjenkoblet av eksisterende `_require_dynamic_m5_observation`. Hver entry
har fem1..480minuttkurver med egne masker, null bak uobserverte mål og separat
base-/dagsprognose. Original target-storage-SHA og alle gamle aggregater består.
Dette gir ingen ekstra finalforwards eller Exit-arbeid.

146 er COMPLETE/PASS:29fokuserte syntetiske cases,51,255sekunder, CPU4G/512M,
0nativeforwards/optimizersteg. Kilde/policy bevart. Hele faktiske trainclosure ble
målt én gang:183/183roller, ingen tilkomne/fjernede. Tre innholdsendrede roller
viser til to fysiske filer (observer og admission, wrapper er alias). Eksakt
nådigest27952ad6e1e669cd59d80c763ad0980a619e53725c06bef36b62d8791121b735
avviker fra115s5385ca4c7afdd8b1c779dfcd4406b26f004372faad05ba388c266466b48676b1.
Ingen compatibility-exception er lagt til. 115 har dessuten ikke fullkurvesporene.
Dette er kilde-/rapporteringskvalifisering, ikke ny faktisk baseline eller læring.

Root: `/home/andre2/GX1_RUNS/NATIVE_M5_FULL_CURVE_OBSERVATION_SOURCE_QUALIFICATION_20261021_146`.
- `TERMINAL.json`:09ebe0b623e9e741039f838a9be317f88a4dc17c772c6d1031f050865d355fe5
- `SOURCE_BINDING_COMPARISON.json`:bbf5d7ba1141beaf6bf9de941ec909226f56622525d26c38c833db3ff7b95371
- `ACTUAL_REVIEW.json`:552452c26912c9f40ca06835950036dbb97c6204ea4fc7d22b19a41b183df10a
- `ACTUAL_PEER_REVIEW.json`:5ac13d3b5ec76980a6b2f5935812558c1fbc2e1d2c48d94ee1672e7b16e4a900

148 har nå fullført den nye nullstegs initialmålingen med dagens eksakte kilder
og fulle TRAIN256-/CONTROL256-kurver. Gjenbruk ferdige inputs, targets,
wholeTRAIN-normalisering og TRAIN-RMS; ikke les rå MBP på nytt. Den nye
baseline-/campaignbindingen skal følge eksisterende eiere og kildekrav.
147 er COMPLETE/PASS,120,032s: eksisterende CPUfresh-publisering, audit4G/512M,
0forwards/updates/refits.230kilder/policy ble bevart, faktisk nåbundet183rolle-kjede.
Planroot: `/home/andre2/GX1_RUNS/NATIVE_M5_FULL_CURVE_FRESH_PREPARATION_20261021_147`.
Den konstruerte ny utrent tilstand gjennom kanonisk eier;114 var template,
ikke checkpoint-adopsjon eller relansering. ONLINE/target/EMA er alle
9c4cf514cbfd5b6f9366f69065abcc0878ef274117e2b5c7e703656e86a79b4a.
Initialstate-SHA940283e6132a0339a1d3dd165d74cbdd64527927f5fdb7f1bc95ffe1c32e7232.
Nyeinitialization/measurementrefs står i147RESULT; kilde-/IO-review er
`ACTUAL_FRESH_IO_REVIEW.json`, SHAd076bfad9ae574bd2471efd118095a4c49dcee914679567ce377144c4b47e92b.
Målt canonicalproducer-IO:49.271.677.365rchar/45.800.828.928read_bytes over111,846s,
kun worker+ventede barn, ikke maskintotal. Ferdige fysiske inputs ble verifisert;
ingen rå MBP-rebuild eller refit. Uavhengig metadata-peer bestod:
`ACTUAL_PEER_REVIEW.json`, SHAde93e8236bd344dda4cb5f789d8a9b21a6c9feb280f9d3a7acb94302b406dbc6.
148 er nå terminal og konsumert: native guard/trainer/observer PASS/0/0,
TRAIN256+CONTROL256,32Entry-forwards og0optimizersteg. Måling1780,868sekunder;
komponentoppsett731,366sekunder er en egen rapporteringsrolle. Hele jobben
stoppet som planlagt ved optimizer-ceiling0 (`RESUMABLE`); dette er ikke
adgang til å fortsette eller en ferdigtrent modell. Windows-oppgaven er Disabled/Last0.
FysiskBoot519 er bekreftet21.10T12:04:44.500Z. Alle183kildefiler og13bundne
inputidentiteter samt policy/rules ble bevart gjennom kjøringen.
Vaktens målte topper var55Ccore/62Cmemory/131,7Wdraw/738MiB, fysisk effektgrense160W.
Første tidlige signerte etterboot-forespørsel timet ut før controlleren klargjorde
nettbroen; originalfeilen er bevart. Senere signert35/42C/160W bestod.
Root: `/home/andre2/GX1_RUNS/NATIVE_M5_FULL_CURVE_INITIAL_20261021_148`.
- `ACTUAL_NATIVE_INITIAL_TERMINAL_REVIEW.json`:170ff4311a895815d38e52b92fa3b5edce14b2789fb40ec4d540123211425a87
- initial_measurement/RESULT.json:a392890405aa9d645f4ed00d8e27bb31282757284194fb1b7d62c3c8d1cbf412
- native receipt:c249f781129e73322c74e19ffb2a189f703c74dc54124fa4c2de10c097504730
- `REBOOT_CONFIRMED.json`:f8273cdea6e00319a6014ef79fc2c794ae2a185c971a7bcbc5642f980bf1978b
- `WINDOWS_TASK_DISABLED.json`:1ead3da8687948a728bd010a15a7de5f181abcf28805e6f596b693a59d1282ab

149 er COMPLETE/PASS,67,848sekunder totalt: CPUaudit4G/512M/900s med
0modellkall/updates.147fresh og148initial0checkpoint er eksakt like for modell,
target, optimizer, EMA, scheduler og sju lagrede CPU/Python/NumPy-RNG-felt.
CUDA-RNG finnes i nativecheckpointet, men sammenlignes ikke mot CPUinitialiseringen.
Eksisterende `require_chronological_learning_measurement` har faktisk bestått
med dagens full480-validator for begge256-raderskohorter. Hele kurvene og egne
masker er dermed kvalifisert innen denne kohorten, ikke hele TRAIN-populasjonen.
Root: `/home/andre2/GX1_RUNS/NATIVE_M5_FULL_CURVE_INITIAL_STATE_AUDIT_20261021_149`.
- `TERMINAL.json`:aecce73f2c98924c96d360d7e2b8b1241c1c27dfbc0c215337bb1e662000cdd9
- `INITIAL_MEASUREMENT_AUDIT.json`:b88aab0e0758d277b25a1d382060d8380e151d4cefa4bb631ef2494b61fcc573
- `STATE_PRESERVATION_PROOF.json`:2b1f61eeec8fbb9f171fd8bd3c71f7196bf529a5c2a281c759bce92a72daafbf
- `ACTUAL_AUDIT_PEER_REVIEW.json`:425eb8d1ec4db1d7aa9cfd7a0163155b12b897f1f8bc6a200d7b78620109c344
117s modell/loss/optimizer/paritetsmekanikk er uendret; bare to observerfunksjoner
i to filer er endret.149/UNCHANGED_LEARNING_MECHANICS_REVIEW.json,
SHA659a93fe87ea756d0b9e78ef0a689c562df714b10503682efaf0b9bb0b3a5b45,
kvalifiserer gjenbruk av det uendrede mekanikkomfanget. Ingen ny4-updateprobe.

150 er konkret forslag til én ny111TRAIN-RMS-epoke, ikke en kjøreplan:
118249TRAINrader én gang, batch16/slutt9,7391updates, maks12000s/13800s ytre,
32faste final ONLINE-forwards på samme TRAIN256/CONTROL256 og fullkurver.
Start147fresh med kvalifisert148/149baseline; ingen108/117treningsvekter.
Root: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_RMS_EPOCH_PROPOSAL_20261021_150`.
PROPOSAL-SHAd65b07a4fa12d6b3812170f8fa81a3718a97e8871924348c76e403f2e4cc2cf8.
Dette avløser145s foreldede reviewforslag. M5beslutninger; M1prisfasit1–480min,
M1Exit først senere. Gjenbruk ferdige inputs/targets/055normalisering/111RMS;
ingen råscan, refit, HPO, fullVAL, Exit, TEST eller broker.

Neste steg er operatørens nye eksplisitte beslutning om150.108konsumerte den
opprinnelig bestilte epoken. Før ny eventuell launch må eksakt ny policy/recipe/
campaign og ferske native maskinvare-/bootporter bindes gjennom eksisterende eiere.
Ingen gammel plan relanseres. Alle execution/training/full_epoch/fullVAL/
Exit/TEST/brokerporter er false. Godkjent baseline er ikke læring eller edge.

## Bestått — gjenbruk uten omkjøring

- 121–125:alle1205aktiveGCverdier på publisertM1→femTF kvalifisert.
126:1205verdier+1205masker bit-eksakt til faktiskM5-context,328119090celler.
TRAIN dekker1205/1205; VAL1181/1205.24felt er korrekt manglende iVAL.
- 127/128/129/133:alle139aktive corecachefelt perTF numerisk kvalifisert,
695aktiveoutputs.133s790-outputindeks omfatter95parkerteoutputs fra eldre
kvalifisering; fullbibliotekets øvrige parkertefelt er utenfor kompaktkontrollen.
- 134:66råMBP-filer/892100712records og sluttmaskjoin er uavhengigPASS;
20,039GBgzip/118,260GBCSV ble lest/parset én gang hver. Sluttjoin157felt/
1139040rader/178829280celler bestod. Rådata skal ikke leses på nytt for dette.
- 139/142:65aktuelle fokuserte sourcecases bestod;142s ytrejobb feilet kun på
JUnit-telling161rapportunits mot65testnoder. Originaler bevares, ingen omkjøring.
- 140/141/143:tre maskvitner dekker alle2525currentaliaser; ferdig rå inputpakke
1.280.070byte, ekte96historierader og15lukkedeTF-vinduer;3445gyldige cases/én
kildekonstant. Inputpakke/caseplan gjenbrukes, ingen ny råscan/rendering.
- 055:wholeTRAIN-inputnormalisering.111:TRAIN-only loss-RMS i egen rolle;
inputnormalisering ble ikke refittet av lossendringen.
- 114fresh/115initial/116state-RNG/117Entry-only fire private updates og fire
paritetsforwards med eksaktresume er bestått.117vekter er ikke maintrain-start.
- 108:den ene bestilte hele epoken er FULLFØRT,118249rader/7391updates/
6090,176s. Alle seks pooledTRAINfeil falt, men1minuttfeil ble dårligere på
begge diagnoser; på512M5-proberader toppet begge sidekandidater ved1minutt.
Diagnostiske valg116SHORT/396FLAT/0LONG. Delvis læring,
ikke kvalifisert Entry-kvalitet.111RMS-endringens læringsnytte er ubevist.

## Maskinvare og resterende grenser

144 brukte eksisterende CUDA-producer20G/512M, én numerisk tråd,CPU0–7,
64tasks og samme signerte bridge/vakter:65Ccore/80Cmemory/160Wgrense/
170Wdraw/12288MiB. Førlaunch ble vedvarende keeper satt300→160W med
eksisterende installer uten nedlasting; bare power_limit_w i konfig endret seg.
Signert førclaim:33C/40C/160W, guard viste trygg oppstart; dette er ingen påstand
om målte peaktemperaturer. Profiletterkontroll og originalconfig/logg er bevart.
Fysisk18.10-omstart er bekreftet, nyeste boot21.10T12:04:44.500Z. Ikke gjenta
eldre omstartsbestilling. Fremtidig nativejobb følger egne faktiske hardwareporter.

Aktivt mål er ufullført: beslutningssporbarhet etter læring og TRAIN-RMS-læring,
LONG/SHORT/FLAT-kvalitet/dekning for valgte og avståtte, generalisering og
økonomisk edge gjenstår hver for seg. Ingen observasjon eller tekniskPASS
åpner TEST, handel eller ny trening. Originaldata/kvitteringer/checkpoints
bevares; opprydding går bare via retention-eieren.
