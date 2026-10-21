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
108s «alle512valgte1minutt» gjelder prognosehorisonten, ikke M1-entry.
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

## Siste faktiske resultat —144

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

145 er bare et fryst, ikke eksekverbart forslag til én ny TRAIN-RMS-epoke:
118249TRAINrader, batch16/slutt9,7391updates, maks én12000s invokasjon og
13800s ytre tak. Det foreslår114fresh/optimizer0-start;108/117vekter gjenbrukes ikke.
`PROPOSAL_M5_CLARIFIED.json`, SHA
4e31258e1a44e3b7db0e83d10cf381e7f9a53972b373754f24f340bfba12c7d4, under
`/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_RMS_EPOCH_PROPOSAL_20261021_145`.
Alle execution-/training-porter er false; ingen fit/campaign følger av forslaget.

Målt diagnosegap: eksisterende `_dynamic_m5_prefix_observations` har480
prediksjoner/targets/masker i RAM, men lagrer bare koordinataggregater og hver
sides modellvalgte horisont. Persistér de allerede materialiserte fulle
per-entry-kurvene og egne masker i samme finalobserver, uten ekstra finalforwards.
115 er faktisk paret aggregate/selected-horizon-baseline, ikke full480×512-baseline.
En endring må få eksakt korrekt kilde-/baselinebinding. Hele ferske trainclosure
mot115 er ennå ikke målt; ingen mismatch er påstått og ingen gate omgås.
Nye fullkurve-baselinespor er ny etterspurt evidens, ikke relansering av115.
Deretter kreves ny eksplisitt beslutning om ekstra epoke og eksakt campaign/
maskinvareadmission. Gjør dette konkrete arbeidet før treningsspørsmålet stilles.
Ingen jobb kjører nå. Ingen gammel plan skal relanseres.

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
begge diagnoser; alle512valgte1minuttprognose og ingenLONG. Delvis læring,
ikke kvalifisert Entry-kvalitet.111RMS-endringens læringsnytte er ubevist.

## Maskinvare og resterende grenser

144 brukte eksisterende CUDA-producer20G/512M, én numerisk tråd,CPU0–7,
64tasks og samme signerte bridge/vakter:65Ccore/80Cmemory/160Wgrense/
170Wdraw/12288MiB. Førlaunch ble vedvarende keeper satt300→160W med
eksisterende installer uten nedlasting; bare power_limit_w i konfig endret seg.
Signert førclaim:33C/40C/160W, guard viste trygg oppstart; dette er ingen påstand
om målte peaktemperaturer. Profiletterkontroll og originalconfig/logg er bevart.
Fysisk18.10-omstart er bekreftet, nyere boot20.10T20:15:55.500Z. Ikke gjenta
eldre omstartsbestilling. Fremtidig nativejobb følger egne faktiske hardwareporter.

Aktivt mål er ufullført: full beslutningssporbarhet, TRAIN-RMS-læring,
LONG/SHORT/FLAT-kvalitet/dekning for valgte og avståtte, generalisering og
økonomisk edge gjenstår hver for seg. Ingen observasjon eller tekniskPASS
åpner TEST, handel eller ny trening. Originaldata/kvitteringer/checkpoints
bevares; opprydding går bare via retention-eieren.
