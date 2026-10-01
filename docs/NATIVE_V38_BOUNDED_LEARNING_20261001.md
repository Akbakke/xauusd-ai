# Native v38: inputbygg og avgrenset læringsbevis — 01.10.2026

Målet er å måle den faktiske delte Entry/Exit-modellen med alle 254 v38-felt,
alle åtte spesialister og eksisterende M5/M1-/MTF-klokker. D1-makrotesten er
fullført og inkonklusiv; den åpner ingen makroinnføring.

## Faktisk læringsmål og hva som må måles

Gjeldende Entry-kontrakt er Bellman-steget inn i første autoritative post-fill
Exit-tilstand. LONG/SHORT får verdien fra den frosne TRAIN-tilpassede Exit-læreren;
FLAT har nullverdi. Den samme modellen og delte encodere eier Exit.
Dette er et estimat under den deklarerte policyen, ikke observerte prisutfall.
Den eksisterende referansepolicyen og økonomikontrakten må bindes i recipe
og rapporteres med eget navn; referanseverdi og optimalitetsverdi må ikke blandes.

En ny v38-måling skal derfor skille:
1. Tilpasning på eksakt TRAIN-mål/populasjon mot fersk initialbaseline og
   konstant tilpasset på samme TRAIN.
2. Senere kronologisk generalisering på frosne rader, uten refitting av
   normalisering, lærer eller terskler fra kontrollen.
3. Observerte kostnadsjusterte markedsutfall fra den lærte Entry/Exit-policyen,
   med alle valgte handler og åpne posisjoner. Separat porteføljeregnskap og
   kapital-/overlappskontroll kreves før en økonomisk botpåstand.

Den eksisterende native prefix-eieren har et avgrenset budsjett på
256 optimizersteg / høyst 4096 Entry-rader, batch 16, seed fra den bundne
initialiseringseieren og ingen full epoch. Dette er en tilgjengelig
kontraktmekanisme, ikke en lansert eller ferdig v38-recipe.
Ingen nye tapsvekter, handelsgrenser, cutoff-søk eller maksimal holdetid innføres.
Et eventuelt før/etter-resultat gjelder samlet v38-læring; det isolerer ikke
automatisk den kausale effekten av de 12 nye feltene fra resten av modellen.

## Konkrete inputavvik og minste nødvendige bygg

Det fullførte v37-datasettet har 242 signalverdier; gjeldende eier krever 254.
M1/M5-featureflater, signalmanifest, datasett, normalisering og initialbaseline
kan derfor ikke gjenbrukes som om de var v38.

Rådataparet er kontrollert med fulle SHA256-hasher og Parquet-footere.
Alle registrerte produsentfiler er identiske, bortsett fra fjerning av en
ubrukt HTF-hjelper; resten av hele modulen har identisk AST.
Den faktiske squeeze-eieren godtar alle seks klokkers frosne parametre,
og squeeze-produsentene er byte-identiske. Disse to uendrede avhengighetene
gjenbrukes. Ingen ny rådatabygging eller squeeze-refitting er nødvendig.

Forhåndsregistrering:
configs/research/NATIVE_V38_INPUT_PREPARATION_20261001.json.
Kildebevis:
 /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INPUT_DEPENDENCY_REVIEW.json.

Bygget bruker eksisterende scripts/run_seq513_rebuild_chain_v1.sh,
samme tidligere vedtatte TRAIN/VAL/TEST-vinduer, samme kalibreringsvinduer,
nye immutable utdata og eksisterende capped-vakter for hvert tungt trinn.
Native trening, normaliseringsfit og optimizersteg åpnes ikke her.
TEST kan bare konstrueres og forsegles av den etablerte byggeieren;
ingen inspeksjon av TEST-utfall, modellbruk, måling eller tuning.

## Nåstatus og kontroll ved overtakelse

Inputbygg er fullført 01.10 kl. 16:00 UTC fra kilde 5bdd75e0; den separate
readiness-eieren ble ferdig kl. 16:22 UTC. Ingen prosess fra byggingen eller
etterkontrollen lever. Originale outputs, kvitteringer og kilde er bevart.

- TRAIN: 652 552 Entry-rader; utviklings-VAL: 70 880.
- Signalbredde 254 og alle åtte familier. Alle 12 nye sweep-/AVWAP-felt er
  endelige på hver Entry-rad og varierer i begge åpne splits.
- Seks readiness-porter består: terminal, preflight, input-liveness,
  pretrain-audit, fysiske TRAIN/VAL-bindinger og metadata-bundet TEST-forsegling.
- Etterkontrollen har ikke lest, hashet eller stat-et TEST-datasett/manifest.
  TEST har null disclosures. Ingen modellkjøring, normaliseringsfit eller trening.
- En separat sluttkontroll bekrefter aktuelle kontrakteiere, ordnede felt,
  Entry-/lifecycle-footere, samme datasetthasher og liveness-populasjonene.
  De store Entry-filene ble fullhashkontrollert av readiness-eieren, ikke
  duplisert av sluttkontrollen.

Runtime: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001.
COMPLETION_REVIEW.json SHA256:
d8f6c847b6978846151de2ee7647594f06d07cb188469743f9c25cdb775a1139.
Datasett-terminal SHA256:
2ce054b1553d64c06f4011d987d3cb41f2bdbb19ccaa3dd5ceae5ae9ce3b55eb.
Eksakte readiness-/signal-/split-/TEST-seal-stier og hasher finnes i
sluttkontrollen og native_v38_preparation_20261001 i begge status-JSON-ene.
Byggegodkjenningen er brukt og stengt; aldri relanser denne kjøringen.

## Fryst læringsdesign og målte kompatibilitetsfeil

configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json er en byte-identisk
kopi av runtime LEARNING_DESIGN_001/DESIGN.json, SHA256
5235db5c9e6ea619ec971673d58a38dc462bcf325122dc5dff0c6d4f8584a864.
Design og kontroll-ID-er skal ikke endres etter utforskning av kontrollutfall.

Fysisk TRAIN er 01.06.2011–31.05.2025; senere fysisk VAL/kontroll er
01.06.2025–30.06.2026. Kontrollens 256 rader er valgt med eksisterende
deterministiske selector, seed 20260911/salt 1, fra samtlige 70 880 VAL-rader.
De ligger i separate fysiske radkoordinater, ikke som ommerkede TRAIN-rader.
Kun tidskolonnen ble lest ved frysingen. Kontroll er gjenbrukt utvikling.

Budsjettet er én fersk modell, 256 optimizersteg, batch 16 og høyst 4096
TRAIN-Entries, ingen full epoch. TRAIN256 fryses fra den faktiske planlagte
TRAIN4096 etter kontroll av mål- og tilstandsstøtte før cutoff. Normalisering
skal tilpasses hele den eksplisitt kvalifiserte fysiske TRAIN-populasjonen
før sampling, aldri bare de 4096 radene eller fysisk VAL/TEST.

Entry-målet er første kjørbare likvidasjonsverdi pluss
reference_policy_state_values fra samme trace: 119/120 ganger gyldig HOLD
Q_mu, med FLAT=0 og bevarte negative verdier. Historisk max(observert verdi,0)
er ikke gjeldende target. Exit bruker samme kausale kostnader, referansepolicy
og frosne grenseverdi; 120 observerte backup-steg er ingen maksimal holdetid.
ONLINE og TARGET skal starte med samme aktuelle funksjon og vekter.

Faktisk kilde og bundne v38-manifester avdekket:
- Native full-TRAIN-eieren krever fortsatt 2021–2026. Den eksakte eksisterende
  vakten ble eksekvert på det nye manifestet og avviste det.
- Historisk prefix-sti krever samme TRAIN-fil også for kontroll. Andre gamle
  eiere krever 5508 VAL-rader. Ny fysisk VAL har 70 880 rader.
- Pilotens Entry-admission krevde pretest_test_guard. V38 har i stedet en
  ekte, completion-bundet TEST-forsegling. Dette er nå rettet via eksisterende
  metadata-validator, bundet til fryst design og fullført readiness.
- M1-piloten forventer et eget pre-TEST-manifest med komplette quote-kolonner;
  v38 binder native-pair M1-proveniens. Ingen konstruerte legacy-felt eller
  påstått quote-kompletthet kan erstatte faktisk kvalifisering.
- Historisk frozen-prefix-lærer fjerner ONLINEs parameterfrie slutt-normer.
  Denne historiske funksjonen skal bevares for gamle forsøk, men ikke brukes
  som fersk v38 TARGET når designet krever samme aktuelle funksjon.

Første minste rettelse er implementert i eksisterende
prepare_unified_exit_lifecycle_v2_pilot_v1._entry_window_scope:
et eksplisitt hash-bundet design styrer begge periodene. Fysiske manifest- og
parquet-bindinger, run-ID, deklarerte vinduer, radantall og klokkehash må stemme.
Standard og historisk full-TRAIN-sti beholder opprinnelige endepunkter.
Samme eier kontrollerte nå alle 652 552/70 880 ekte tidsstempler; 15 fokuserte
syntetiske tester består, inkludert feil kilde/rolle, overlap, naiv klokke,
endret radantall/klokke/hash og historisk oppførsel. Ingen prisutfall ble lest.

Bevis under runtime LEARNING_DESIGN_001:
- COMPATIBILITY_REVIEW.json: reprodusert opprinnelig avvisning og eksakte
  CONTROL256-ID-er.
- CALENDAR_ADMISSION_REVIEW.json: ekte kalender/populasjon består den nye
  produksjonseieren. Ingen full pilot-admission eller native launch hevdes.
- CODE_REVIEW.json: eksakt endret kode, 15 beståtte tester og syntakskontroll.

## Forslag fra operatøren: én M1-råkilde for alle klokker

Operatøren spurte 01.10 om én M1-kilde kan danne M5 Entry og M15/H1/H4/D1.
Kildeforløpet er nå kontrollert: MTF-eieren bygger allerede M15/H1/H4/D1
fra M5, mens M1 og M5 kommer fra separate native kilder. En felles autoritativ
M1-råkilde kan dermed forenkle rådataproveniensen. Kildeskiftet er ikke utført.

En avgrenset mekanikkmåling på hele TRAIN-året 2024 brukte faktiske,
hash-kontrollerte native årsfiler og eksisterende multi_tf_resample-eier.
354 866 M1-rader ga 71 208 M5-barer, nøyaktig samme klokker som native M5.
Alle 13 markedsfelt var eksakt like på hver bar: mid/bid/ask OHLC og aktivitet.
70 809 barer hadde fem observerte M1-rader; 399 hadde én til fire. Også disse
399 samsvarte eksakt. Ingen minutter ble fylt inn, ingen barer ble lagt til.
Målingen er rådatamekanikk, ikke mål-/modell-/lønnsomhetsevidens.

Rapport:
 /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/M1_SINGLE_SOURCE_2024_AUDIT/RESULT.json
SHA256: 37704fd5fc91e66fa67499eb96f3efb74db616f046600b4ebca4e2045c910ebb.
Plan, kildehasher og original operator er bevart ved siden av resultatet.
Kun 2024 TRAIN ble åpnet; TEST og trening er urørt. Året alene beviser ikke
full historikk-, feature-, normaliserings- eller train/serve-paritet.

Neste steg etter denne årskontrollen var å kvalifisere resten av historikken.
Den fullførte kontrollen med samme bargrenser og gapregler står nedenfor.
Bruk den komplette M1-råkilden med mid/bid/ask; den feature-/warmup-trimmede
base28-visningen er ikke hele råhistorikken. Bygg rå OHLC og aktivitet per
tidsramme før de eksisterende feature-eierne beregner indikatorer. Aggreger
aldri ferdige M1-features. Entry forblir M5 og Exit M1, med delte MTF-cacher.

Ferdige v38-inputs og tidligere evidens bevares. En senere kildeendring må
deklareres i eksisterende kildekontrakter; avledet M5 skal ikke påstås å være
en separat native OANDA-M5-kilde. Det er ikke nødvendig å bygge alt på nytt
uten først å måle hvilke eksisterende bytes som faktisk endres.

## Hele før-TEST-historikken kontrollert; eksisterende M1 kan gjenbrukes

Den etterfølgende fullkontrollen er ferdig. Perioden 01.06.2009–30.06.2026
har 5 959 045 M1-rader og 1 215 514 M5-barer. Alle klokker og alle 13 markedsfelt
er eksakt like mellom aggregerte M1 og opprinnelige M5. Ingen manglende eller
ekstra barer. 2024-resultatet ble gjenbrukt; de andre 17 årene/delårene ble
kontrollert med eksisterende M5-resampler og en uavhengig heltalls-grid/
numpy-reduceat-beregning.

Fysiske pre-TEST-parentfiler ble brukt. Deres hasher er bundet til eksisterende
successor-metadata; alle årshasher til og med 2025 er identiske. Ingen
successor-2026-data med TEST-rader eller TEST-datasett/manifest ble åpnet.
Dette er rå markedsverdier, ikke full feature-/normaliserings-/modellparitet.

Den eksisterende filen
HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source/m1_quotes.parquet
er nå kvalifisert av require_unified_exit_pretest_m1_quote_authority på samtlige
5 959 045 rader mot den opprinnelige M1-kilden. Den har komplett mid/bid/ask
og aktivitet, og fysisk slutt før TEST. Ingen ny fil eller featurebygning
trengs for å skaffe dette rågrunnlaget. Kildeskifte i den aktive pipeline er
ikke utført; tidligere v38-manifester skal aldri omskrives til ny proveniens.

Aggregert rapport: docs/NATIVE_M1_SOURCE_PARITY_20261001.json.
Detaljer ligger i runtime M1_SINGLE_SOURCE_PRETEST_AUDIT/RESULT.json og
EXISTING_M1_AUTHORITY_REVIEW.json. Originale operatører, kildehasher og
årsmålinger er bevart. Ferdige sammenligninger skal ikke gjentas.

## TEST-seal-kompatibilitet rettet og målt

Eksisterende pilot-eier bruker nå den eksplisitte test_guard_event-bindingen
i recipe når designet er bundet. Den krever samme fullførte readiness,
alle seks grønne bevispunkter, eksakte TRAIN/VAL-filer og samme seal-hendelse.
Den faktiske forseglingen valideres av require_prefreeze_test_seal_lineage.
Den historiske eksplisitte pre-TEST-stien beholder sin opprinnelige oppførsel.

Under review ble rekkefølgen strammet: en oppgitt seal-peker sammenlignes med
den frosne readiness før generisk filkontroll. En feilpeker til TEST-parquet
skal aldri åpnes eller stat-es bare for å bli avvist etterpå. En ny
regresjonstest beviser dette. 21 fokuserte tester består; syntaks og diff er
kontrollert. Ekte v38-metadata består den nye produksjonseieren med
Path.open/stat/resolve sperret for begge TEST-artefaktene: null forsøk.
Dette beviser den lokale adgangsporten, ikke full native kjørbarhet.

Runtime SEAL_ADMISSION_PATCH/REAL_SEAL_FINAL_REVIEW.json og CODE_REVIEW.json
binder aktuell kilde og bevis. Den første før-korrigering-reviewen er bevart
som historie; siste ferdige review er autoriteten.

## Komplett M1-kilde bundet i faktisk klargjøring

Den nye eksplisitte recipe-stien bruker den tidligere kvalifiserte M1-filen.
Eksisterende lifecycle-kontrakteier gjenbruker sin hash-bundne kvittering og
rehash-kontrollerer M1-filen, quote-manifestet, pair-lineage og native manifest.
Den kostbare rad-for-rad-kvalifiseringen gjentas ikke. Eieren for native
autoritet er gjenbrukt også for metadata-validering; ingen falsk trenings-
eller legacy-evidens konstrueres.

Pilot-eieren krever samme originale M1-autoritet i begge frosne Entry-splits.
Den fullhash-kontrollerte successor-metadataen må navngi nøyaktig den kvalifiserte
pre-TEST-parenten, inkludert kilde-/manifest-/kanoniske radhasher, produsent,
radantall og tidsgrense. Originale v38-manifester og featurefiler er uendret.
Ny proveniens og at fysiske M1-radnumre ikke kan gjenbrukes bindes i pilot-hashen.

144 tester i de tre berørte testfilene består, inklusive historisk oppførsel,
endrede kildebytes, feil parent, splittet autoritet, feil tidsgrense og manglende
design/kvalifisering. Dette er syntetiske mekanikktester. En separat ekte
klargjøring rehash-kontrollerte alle nødvendige TRAIN/VAL- og M1-inputs og
kontrollerte alle 652 552/70 880 tidsstempler. Path.open/stat/resolve-sperrene
registrerte null tilgangsforsøk til TEST-artefakter, gammel filtrert råfil
eller successor-rådata med TEST-rader. Normer/modell/optimizer er ikke kjørt.

Runtime M1_REBINDING_PATCH/OUTPUT:
- PREPARATION_RECIPE_20261001T173806836787Z.json:
  b64e7b2641a767bc680239d9f2d6b450515690b2c8148cd01430320a4142ae06.
- PREPARATION_READINESS_20261001T173846817860Z.json:
  d70543ef9013378550c0b9f6500f374dd0fe6ed9370aa07b2b7627bd1c06f720.
- SOURCE_REBINDING_REVIEW_20261001T173846858521Z.json:
  f668abc946f0870a8af9b77f389ed2f29a66d6e7929c538d72754a68f8e585c4.

Kvitteringene er publisert gjennom eksisterende immutable event-eier.
Denne første klargjøringsrapporten var BLOCKED ved entry_window_adoption:
den beviste innledende kilde-/kalenderadgang. De etterfølgende komponentene
og den nyere readiness-grensen er dokumentert nedenfor. M5s produksjonskilde er fortsatt
den opprinnelige; rådataparitet åpner ikke en udeklarert ommerking.

## Fryste perioder og fysiske M1-koordinater ført videre

Entry-child-ruten med eksplisitt kildebytte adopterer bare de verifiserte
Entry-bytene. Den gjenåpner ikke den gamle native-pair M1-kilden eller adopterer
gamle M1-tilstander. Frosset readiness/seal, opprinnelig lifecycle-root,
TRAIN-/VAL-path og hash, original M1-autoritet og den nye parent-proveniensen
må fortsatt stemme. Historisk full-v1-admission er bevart for gamle oppskrifter.

Alle fysiske Entry-rader inngår i de vedtatte vinduene. Child-manifestene peker
derfor direkte på de samme ferdige parquet-filene, med samme SHA256 og samtlige
features. 21 160 180 085 bytes blir gjenbrukt uten kopi eller omskriving.
Dette er ingen ny beregning eller ommerking av de opprinnelige v38-manifestene.

Én felles kalenderkontroll i eksisterende child-eier binder de faktiske
TRAIN-/kontrollperiodene, separate fysiske kilder, radantall og klokker til
det uendrede fryste designet. Den brukes av M1-view, compact-lifecycle,
summary-fit og normaliseringsadmission. Den gamle TRAIN-slutten 01.06.2026 og
VAL-forventningen 5 508 beholdes bare i den historiske ruten. Nye eksplisitte
forventninger som motsier designet avvises; ingen stille radutvelgelse skjer.
Publisering av Entry-/M1-views bruker fsync og eksisterende no-replace-eier;
admission-kvitteringen er også fsync-kontrollert før publisering.

Den faktiske klargjøringen er fullført under audit-cgroup 4 GiB/512 MiB swap:
- TRAIN-view: 4 884 638 M1-rader, hvorav 474 kontekstrader og 4 884 164 rader
  i fit-vinduet. Siste rad er 30.05.2025 kl. 20:59 UTC, før TRAIN-cutoff.
- Kontroll-view: 382 744 M1-rader, hvorav 478 kontekstrader og 382 266 rader
  i kontrollvinduet. Siste rad er 30.06.2026 kl. 23:59 UTC.
- Alle 652 552/70 880 Entry-rader finner eksakt første M1-rad ved Entry+5 min.
  Ingen manglende første tilstander. Dette beviser ikke senere tilstandsstøtte.
- En separat Arrow-sammenligning mot riktig fysisk parent-slice bekrefter
  nøyaktig likhet i alle tids-, mid-, bid-, ask- og aktivitetsfelt.
- Null tilgangsforsøk til TEST-artefakter eller gammel råkilde med TEST-rader.
  Ingen modeller, normaliseringsfit eller optimizersteg ble kjørt.

62 fokuserte tester besto den sammenhengende kilde-/view-/kalenderrettelsen.
Ytterligere seks tester og ekte metadata-/byte-admission bekrefter at
normaliseringsklargjøringen bruker 652 552/70 880 fra designet i den nye ruten;
gamle eksplisitte 65 295/5 508 avvises. Dette er admission, ikke full
normaliseringspopulasjon, beregnede normer eller læring.

Resultater under CHILD_COORDINATE_PREPARATION_001/EVENTS:
- RESULT_20261001T175552486447Z.json:
  90a9c712c141476b756de4f05163c5023441d58fbe1ac7834d8559c638b80610.
- READINESS_20261001T175552467344Z.json:
  7af4baa39e81abf1f10a95eb5d159743d8dc76e15ad47cde785478f147189669.
- NORMALIZATION_ADMISSION_REVIEW_20261001T175753519716Z.json:
  d82ef0d98b14e6d49e7b0b7b95a5c4df8baa65904448f9528d30f2bf2aaefe48.

Ferdige Entry-metadata/admission og M1-views ligger under
LEARNING_PREPARATION_001; eksakte stier/hashes finnes i resultatet og statusfilene.
Ikke relanser produsentene. Den nye readiness er fortsatt BLOCKED ved
train_economics, etter at Entry-adoption og child-admission har bestått.

## TRAIN-tilpassede markedspauser og M1-støtte kontrollert

Den eldre pausepolicyen var tilpasset til juni 2026 og omfattet dermed deler
av den nye kontrollperioden. Den er ikke gjenbrukt. Samme deklarerte krav på
12 observasjoner for daglige pauser og helger er tilpasset 4 884 164 faktiske
TRAIN-tidsstempler før 01.06.2025, uten de 474 kontekstradene. Ingen terskelsøk,
prisutfall eller kontrolltilpasning. Resultatet er 34 gjentakende signaturer;
dette er en prosjektutledet policy, ikke en verifisert offisiell markedskalender.

På alle 652 552 TRAIN-/70 880 kontroll-Entries er eksakt første tilstand,
klassifisering av alle gap og antall observerbare overganger kontrollert
uavhengig. TRAIN har 2471 kjente pauser og 63 806 ukjente gap; kontroll har
263 og 49. Ukjente gap sensurerer forløpet. Ingen Entry mangler en observert
etterfølger. 30 872 TRAIN- og 119 kontroll-Entries har færre enn de deklarerte
120 backup-overgangene. Dette er ikke automatisk ugyldige mål: kontrakten
kan bevare gyldig lærer-bootstrap ved en observasjonsgrense, og 120 er ingen
maksimal holdetid. Alle 256 fryste kontrollpunkter har minst 339 overganger.

Eksakte første tilstander, høyresensurgrenser og overgangstellinger er lagret
som immutable NPY-artefakter. Gjenbruk dem. Kjent pause tillater observasjon
etter gjenåpning; kostnader over faktisk veggklokketid må fortsatt følge
økonomieieren. Klokkestøtte beviser ikke featuredekning eller komplette mål.

En gjenværende TRAIN_END-default i normaliseringspopulasjonens eksisterende
bygger er rettet til den fryste kalenderens sluttdato. Eksplisitt avvik avvises
før kilde-I/O. Den gamle defaulten beholdes bare for den historiske ruten.
Dette var en konkret kontraktmismatch, ikke målt lekkasje fra kontrollutfall.
Pauseeieren bruker eksisterende atomisk no-replace-publisering med fsync.
13 fokuserte tester består, inkludert faktiske fil-/katalogkollisjoner; ingen
normalisering, modellforward eller optimizersteg er kjørt.

Ekte audit er ferdig under capped 4 GiB/512 MiB swap. Alle gapavgjørelser og
fit-støttetellinger har uavhengig kontroll, og overgangstellingene matcher en
separat to-peker-orakelberegning. TEST og gamle blandede rå-/featurekilder
fikk null tilgangsforsøk. Senere featurekildegjennomgang leste bare metadata.

Under runtime CLOSURE_STATE_SUPPORT_001/EVENTS:
- RESULT_20261001T180740533994Z.json:
  bf02567ac6a5b2a89651fd9cd0af7d23db93b214281dab5e159662c15baf12a3.
- FINAL_CODE_AND_METADATA_REVIEW_20261001T181339812965Z.json:
  bf89f8a23461ab3c88d7f27d234577c2a570df752e53af0bb3190a5f4d15ed66.

Policy og split-schedules/closure-authorities ligger under
LEARNING_PREPARATION_001. Resultatet binder alle stier, hasher og arrayfiler.
Ingen produsent skal relanseres. Readiness ved train_economics er fortsatt
uløst; denne klokkekontrollen er ingen økonomisk eller native launch-autoritet.

## Neste konkrete grense

Kilde-, kalender- og TEST-seal-portene, Entry-child og M1-views er kontrollert.
Gjenbruk den bundne recipe, de publiserte komponentene og siste readiness.
Markedslukking/ukjente gap og observerbar tilstandsstøtte er nå kontrollert
som beskrevet over. Neste steg er faktisk M1-/MTF-featuredekning gjennom
fysisk før-TEST-input og eksisterende eiere. Eksisterende M1-featureflate er
bundet til gammel filtrert base28, mens ny admission krever komplett M1.
Originale manifester skal ikke ommerkes, og vaktens kildekrav skal ikke svekkes.
Det berikede M1-manifestet binder ekte native M1, men kilden går inn i
TEST-perioden. Dagens featureeier fullhasher den kilden og leser hele
tidskolonnen; å bytte bare alignment er derfor ikke en før-TEST-begrenset
gjenbruksrute. Checkpoint-metadata inneholder bare 14 kontekstfelt, ikke en
dokumentert komplett erstatning. Kvalifiser trygg gjenbruk eller minste
nødvendige før-TEST-rekonstruksjon før ny materialisering startes.
Feature- eller berikede databytes er ikke åpnet i denne metadatakontrollen.
Ingen gamle base28-radnumre eller v37-normer skal føres videre som om de gjaldt
nye kilder og 254 felt. Native indeks/trening/måling har fortsatt gamle
femårskrav og TRAIN/VAL-koordinatantakelser som må rettes gjennom eksplisitt scope.
Frossen TARGET må følge samme aktuelle funksjon som ONLINE.

Initial/final TRAIN og senere kontroll må måles mot samme-TRAIN-konstant,
med sentrert feil, tilstandsavhengig variasjon og retnings-/handlingsfordeling.
For hver forhåndsdeklarert Entry/Exit-side må forbedringen også bestå den
låste parvise uke-bootstrapen mot både initialisering og TRAIN-konstant.
Manglende/inkonklusiv evidens åpner ingen utvidelse eller retuning.

Native initial-/sluttmåling har economic_rollout=false. Kostnadsjustert
porteføljeøkonomi med alle valgte handler, åpne posisjoner og én posisjons
kapasitet må bindes separat mot FLAT og forhåndsvalgt samme-risiko LONG.
AVWAP-vekt er prisoppdateringsaktivitet, ikke omsatt volum.
Normalisering og native trening er fortsatt stengt til faktiske bindinger
er ferdige. Senere uavhengig generalisering, train/serve-paritet, full B og
offline driftskvalifisering er fortsatt ufullført. Ingen edge er dokumentert.
