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
Markedslukking/ukjente gap og observerbar tilstandsstøtte er kontrollert.
Eksisterende M1-featureflate er bundet til gammel filtrert base28, mens ny
admission krever komplett M1. Originale manifester skal ikke ommerkes, og
vaktens kildekrav skal ikke svekkes. Metadatakontrollen over åpnet ingen
feature- eller berikede databytes; checkpoints med 14 kontekstfelt er ingen
dokumentert komplett erstatning for beriket M1.

Kildegjennomgangen viser at vi kan bruke den etablerte featurebyggeieren
direkte med komplett før-TEST-M1 som alignment og original beriket M1 som
uendret beregningskilde. Bare M1-featureflaten materialiseres på nytt.
Ingen ny Entry-bygging, M5/MTF-beregning, rangering eller parameterfit.
configs/research/NATIVE_V38_M1_REALIGNMENT_20261001.json og runtime
M1_FEATURE_REALIGNMENT_001/PLAN.json har identiske bytes, SHA256
0e65f67c5300be2bba1f2e0a15f21cb1519351ffd3a9f06f1eda7755bda6fc6f.
Dette er en separat deklarert inputreparasjon, ingen relansering av fullført bygg.

Lesesomfanget er konstruksjon gjennom den eksisterende eieren, ikke den
tidligere rene klokkerevisjonen: den delte berikede kilden omfatter juli/august
2026 og blir fullhashet og kausalt transformert internt. Vi hevder ikke null
lesing av rå inputbytes fra TEST-perioden i dette bygget. Forseglet TEST-datasett
og -manifest blokkeres før I/O; ingen targets, handelsutfall, modell, fitting
eller tuning brukes. Fryste registry-/squeeze-parametre gjenbrukes uendret.
Original pair-proveniens beholdes. Den nye flaten må bare ha tidsstempler fra
komplett M1 før 01.07.2026, etter faktisk kausal warmup, uten interne utelatelser.

Produsenten bruker eksisterende 10 GiB/512 MiB capped-grense, én jobb.
Alle utdatarader kontrolleres av eksisterende materializer; en separat
tidsstempelkontroll krever eksakt alignment-suffiks og dekning av samtlige
nye TRAIN-/kontroll-M1-visninger. Runtime START, RESULT og TERMINAL binder
faktisk tilstand. Kilden fryses mens jobben kjører; aldri start en kopi.
Bygget er fullført 01.10 kl. 19:55:27 UTC med exit-kode 0; kilde fa71854a
var uendret. Ikke relanser det. Ny output er
LEARNING_PREPARATION_001/M1_FEATURE_BASE/m1_feature_base.parquet:
5 523 147 rader, 06.09.2010 kl. 02:46 til 30.06.2026 kl. 23:59 UTC.
435 898 innledende råklokker ligger før faktisk kausal warmup; deretter
matcher samtlige featuretidsstempler komplett alignment uten interne hull.
Alle 4 884 638 TRAIN- og 382 744 kontroll-M1-rader har eksakt featuredekning.
Dette inkluderer kontekstrader og gir ingen tillatelse til å tilpasse på VAL.

Eksisterende eier har kontrollert alle utdata før publisering; separat
tidsstempelorakel og sluttkontroll bekrefter nye kilde-/klokkebindinger,
uendrede 254 ordnede felt og eksakt samme registry-/squeeze-parametre.
Ingen originale artefakter er endret. Ny Parquet SHA256:
fad9b4997b8fe3d7d228b1c11cd33747f226423affbf3dbca7585b06f0da7636.
Sidecar SHA256:
1a71ad3ae54f616c0165863bbf7144bb886ffb0ea014b76b6290a90111c9e4ef.
Under M1_FEATURE_REALIGNMENT_001/EVENTS:
- RESULT_20261001T195526063354Z.json:
  75c4bde0237f0153caf5a89cee0331108881233d9599930568ed5ce1b3ff445d.
- TERMINAL_20261001T195526079697Z.json:
  4051686456775223fed5e2ec005add99e63bb9666cc6cfa5daa18cf04e148b7a.
- FINAL_RESULT_REVIEW_20261001T202027193820Z.json:
  63c42b53f1c4de74e11edabc7167bab4f89cc4843c881723b82765b69e95173a.

Neste konkrete jobb er å binde denne flaten i eksisterende normaliserings-,
indeks- og måleiere. Materialisering/dekning er bestått; normalisering,
mål-/økonomiadgang og læring er fortsatt ufullført.

Operatørens nye ønske om sporadisk PC-omstart er utført etter terminal og
maskinfelles tomme prosjekt-/GPU-køer. Windows startet 01.10 kl. 20:22:26 UTC;
Linux boot-ID er ny. Kilde, kvitteringer og sidecar-hasher er bevart.
En separat capped 4 GiB-kontroll etter omstart bekrefter gyldig Parquet-footer,
5 523 147 rader og korrekt 254/71 feltbredde; ingen modellfit/forward.
HOST_RESTART_001/POST_RESTART_20261001T202913477249Z.json har SHA256
a41d3ac1bdaa82916c3c1212926a3d1fa2c0316a83a5d4065b050e15af4a249a.
Eksisterende gx1-3090-lan fungerer. Tailscale-tjenesten kjører, men den
opprinnelige SSH-ruten svarte fortsatt ikke; nettverkskonfigurasjonen er uendret.
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

## Full sekvenskontroll før normaliseringspopulasjon

Sekvenskontrollen for hele TRAIN er nå bundet som TRAIN_SEQUENCE_AUDIT_001;
ikke start en kopi hvis LAUNCH_CLAIM eller terminal allerede finnes.
Eksisterende eier skal sammenligne alle 652 552 sekvenser/snapshots med M5-flaten.
To konkrete publiseringsløp er rettet til atomisk no-replace med fsync.
13 fokuserte tester består, inkludert samtidige fil-/katalogkollisjoner.
Dette åpner ingen normaliseringsfit eller modelltrening.

Plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/TRAIN_SEQUENCE_AUDIT_001/PLAN.json
SHA256: 690347173cf65766e904a4a2b347288ddebe48e8e558972b38e42bcfba0a4687.
Kontrollen leser bare time/seq/snap fra TRAIN og fullhasher filen;
VAL- og TEST-datasett/manifester blokkeres før I/O. Den delte M5-featureflaten
fullhashes og valideres gjennom eksisterende eier, også med senere prisinputs.
Ingen utfallsanalyse, modellforwards, normaliseringsfit eller optimizersteg.
Kilden fryses under capped audit (4 GiB / 512 MiB swap); terminal kreves.

Full sekvenskontroll avsluttet 01.10 kl. 20:52:38 UTC med exit 0.
Hele TRAIN-sekvenskontrollen er fullført: alle 652 552 seq/snap-rader
matcher bundet M5-flate eksakt. Terminal exit 0, uendret kilde og null
VAL-/TEST-datasettilgang. Gjenbruk TRAIN_SEQUENCE_AUDIT_001; ikke relanser.
Normaliseringsforberedelsen gjenbruker nå dette beviset bare ved eksakt
fil-/hash-/populasjonsbinding; endrede bytes eller bevis avvises.
Kun M5-klokken lastes der signalverdier allerede er kontrollert.
19 fokuserte tester består. NORMALIZATION_POPULATION_001 er bundet for
hele fysisk TRAIN gjennom eksisterende eier, uten statistikkfit eller trening.

Audit SHA256: 559082adcc9726f9bd889e6d91254a8c900781ab80379988b22b64521fcc1a46.
Normaliseringspopulasjonsplan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NORMALIZATION_POPULATION_001/PLAN.json
SHA256: 140544c8dc247df62b8d6768ece6f87df17663468f4598a87d2d09381112e313.
Dette publiserer bare fysisk populasjon og featureidentitet. Eksisterende
TRAIN-geometri gjenbrukes som uavhengig differanse-/union-orakel. VAL-filen
fullhashes kun for identitet; ingen VAL-utfall dekodes. Delt M5-kilde/MTF
hashes, M5-klokke leses, og ny før-TEST-M1-flate kontrolleres for valgte
TRAIN-verdier. Null fit/optimizer/model-forward; TEST-datasett forblir sperret.

Normaliseringspopulasjonen avsluttet 01.10 kl. 21:11:30 UTC, exit 0,
kilde 3f2fb070 uendret. 19 fokuserte tester består. NORMALIZATION_POPULATION_001 er fullført med
exit 0: hele TRAINs 652 552 entryer gir 955 670 unike M5-kontekstrader og
3 995 148 unike observerbare M1-tilstander. M1-unionen matcher uavhengig
den tidligere kontrollerte geometrien eksakt. Ingen statistikk er tilpasset.
Gjenbruk de tre publiserte inputartefaktene; ikke relanser produsenten.
Neste steg er tilstandsindekser og binding av normalisering/mål med separate
TRAIN-/kontrollkoordinater. Native launch, læring og økonomi gjenstår.

Faktisk kjøring gjenbrukte den komplette parent-sekvenskontrollen;
full ny sekvensdekoding og en unødvendig ca. 1,17 GB M5-signalallokering
er fjernet fra normaliseringsforberedelsen. Dette er målt inputkonsistens,
ikke læringskvalitet eller lønnsomhet. Null TEST-tilgangsforsøk.

result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NORMALIZATION_POPULATION_001/EVENTS/RESULT_20261001T211130374293Z.json
SHA256: 2bb931ede90755b4b58cfc08578a1ab59c09d1c51cb93607cc41b660ce63b082.
terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NORMALIZATION_POPULATION_001/EVENTS/TERMINAL_20261001T211130398642Z.json
SHA256: 2ae8ffdb8b0089f2eeeabb3300aeb122e95ba58e88549d846247b8b66bf0e86d.
final_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NORMALIZATION_POPULATION_001/EVENTS/FINAL_METADATA_REVIEW_20261001T212637944404Z.json
SHA256: 20a201a5854974f3383eb001dc00067b33a83f9e5330fc1848c4fad1d3983776.
CHILD_NORMALIZATION_VIEW.json: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/NORMALIZATION_INPUTS/CHILD_NORMALIZATION_VIEW.json
SHA256: e7c11c81a22e181c530fed02f9c56cc04481030fc4fd69d99a1e57d1edc85391.
CHILD_TRAIN_SEQUENCE_RECONSTRUCTION_AUDIT.json: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/NORMALIZATION_INPUTS/CHILD_TRAIN_SEQUENCE_RECONSTRUCTION_AUDIT.json
SHA256: fbbf2b5b2fae6e88b58fa5c238e3c01e4c020ea1480143308b222da1ca6ba70e.
TRAIN_NORMALIZATION_POPULATION_WITNESS.json: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/NORMALIZATION_INPUTS/TRAIN_NORMALIZATION_POPULATION_WITNESS.json
SHA256: 451a9db012e96d274dcbb9fb2dd4c6d12ef6b15cf1341bd6b3396c2ddc5d5f86.

## Bundet full-TRAIN-basefit før native indekser

Normaliseringskoden er nå bundet til de faktiske full-TRAIN-artefaktene før
statistikkfit. Feil vitne, byttede filer, feil MTF-sti og endrede bytes avvises.
Den eksisterende diskbaserte M1-innleseren erstatter store RAM-kopier;
historikkunion beregnes per sammenhengende intervall i stedet for per rad.
25 fokuserte tester består, inkludert eksakt normparitet og navnekollisjon.
BASE_NORMALIZATION_FIT_001 er nå bundet til én CPU-fit av basestatistikk på
hele kvalifisert TRAIN, før sampling. Dette løser en forutsetning for indeksene.
Bare denne normfitten er åpnet; lifetime-fit, modelltrening og TEST er stengt.

Kildebevist: 5 523 147 M1-rader krever 7 224 276 276 bytes (6,73 GiB)
for bare signal/ctx-matrisene. Den tidligere Arrow-pluss-sammenkopieringen
beholder også originaltabellen. Nå gjenbrukes load_m1_feature_surface med
komplett diskbacking og samme validering; den kanoniske robuste fitterens
128 MiB arbeidsblokk, median/IQR og aliassemantikk endres ikke.
Ingen populasjon kuttes. Alle Entry-rader og kvalifiserte M1-stater
inngår, og MTF-utvalget gjøres av eksisterende kausal eier. 10 GiB/512 MiB
producer-cap gjenbrukes. Normalisering er CPU-preprosessering; ingen modell.
Plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/BASE_NORMALIZATION_FIT_001/PLAN.json
SHA256: e09500b4a7e75c451a0b047b8e2bdae30efc2fd90509a395e63387ba370dddf4.
Operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/BASE_NORMALIZATION_FIT_001/OPERATOR_002.py
SHA256: df616634903101d26b9d2333c764af8a91cfa02bd6d282f17e0084b5967bc539.
Den aldri kjørte OPERATOR.py er bevart; aktiv operator registrerer også
fit-start korrekt hvis den første statistikkflaten feiler.
Full inputbygging og populasjonskontroller er ikke relansert. TEST- og
VAL-datasett/manifester sperres for denne kjøringen. Delte prisfeature-/
MTF-inputs kan omfatte senere rader og identitetskontrolleres; bare fryst
TRAIN-union tilpasses. Terminal og strict-load kreves før videre binding.

Fullført 01.10 kl. 21:49:10 UTC med exit 0, kilde 300404cc uendret.
BASE_NORMALIZATION_FIT_001 er fullført med exit 0 og strict-load PASS:
alle 254 signalfelt, kontekst og M5/M15/H1/H4/D1 er tilpasset på fryst TRAIN.
5 748 166 lokale feature-rader og 4 647 700 kontekstrader inngår; null VAL/TEST.
Alle MTF-utvalg er tilgjengelige før TRAIN-slutt. Ingen modellforward/optimizer.
Basefitten skal gjenbrukes; tillatelsen er brukt opp. Før indeksbygg gjenstår
lifetime-statistikk, no-cap-økonomibinding og samlet norm-/førstetilstandsbevis.
Native trening og TEST er stengt; ingen edge er dokumentert.

TRAIN-beslutninger: 652 552 Entry og 3 995 148 M1-stater. Lokal union
er 955 670 M5-rader + 4 792 496 M1-historikkrader. Kausal MTF-normalisering
brukt unike M5=864 831, M15=331 256, H1=83 704, H4=21 727 og D1=3 867.
Siste tilstandsbar er 30.05.2025 kl. 20:59 UTC; alle MTF-barer var
tilgjengelige senest siste TRAIN-beslutning og før 01.06.2025. Historisk
kontekst før første TRAIN-entry er del av kausale inputvinduer.
Registrert maksimal prosess-RSS er 10 709 020 KiB. Dette er en annen
regnskapsstørrelse enn cgroup-minne; den verifiserte 10 GiB-cgroup-grensen
var aktiv, og jobben avsluttet normalt. Ikke kall RSS en målt cgroup-topp.

normalization: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/BASE_NORMALIZATION/normalization.json
SHA256: b2232f23f6e5c335b7102996427e9fcd6fbf51f11956afc5b3e7e0c20bd54a36.

result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/BASE_NORMALIZATION_FIT_001/EVENTS/RESULT_20261001T214910084388Z.json
SHA256: ffd68318389cd6802a4082fe5c4d7e5ea121938361b9a4ccc193a042846c331c.

terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/BASE_NORMALIZATION_FIT_001/EVENTS/TERMINAL_20261001T214910104710Z.json
SHA256: 400e1b69fc06682c7aa297b73cd4657e00d89eddc52a6a98dba71aee91af7fb6.

final_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/BASE_NORMALIZATION_FIT_001/EVENTS/FINAL_NORMALIZATION_REVIEW_20261001T215731942694Z.json
SHA256: 846a3042232dbf1a01c04ffbafe980243303cee131cf50314f3ab87f320ad033.

Nøkterne grenser: dette er inputstatistikk og konsistens, ikke modellfit,
læringsmåling, økonomi eller train/serve-paritet. Lifetime-normalisering og
samlet førstetilstands-/indeksbinding gjenstår. Full B forblir ufullført.

## Lifetime-normalisering — bundet 02.10.2026

Lifetime-eieren er rettet: sample-autoriteten strømmer nå uten å beholde
4 026 919 sample-objekter. Hashrekkefølge og utvalg er uendret. Publisering
kontrollerer skrevet manifest/arrays, strict-loader normaliseringen og bruker
fsync + atomisk no-replace. 29 fokuserte tester består, inkludert uendret
kontrakthash, begrenset sample-retensjon, korrupt staging og katalogkollisjon.
LIFETIME_SUMMARY_FIT_001 er bundet til én full-TRAIN-fit og VAL-telling uten
VAL-fit, under eksisterende 10 GiB/512 MiB producer-cap. Kilden fryses under
kjøring. Basefitten gjenbrukes; native trening, broker og TEST er stengt.
No-cap-kostnadsbindingen dekker foreløpig ikke den nye historiske perioden.
Indekser, læringsmåling, økonomi og edge er fortsatt ubevist.

Kilde-/geometribevis: fire eksisterende sparse-tabeller krever 6 653 317 760 bytes;
fitmatrisen har 8 053 838 × 7 float64 = 451 014 928 bytes. Dette er arraystørrelser,
ikke målt minnetopp. Ingen populasjon, feature eller beregning er redusert.
TRAIN: 652 552 Entry-par, 4 026 919 samples, begge sider per sample.
VAL: 70 880 Entry-par; bare livsløpsantall og sample-autoritet, null fitrader.
Eksakte count-arrays sammenlignes med fullført CLOSURE_STATE_SUPPORT_001.

plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LIFETIME_SUMMARY_FIT_001/PLAN.json
SHA256: ea47bf63c769fa8d16c7c0d0d1cd84b47c91379623a8965a0f25c94dab47c11f.

operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LIFETIME_SUMMARY_FIT_001/OPERATOR.py
SHA256: 6ba93350ba56f05d0dc14d379a16960cfef50721e7e245db1b6d5b48788593aa.

Den eksisterende prospective kostnadspolicyen dekker 01.06.2021–01.07.2026,
mens TRAIN nå starter 01.06.2011. Kostnadene er prospektive forskningsantakelser,
ikke verifisert historisk kostnadsfasit. Dens krav om broker-revalidering
åpner ikke broker under gjeldende prosjektregler. Dette er en separat
uavklart binding; lifetime-statistikk avhenger ikke av disse kostnadene.

Lifetime-kjøringen er fullført 01.10.2026 kl. 22:18:22 UTC (02.10 kl. 00:18 Oslo).
Lifetime-normaliseringen er fullført med exit 0 og strict-load PASS.
4 026 919 samples fra alle 652 552 TRAIN-entryer gir 8 053 838 siderader.
TRAIN-/VAL-counts matcher tidligere kontrollert geometri eksakt; VAL fikk
ingen fit. Kilden var uendret, TEST-tilgangsforsøk = 0, modellsteg = 0.
29 fokuserte tester består. Base- og lifetime-statistikk skal nå gjenbrukes.
Sluttbindingens samme no-replace-publiseringsfeil er også rettet; 7 fokuserte
tester består. FINAL_BINDINGS_001 er bundet til samlet normalisering og
første M1-tilstand for hver Entry; dette åpner ingen fit eller modellkjøring.
Kostnadsdekning/broker-revalidering, indekser og mål gjenstår. Ingen edge.

Maksimal prosess-RSS 8 701 396 KiB; dette er ikke målt cgroup-topp.
Tillatelsen er brukt opp. Begge count-filer er byte-identiske med de
tidligere geometrifilene, inkludert totalt 1 072 124 182 TRAIN-overganger
og 977 163 449 VAL-overganger. Disse er observerbar støtte, ingen holdetidsregel.

lifetime_result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LIFETIME_SUMMARY_FIT_001/EVENTS/RESULT_20261001T221822646265Z.json
SHA256: 8033c8b4c656860531f9f07425d503ba42ea37f6b1f583aa2cf0b52af1be79ed.

lifetime_terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LIFETIME_SUMMARY_FIT_001/EVENTS/TERMINAL_20261001T221822663357Z.json
SHA256: 91395b324487818e0ac50c1b187db814160a8abfb518c043a4b2b4d608c70e4a.

lifetime_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LIFETIME_SUMMARY_FIT_001/EVENTS/FINAL_LIFETIME_REVIEW_20261001T223127425594Z.json
SHA256: 019a70575f2034400f426a49235c877fd0a9cb80354a2095d3b235e76c7ac513.

## Samlet normalisering og førstetilstand — bundet 02.10.2026

plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_001/PLAN.json
SHA256: 0a88f45358bb7e486d86e7e3221daf69e2a613cff09ebb123e5b6d60ea1c36e2.

operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_001/OPERATOR.py
SHA256: a10ac82c3809420eac27dce259355c9c906577374382a9632688c01b631f971f.

recipe: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_001/RECIPE.json
SHA256: 6b05350485cd825e3b0dcbb296369a6ad05ebb40cfd7b0f291fea2994f56bad8.

Eksisterende sluttbindingseier har nå verifisert staging-inventar og skrevne
bytes før atomisk no-replace-rename og katalog-fsync. Kjøringen bruker 4 GiB
audit-cap og sammenligner første M1-posisjon/count-hasher med tidligere
uavhengig geometri. Ingen ny fit, sampler-valg, native admission eller økonomi.

## Observert nullspread-mismatch — 02.10.2026

Sluttbindingens første forsøk feilet før publisering: én ekte TRAIN-rad,
12.12.2012 kl. 17:00 UTC, har BID=ASK. Ingen kryssede priser eller
float32-kollaps ble funnet på 652 552 TRAIN-/70 880 VAL-førstetilstander.
Entry-fill-eieren avviste likhet selv om de øvrige aktive kontraktene
aksepterer ASK>=BID. Minste rettelse er <= til <; positive, endelige priser
kreves fortsatt. 22 fokuserte tester består, inkludert ekte bridge-kode
med nullspread og fortsatt avvisning av kryssede/ugyldige priser.
FINAL_BINDINGS_001 og feilen er bevart. FINAL_BINDINGS_002 er bundet
etter rettelsen; base-/lifetime-fit gjenbrukes uendret. Ingen nye modellsteg.


diagnose: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_001/EVENTS/FILL_QUOTE_MISMATCH_DIAGNOSIS_20261001T223803594386Z.json
SHA256: 85339887921ae83bc0547bb78723c8edfc6cfc9d4d608d07650451c821b99a49.

failed_terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_001/EVENTS/TERMINAL_20261001T223622327500Z.json
SHA256: b26c26c690de58b880d771a45f539aeb9616ecbec47f0cfcd1dfdd5c33dcc75c.

active_plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_002/PLAN_002.json
SHA256: ddf6a8526bfc34331fc3de2f5f6a2ccc37a1247a6b8a60c3ec11531fc313ce4a.

active_operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_002/OPERATOR_002.py
SHA256: 36da7d078c75cf53e5184b345ce0600e638b27b871c2775785dbd8b551c970a6.

Den aldri kjørte første planen/operatoren i FINAL_BINDINGS_002 er bevart;
aktiv PLAN_002 binder riktig testantall 22. Ingen beregning er relansert der.

## Sluttbindinger fullført — 02.10.2026

Base- og lifetime-normaliseringen samt samlede førstetilstands-bindinger
er fullført. Lifetime-fit: 8 053 838 siderader fra alle 652 552 TRAIN-entryer;
VAL/TEST-fit = 0. FINAL_BINDINGS_002 sluttet med exit 0 og uendret kilde.
Samtlige 652 552 TRAIN-/70 880 VAL-entryer kobles eksakt til første M1-bar;
posisjoner og tilstandstelling matcher tidligere uavhengig geometri.
Alle publiserte hasher og samlet normalisering er etterkontrollert.

Rettet: unødvendig lagring av over fire millioner sample-objekter,
overskrivbar publisering hos lifetime-/sluttbindingseierne og én faktisk
nullspread-mismatch i Entry-fill-kontrakten. Testene bevarer utvalg/hash,
avviser korrupt staging og navnekollisjoner og godtar BID=ASK uten å
godta kryssede eller ugyldige priser. Fokuserte tester: 29, 7 og 22 i
de tre respektive endringsbølgene; påkrevde Git-kontraktssjekker består.
Det feilede FINAL_BINDINGS_001 er bevart; ingen normalisering er refittet.

Neste: eksisterende indekseier må bindes til nye eksplisitte stier/kalender.
Før faktisk indeksbygg må no-cap-kostnadsdekning fra 2011 og policyens
krav til oppdaterte broker-vilkår avklares. Gammel policy starter i 2021
og sist registrerte kontoverifisering er 10.09.2026. Broker er stengt.
Deretter gjenstår mål, separate TRAIN/kontrollkoordinater og lærerparitet.
Native trening, TEST, live/paper og spending er stengt. Ingen edge er bevist.
Gjenbruk fullførte artefakter; ikke relanser produsentene.

Historisk teknisk bakgrunn følger. Tidligere neste-steg-tekst nedenfor
er erstattet av sammendraget over og gjeldende next_action i statusfilene.

FINAL_BINDINGS_002 fullført 01.10 kl. 22:43:13 UTC (02.10 kl. 00:43 Oslo),
kilde b093fa254820e5d53fed0bbb64a8fa405340b733. Prosess-RSS maksimalt
1 613 620 KiB; 4 GiB-cgroup aktiv. Ingen fit, modellforward eller optimizersteg.
Samlet normkontrakt SHA256: 9a30939aa60041d91bc1bfdd0eede3c5afd7c691d801e0fc3e5049d553c73d37.

result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_002/EVENTS/RESULT_20261001T224313184213Z.json
SHA256: af45d48c081559777e8119f8cde0cfd2a1f077c416c48ee0800adb003f06900d.

terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_002/EVENTS/TERMINAL_20261001T224313203883Z.json
SHA256: 9a1ded0f7026f35669206272974301bcd6f378fbe9d16f51324040fb21edb7e5.

final_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/FINAL_BINDINGS_002/EVENTS/FINAL_BINDINGS_REVIEW_20261001T225810309909Z.json
SHA256: 0bee59989c5fa28f6652c2a4127c755922622dfa8597226fc2e23104f9cf2814.

COMPOSITE_NORMALIZATION.json: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/FINAL_BINDINGS_V1/COMPOSITE_NORMALIZATION.json
SHA256: 26be1942c28718fefcdf3790eda9fa94520f90a9d479833acc0b8636ea75d0e5.

FINAL_BINDINGS_BUNDLE.json: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/LEARNING_PREPARATION_001/FINAL_BINDINGS_V1/FINAL_BINDINGS_BUNDLE.json
SHA256: a9dba4e5dc985cc265be5bc348f30f0df6c41054d35b0b139b587378730465e7.

Avgrenset videre feilsøk: indekseierens _paths bruker fortsatt gamle
ENTRY_WINDOW-parquet-, M1_CHILD_VIEWS_V1- og CLOSURE_AUTHORITY-navn.
Ikke lag datakopier/aliaser; bind faktiske artefakter i eksisterende eier.
Dens kalender og os.replace-publisering må også kontrolleres før bruk.
Prospective cost-produsent har fortsatt os.rename før sluttsjekk og
sletting av output i feilsti; no-cap-/readiness-eiere bruker replace.
Disse ble kildeinspisert, ikke kjørt eller endret i denne bølgen;
rett konkrete publiseringsblokker før de eventuelt tas i bruk.

## Indekseier: faktiske stier, kalender og økonomisk identitet — 02.10.2026

Indekseieren bruker nå faktiske filstier fra de ferdige, hashbundne
sluttbindingene og Entry-admission. Nye kalendergrenser kontrolleres mot
det fryste designet. Ingen datakopier eller aliaser. Den separate gamle
siste-år-modusen beholder sin kalender og skal ikke brukes for dette bygget.
Økonomi valideres nå mot Entry-datasettets ID, radantall, begge sider og
kalender, gjennom eksisterende livsløpseier. Den gamle sjekken brukte
økonomiartefaktens egen ID som forventning.
Alle tre indeks-publiseringsruter bruker fsync + no-replace; data og
kildebindinger kontrolleres før endelig navn. 37 fokuserte tester består.
INDEX_SOURCE_AUDIT_001 er bundet til kontroll på ekte metadata, uten
Parquet-/TEST-tilgang, fit eller publisering. Kostnadsautoritet gjenstår.


plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_SOURCE_AUDIT_001/PLAN.json
SHA256: cdb191fdc42207663fea424ddef3a9095e9ffb16e3b57fef29a1451eeef64878.

operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_SOURCE_AUDIT_001/OPERATOR.py
SHA256: 72e5f352d6ae0aafdc18d0b60a6e40693bd50d8ddeae956732f02aa82403bc97.

Korrigering til forrige neste-steg-beskrivelse: full-TRAIN-indeksbyggeren
hadde ingen hardkodet 2021-start. Den gamle kalenderen tilhører separat
latest-year-utvalg. Full-TRAIN-ruten bevares og kontrollerer nå eksplisitt
kalenderen fra bundet admission/design. Dette er ingen ny populasjon.

## Kildekontroll fullført; konkret vilkårskontroll klar til beslutning — 02.10.2026

Indekseierens nye kilde-/kalenderkontroll er kjørt på de ekte metadataene:
652 552 TRAIN-/70 880 VAL-rader, exit 0, null Parquet-/TEST-tilgang.
37 fokuserte tester består. Faktiske filstier gjenbrukes uten kopier; økonomi
bindes til Entry-ID, alle rader/sider og kalender. Alle indeksruter publiserer
nå med fsync/no-replace etter kontroll av data og kildebindinger.
Selve indeksene er ikke bygget: kostnadsautoriteten mangler fortsatt.

En konkret, testet lesekontroll er klargjort som COST_TERMS_REVALIDATION_001:
maksimalt to GET-kall mot OANDA practice, ett for kontoens vilkår og ett
for XAUUSD-instrumentets vilkår. De 258 lagrede historiske fyllene og
finansieringsobservasjonene gjenbrukes byte for byte. Ingen nye
transaksjonsoppslag, ordre, handel, redirects, retries eller spending.
15 syntetiske tester består; ingen faktisk broker-forespørsel er gjort.
HTTP-feil skjuler konto-URL, og publiseringen er atomisk no-replace.
Manglende eller endret miljø avvises før forespørsel.

GX1_RULES.md stenger broker-adgang. Denne ene avgrensede lesekontrollen
krever derfor et uttrykkelig brukerunntak. read_only_broker_terms_authorized
er false; klargjøring er ingen godkjenning. Kostnadspolicyen krever ferske
vilkår og dekning fra 2011 før økonomi-/indeksbinding kan ferdigstilles.
Native trening, TEST, live/paper og spending er fortsatt stengt.


index_source_result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_SOURCE_AUDIT_001/EVENTS/RESULT_20261001T231110412906Z.json
SHA256: 4c2a44c74fc67e71d0eef94ba2569904723826500c7b6aa0f92e7bd9d58b59ca.

index_source_terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_SOURCE_AUDIT_001/EVENTS/TERMINAL_20261001T231110432438Z.json
SHA256: 720c7645c55e17988d5ac2cb52f3a5a8dc5eb0aeb6f78ed51e8fdaa97760c990.

index_source_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_SOURCE_AUDIT_001/EVENTS/FINAL_SOURCE_REVIEW_20261001T232144276281Z.json
SHA256: 19f760a6772baba8bd4a8d3cf857b276470826ebbad8f1b0a994303d50b3e792.

terms_plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/COST_TERMS_REVALIDATION_001/PLAN.json
SHA256: 22d5b8de8d166a4d0b75669d51bf348b203d128d19e8fb7f7da0dd68b0e0e343.

terms_operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/COST_TERMS_REVALIDATION_001/OPERATOR.py
SHA256: 04346ae2429f8532a859b359e3f882b719780ec96724751becea4541a7be2de4.

Begrunnelsen for brukeravklaring er GX1_RULES.md, Omfang: «Live, paper,
demo, broker, daemon, collector, publisher, promotion, drift og
online-adaptasjon er forbudt». Forslaget åpner bare de to nevnte
lesekallene og lokal renset vilkårsevidens, ikke noen handelsfunksjon.
Konfigurasjonen må være practice og samsvare med den bundne gamle
evidensen; credentials leses først ved en autorisert kjøring. HTTP-redirects
avvises. Kontonummer, token, saldo og råresponser lagres ikke i artefakten.

Eksisterende full-observasjonsrute er beholdt og syntetisk testet, men er
ikke autorisert av dette forslaget. Testfixturens feilskrevne API-feltnavn
ble rettet før siste grønne testkjøring; produksjonsfeltet ble bevart.
Ny vilkårsevidens blir fortsatt ikke historisk kostnadsfasit eller
lønnsomhetsbevis. Hvis dagens gebyr-/finansieringsvilkår avviker fra den
forhåndsregistrerte policyen, må policybindingen vurderes separat før bruk.

## Kostnadspublisering rettet; brukerunntak avventes — 02.10.2026

Kostnadskjedens publisering er nå rettet i eksisterende eiere:
sluttkontroll av alle hashbundne bytes før publisering, fsync og atomisk
no-replace. Eksisterende bevis overskrives eller slettes ikke ved feil;
mislykket staging bevares for retention-eieren. 38 fokuserte tester består.
Ingen kostnadstall er endret, og ingen ny reell kostnadspakke er publisert.
Brukerspørsmålet om den avgrensede vilkårskontrollen er sendt; svar avventes.

Kildebevist feil: kostnadspolicyprodusenten brukte overskrivbar rename før
strict-load og slettet målkatalogen ved unntak. No-cap- og readiness-eierne
brukte fast midlertidig fil og replace. De bruker nå den eksisterende
immutable-eierens no-replace-operasjon. Endelige filstier og hasher bevares
gjennom en eksplisitt staging-lesekobling; ordinære lesere bruker fortsatt
de endelige filene og strenge hasher. No-cap publiserer autoritet sist,
etter ny kontroll av dens varige kildefiler. Feilet staging slettes bare
gjennom retention-eieren.

Målt på syntetiske testdata: 38 beståtte tester, inkludert for kort
quotedekning, korrupte staging-bytes, ekstra filer/symlinker, kollisjon
med tom eller fylt målkatalog, sen filkollisjon, kildeendring mellom
counts og autoritet samt fsync-feil etter publisering. Syntaks,
stale-path-kontroll og diff-sjekk består. Ingen parameterverdi, modell,
mål eller populasjon ble endret. Omfattende innrykkdiff i policyprodusenten
skyldes fjerning av try/except-grenen som slettet output; innholdet
ellers er bevart. Gjeldende policy og økonomiske konklusjoner er uendret.

Ubevist: ny faktisk kostnadsautoritet, nåværende brokervilkår, indeksbygg,
native læring og handelsøkonomi. Det uttrykkelige brukerunntaket er
fortsatt ikke mottatt; ingen brokerforespørsel er utført.

## Designbundet lærerfunksjon og gjenoppretting — 02.10.2026

Lærerfunksjonen er nå eksplisitt bundet gjennom de eksisterende native
eierne. V38 velger samme aktuelle funksjon for ONLINE og TARGET, med begge
parameterfrie normer bevart; historiske forsøk beholder sin opprinnelige
lærer. Vekthash alene slipper ikke gjennom feil eller manglende
funksjonsidentitet. To overflødige modellkopier for strukturkontroll er fjernet.

Fokuserte syntetiske tester består: 108 bestått/3 hoppet over i første
samlede grønne runde; etter siste gjenopprettingsrettelse består de 51
berørte målingstestene/3 hoppet over. Tre utelatte kombinasjoner gjelder
avledede mål utenfor deres TRAIN-only-scope. Samme frosne evalueringsmodus
gir bit-identiske Entry-/Exit-utdata i testmodellen. Ulik requires_grad-
status ga et lite CPU-avvik og er ikke dokumentert native train/serve-paritet.

Dette er kilde-/testbevis. Reell v38-initialisering og læring er ikke kjørt.
Native kobling til separate fysiske TRAIN-/VAL-kilder, læringsmål og
komplett admission gjenstår. Nytt design avvises hvis TRAIN forsøkes gjenbrukt som
fysisk VAL. Vilkårsspørsmålet er fortsatt ubesvart; broker er stengt.

Kildebevist: _copy_frozen_prefix_reference_model fjernet tidligere
ONLINEs parameterfrie encoder.norm og siste fuse-norm ved alle prefix-kall.
Det endret funksjonen uten å endre state-dict-hashen. Nå velger eksisterende
eier funksjonsparet fra det hashbundne designets eksplisitte separate
TRAIN/VAL-roller og aktuelle TARGET-deklarasjon. Den fryste runtime-designfilen
er byte-identisk med repository-kopien (SHA256
5235db5c9e6ea619ec971673d58a38dc462bcf325122dc5dff0c6d4f8584a864).

Kallstedsrevisjonen dekker alle tre læreropprettingene i native-koordinatoren
(fersk, resume og epokeovergang) samt initial-/finalmåling. De får nå eksakt
funksjonsparet fra bindingen. Den historiske joint-gradient-proben beholder
sin opprinnelige lærer. De to gamle frozen-policy-/representasjonsrutene
validerer fortsatt ONLINE-strukturen, men allokerer ikke lenger en full
modellkopi bare for å kaste den. Ingen historiske checkpoints er omskrevet.

Gjenoppretting krever komponentens funksjonsidentitet. I aktuell modus
må også initialkvitteringen og fersk tilstandsfil deklarere samme funksjon.
Lagringsskjemaet i gamle kvitteringer uten feltet bevares i historisk modus;
et oppgitt felt som motsier bindingen avvises. Måling sjekker funksjonsparet
mot både det fryste designet og den varige sesjonskontrakten før lærerkopi
eller forward. Identiske vekthasher er utilstrekkelig.

Testavklaring: første sammenligning hadde trainable ONLINE mot frozen
TARGET, begge i no-grad/inference, og ga Exit-Q-avvik
2.2351741790771484e-08 bps på syntetiske CPU-inputs. Gjentatt ONLINE var
eksakt; både-frozen og både-trainable var eksakte. Avviket ved ulike
requires_grad-flagg bestod også uten MHA-fastpath. Dette isolerer forskjellen
til utførelsesmodusen, men beviser ikke en bestemt intern kernelårsak.
Testen for funksjonskopiering sammenligner nå samme frosne modus og krever
fortsatt bitlikhet, bevart RNG, like vekter og uavhengig parameterlagring.
Ingen numerisk produksjonstoleranse er endret.

Fokuserte regresjoner dekker avvikende/manglende funksjonsmetadata, ukjent
funksjon, endret design, innblanding av samme fysiske kilde for kontroll,
initial-/resume-semantikk, bevaring av pointer/RNG og avvisning før forward.
Syntaks, stale-path-scan og diff-sjekk består. Ny faktisk initialbaseline,
native train/serve-paritet, fysisk kontrollruting, etikettkoordinater,
kostnadsautoritet og læring er ufullført. training_enabled=false består.

## Gjenbruk av ferdige hjelpefasiter — forhåndskontroll 02.10

AUXILIARY_REUSE_PRECHECK_001 er fullført med exit 0 og uendret kilde.
Alle 652 552 TRAIN-/70 880 VAL-rader, inkludert CONTROL256, har komplett
tidsstøtte innen egen periode. De 37 faste hjelpefasitene krever opptil
96 observerte M5-barer; de fryste policyene bruker 19 M5 / 95 M1-minutter.
TRAIN-policyene er identiske i begge datasett, og fem måleiere er byte-like
produksjonskoden. Originale hjelpefasiter skal gjenbrukes uten ny policy-fit
eller egen produsent for erstatningsetiketter.

Dette er ekte klokke-/metadatabevis; målverdiene er ikke uavhengig beregnet
på nytt. Native binding av separate fysiske kilder og radkoordinater samt
Entry/Exit-reference-Q gjenstår. Ingen ny fit, modellkjøring eller TEST-tilgang.
Vilkårsspørsmålet er fortsatt ubesvart og broker-adgang er stengt.

Kilde: `AUXILIARY_REUSE_PRECHECK_001` under gjeldende run-root.
Plan-SHA: `8902790fc1f805710c0c0f92b189e5a0a66c34cb09d965bd38b780d79dbc2efb`.
Resultat-SHA: `a5175f4e2ef1db81e0d1d35fe66270dfd3f5ad69243cc37bf97d530670326f68`.
Terminal-SHA: `dcff68749ded9f90c94910560b0ea4299fd152404fc8d20091f57668641acd8a`.
Eksakte stier og sluttkontroll er bundet i begge statusfilene.

Lesekontrollen hashet de bundne datafilene og leste bare tre tidskolonner:
komplett før-TEST M1, opprinnelig fysisk TRAIN og fysisk VAL. 5 959 045 M1-
tidsstempler gir 1 215 514 M5-tidsstempler, med gjenbruk av full råprisparitet.
Retningspolicy og posisjonsstørrelses-ECDF består eksisterende strict-load.
Tilpasningsperioden er 01.06.2011–31.05.2025; eksakt policy-fit-slutt
23:54:59 kommer fra eksisterende eier, ikke en ny grense.

Siste TRAIN-støtte for K96 stenger 30.05.2025 kl. 21:00 UTC. Siste VAL-
støtte stenger 01.07.2026 kl. 00:00 UTC og tilhører M5-baren som åpnet
30.06 kl. 23:55; ingen juli-/TEST-bar er lest. De eksakte M1-policyutfallene
slutter senest henholdsvis 30.05.2025 og 30.06.2026 kl. 14:35/16:35 UTC.

Nyeste hendelser, kvitteringer, hasher og alle rapporterte grensetider er
etterkontrollert uten ny full datasettlesing. Dette beviser verken de
numeriske etikettene uavhengig, aktuelle reference-Q-mål, læring, økonomi,
train/serve-paritet eller bytte av M5-produksjonskilde.

## Fysisk VAL-kontroll i eksisterende eiere — 02.10

Kontrollkjedens gamle TRAIN-/juni-binding er rettet hos seks eksisterende
eiere. En eksplisitt fryst VAL-kontroll binder hele fysisk VAL, filstier,
radmapping og CONTROL256 gjennom tilstandsbygger, referansemåling, native
kontekst og gjenopptak av replay. Historisk standardrute er bevart.

Syntetiske tester: 131 i samlet grønn runde; etter siste filstirettelse
består alle 60 tester i berørt fil. Totalt 132 unike testtilfeller er dekket.
Ekte fryst design gjenkjennes, men avvises uten bundet VAL-indeks før
datalesing. Ingen reell modellkjøring, læring eller TEST-tilgang er utført.

Hovedbyggerens recipe-/treningsbinding krever fortsatt den gamle felles
TRAIN-kilden og må tilpasses før native kjøring. Dataset-eierens gjenbruk av
originale hjelpefasiter er nå rettet og kontrollert som beskrevet øverst.
Ferske kostnadsvilkår og faktiske indekser mangler fortsatt.

Evidens: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/PHYSICAL_CONTROL_SOURCE_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T004908392141Z.json`, SHA `7fecbea0e848190367e203c2c302168e78b7fbbf6a83ee76b5cbf4213c45e378`.
De opprinnelige JUnit-rapportene er bevart i hashbundet TEST_EVIDENCE-hendelse.
Dette er kilde- og syntetisk integrasjonsbevis, ikke native admission.

Feilklassen ble fulgt gjennom kohorteier, tilstandsbygger, Entry-referanse,
native kontekst og replay-/resultatvalidering. Den nye kontrollen krever
samme hashbundne VAL-indeks og -manifest samt full foreldretidsakse og
identisk radmapping. En kopi på ubundet manifeststi avvises selv ved like bytes.
Kohorten kan ikke fjernes eller byttes når tilstandsbyggeren er bundet.
Ingen ny beslutningsregel, horisont, normalisering eller modell er innført.

Full komponentrute, direkte gjenbruk av Dataset-hjelpefasiter og separate
trenings-/målekoordinater gjenstår. Eksisterende kostnads-/kjøreporter er stengt.

## Fysisk Dataset-binding og avgrenset målkontroll — 02.10.2026

Dataset-bindingens gamle krav om erstatningsfasiter fra felles TRAIN er rettet.
V38 gjenbruker nå originale mål fra hver fysisk TRAIN-/VAL-fil, med egen klokke,
radbinding og kanoniske TRAIN-policyer. Historisk prefix-rute er bevart.

PHYSICAL_AUXILIARY_BINDING_AUDIT_001 sluttet med exit 0 og uendret kilde.
Den samme funksjonen som Dataset kaller, har kontrollert tid og alle 47 aktive
målkolonner på ekte 652 552 TRAIN-/70 880 VAL-rader. Alle verdier/dtyper er
gyldige og bevares eksakt; CONTROL256 bindes bare til fysisk VAL. Ingen ny fit,
modellkjøring eller TEST-tilgang. Dette er en ekte målprojeksjon, ikke full
native Dataset-konstruksjon eller uavhengig ny beregning av fasitene.

81 fokuserte syntetiske tester og påkrevde Git-kontraktssjekker består.
Testene dekker faktisk Dataset-konstruksjon og __getitem__ på testdata.
Recipe-/hovedbygger-/trenings- og målebindingene må fortsatt tilpasses dagens
normalisering, målbevis og separate radkoordinater. Indekser/kostnadsautoritet
gjenstår, og brokervilkårsspørsmålet er ubesvart. Ingen læring eller edge er bevist.

## Recipe-binding av eksisterende full-TRAIN-bevis — 02.10.2026

Recipe-eierens identitetskontroll kan nå gjenbruke ferdig normalisering fra
hele fysisk TRAIN og beviset for originale hjelpefasiter. Den binder separate
TRAIN-/VAL-kilder, kildeklokker, hele TRAIN-populasjonen og sampleautoriteten
for lifetime-normalisering. Historisk prefix-rute er bevart.

RECIPE_PREPROCESSING_AUDIT_001 sluttet med exit 0 og uendret kilde.
Den faktiske funksjonen i recipe-eieren godtar de ekte ferdige artefaktene:
652 552 TRAIN-/70 880 VAL-rader og de opprinnelige normaliseringshashene.
Kontrollen leste metadata/radarrayer, ingen Parquet-data eller TEST. Ingen
ny fit, modellkjøring eller optimizersteg. Native koordinater er eksplisitt ubundet.

119 fokuserte syntetiske tester består; tre avledede måltilfeller utenfor
deklarert TRAIN-only-scope er utelatt. Testene fanget forskjellen mellom
prosjektets hashformater; den nye kontrollen bruker normaliseringsprodusentens
egen hashfunksjon. Påkrevde Git-kontraktssjekker består.

Dette er datagrunnlagets identitet, ikke full native recipe eller launch.
Hovedbygger-/trenings-/målebindinger og faktisk epoch0/4096/TRAIN256-rekkefølge
gjenstår. Indekser og fersk kostnadsautoritet mangler fortsatt, og
brokervilkårsspørsmålet er ubesvart. Ingen læring eller edge er bevist.

## Sampler-populasjon og uforanderlig benchmarkkvittering — 02.10.2026

Benchmarkens gamle binding til 65 295 TRAIN-rader er rettet. Populasjonen
kommer nå fra den kanoniske sampler-kontrakten, og alle kandidater må ha
samme kilde og populasjon. Kandidatfabrikken gjenbruker produsenteieren
framfor en ufullstendig kopi av valideringen. Endrede kontrakthasher,
utvalgsregler og blandede kilder avvises.

Resultatpublisering er rettet til eksisterende fsync/no-replace-eier.
Navnekollisjon, inkludert en fil som dukker opp under publisering, kan
ikke overskrive tidligere evidens. Korrupt staging og dangling symlinker
avvises; feilet staging bevares for retention.

46 unike fokuserte syntetiske tester er dekket: 44 i samlet grønn runde,
deretter 31 i berørt fil etter en siste hashkontroll. Ingen utelatte tester.
SAMPLER_SOURCE_REVIEW_002 har kontrollert de ekte fryste kandidatene med
652 552 TRAIN-rader. De gir 80/40/20 delrunder per populasjonssyklus og
forskjellige første 4096 Entry-ID-er. Exit 0; kilde og inputs uendret.
Bare metadata og den eksisterende kausale sampler-eieren ble brukt;
ingen Parquet-/TEST-lesing, modellkjøring, optimizer eller broker-kall.
Første review-forsøk feilet før målingen på et feil argumentnavn i
kvitteringskallet; den opprinnelige operatoren og feilkvitteringen er bevart.

Faktisk throughput-/minnebenchmark og sampler-valg er fortsatt ikke kjørt.
Treningsrekkefølgen kan derfor ikke fryses ennå. Historisk 65 536-valg og
V3→V4-overføringskvittering er ikke autoritet for v38. Hovedbyggerens faste
budsjett, binding av nytt målt valg og separate TRAIN-/VAL-koordinater
gjenstår. Ferske kostnadsvilkår og faktiske indekser mangler; det tidligere
brokerspørsmålet er ubesvart. Native trening er stengt. Ingen edge er bevist.

Evidens: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/SAMPLER_SOURCE_REVIEW_002/EVENTS/SOURCE_REVIEW_20261002T015613832707Z.json`, SHA `a3a9e0fb9847ce8b24c316c2afff513f0d5049241dbb1dc0b77a03486aedb698`.

## Direkte målt sampler-valg og faktisk arbeidsmengde — 02.10.2026

Ny sampler-admission er koblet til de eksisterende eierne. Et ferskt
benchmark kan nå bindes direkte til sin eksakte full-TRAIN-indeks, kandidatfil,
TRAIN-indeksmanifest og sluttbindinger. Vinneren etterkontrolleres med samme
rangering som produsenten bruker. Den historiske V3→V4-ruten er bevart som
historisk autoritet; den kan ikke brukes som dagens v38-valg.

En ytterligere mismatch ble rettet: benchmarken materialiserte uten dagens
fryste referansepolicy og tidsgrense. --chronological-design binder nå disse
til samme factory/collator som native bruker, og kvitteringen registrerer
faktiske Dataset-sekvenslengder. Native krever identisk design, policy,
tidsgrense og MTF-geometri. Hovedbyggeren bruker budsjettet fra den kontrollerte
kvitteringen og sjekker adapterens faktiske kontrakt/policy etter konstruksjon.
Et ufullstendig benchmark eller feil budsjett, kilde, vinner eller arbeidsmengde
slipper ikke gjennom. Publisering av valgt sampler bruker eksisterende
atomiske no-replace-eier; den gamle overskrivbare writer-kopien er fjernet.

149 unike fokuserte syntetiske tester består: 148 i samlet runde og 34 i
berørt fil etter én ekstra CLI-koblingstest; ingen utelatte tester.
MEASURED_SAMPLER_ADMISSION_REVIEW_001 sluttet med exit 0 og uendret kilde.
Ekte fryst design binder referansepolicy 61c8aaaa… og cutoff 01.06.2025 UTC.
Den faktiske native inngangskontrollen avviser designet uten målt sampler,
før data-/modellkonstruksjon. Kandidatmetadata har fortsatt 652 552 TRAIN-rader.
Dette er ekte metadatakontroll og kilde-/syntetisk integrasjonsbevis.

Ingen reell throughput-/minnebenchmark eller sampler er valgt, og ingen
native koordinater er fryst. Ferske kostnadsvilkår og faktiske indekser mangler;
det tidligere, avgrensede brokerspørsmålet er fortsatt ubesvart.
Fysisk komponentruting, trenings-/målekoordinater og full native admission
gjenstår. Neste kodearbeid er å gjenbruke fullført fysisk preprocessing og
originale hjelpefasiter i hovedbyggerens eksisterende komponentbinding.
Ingen ny fit, modellkjøring, optimizer, TEST eller broker-tilgang er utført.
Native trening er stengt. Ingen læring eller edge er bevist.

Evidens: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/MEASURED_SAMPLER_ADMISSION_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T022003203735Z.json`, SHA `c14316d18dc4269d3637a3ef65325d57f978834e180d62620ab0e7f209edf16a`.

## 02.10: fysisk komponentruting og fryste native radkoordinater

Hovedbyggerens antakelse om felles TRAIN-/kontrollfil er rettet for v38.
Den gjenbruker fullført normalisering og originale hjelpefasiter, og bygger
separate TRAIN-/VAL-datasett med hver sin sekvenskontroll og featurekilde.
Kalendervinduene må stemme med det fryste designet. TRAIN-indeksens forelder
bindes eksplisitt til den samme fysiske TRAIN-kilden.

Native radkoordinater må bindes til et faktisk målt sampler-valg.
Hele TRAIN-rekkefølgen, første 4096 rader og TRAIN256-proben kontrolleres;
proben bruker den eksisterende deterministiske selectoren. Hovedbyggeren
sammenligner fryst rekkefølge med den faktiske adapteren og validerer
kontrollkonteksten før modell eller optimizer opprettes. Historisk prefix-rute
beholder sine opprinnelige kilder og bindinger.

134 unike fokuserte syntetiske tester består. Etter siste koblingstest
består alle 38 tester i komponentfilen; ingen tester er utelatt.
Testene dekker separate filer, feature-/sekvensruting, samplerbudsjett,
normaliseringsgjenbruk og avvisning av endrede bytes, kalender og rekkefølge
før initialisering. Dataset-, factory- og modellobjekter er mockede i
rutingtesten; den er ikke en reell native gjennomkjøring.

PHYSICAL_COMPONENT_SOURCE_REVIEW_001 sluttet med exit 0 og uendret kilde.
Den faktiske komponentkontrollen gjenbruker de ekte fullførte
preprocessing-artefaktene for 652 552 TRAIN-/70 880 VAL-rader og stopper
på NATIVE_PHYSICAL_COORDINATES_REQUIRED. Ingen rå-Parquet eller TEST ble
lest; ingen ny fit, reell modellkjøring, optimizer eller broker-tilgang.

Treningskoordinatorens og måleeierens gamle fellesfilbindinger gjenstår.
Faktiske indekser, ferske kostnadsvilkår, benchmark, sampler-valg og native
radkoordinater er fortsatt ikke kvalifisert. Det tidligere avgrensede
brokerspørsmålet er ubesvart. Neste kodearbeid er koordinatorens binding av
separate TRAIN-/VAL-kilder og originale mål, med gjenbruk av de nye
koordinateierne. Native trening er stengt. Ingen læring eller edge er bevist.

Kildekontroll: PHYSICAL_COMPONENT_SOURCE_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T024836958307Z.json
SHA256 c31a35326435e387f8d5d14a12514c1866753026b23877e6ce890f3096edafbf.

## 02.10: fysisk treningskoordinator og uendret sesjon ved gjenopptak

Treningskoordinatoren kan nå binde v38 til separate fysiske TRAIN-/VAL-kilder,
fullført normalisering, originale hjelpefasiter og fryste native radkoordinater.
Hver rad-ID kontrolleres mot sin egen kilde; like tall i to forskjellige
filer er ikke lenger feilaktig behandlet som overlapp i samme populasjon.

Dataset-roller, radmasker, målkolonner, policyhashene og den faktiske sampler-
kontrakten må stemme med de fullførte bindingene. Sesjonskontrakten gjenbruker
sine eksisterende filhasher til å avvise endrede TRAIN-/VAL-bytes, og bevarer
design, kildefiler, valgt sampler, epoch0-rekkefølge og modellfunksjon ved
gjenopptak. V38 kan ikke arve den historiske utvidelsen til 512 steg.

234 fokuserte syntetiske tester består; ingen tester er utelatt.
Sammenhengende 4 steg og 2+2 med gjenopptak gir identisk modell, lærer,
optimizer, EMA, scheduler, RNG, radrekkefølge og fremdrift i den eksisterende
sesjons-/checkpointkoden. Endrede kildebytes, rekkefølge eller koordinatbinding
avvises uten nye optimizersteg eller endring av aktiv checkpointpeker.
Treningsfunksjonen og lærerens kopieringsfunksjon er erstattet i denne testen;
den bruker syntetiske rader og en liten lineær modell, ikke native CUDA.

PHYSICAL_COORDINATOR_SOURCE_REVIEW_001 sluttet med exit 0 og uendret kilde.
Koordinatorens faktiske inngangskontroll gjenbruker de ekte fullførte
preprocessing-artefaktene og stopper på NATIVE_PHYSICAL_COORDINATES_REQUIRED,
før Dataset-/modell-/optimizerobjekter. Kontrollens deklarerte kilder har
652 552 TRAIN-/70 880 VAL-rader. Ingen rå-Parquet-/TEST-lesing, broker-kall,
fit eller modellkjøring på ekte data er utført.

Måleeierne og deres initial-/sluttmåling må fortsatt kobles til separate
TRAIN-/VAL-kilder. Faktiske indekser, ferske kostnadsvilkår, benchmark,
sampler-valg og native radkoordinater mangler. Det tidligere avgrensede
brokerspørsmålet er ubesvart. Neste kodearbeid er den eksisterende målekjedens
kilde- og koordinatbindinger; fullført preprocessing skal gjenbrukes.
Native trening er stengt. Læring, edge og reell native gjenopptaksparitet
er ikke bevist.

Kildekontroll: PHYSICAL_COORDINATOR_SOURCE_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T030521080102Z.json
SHA256 fe92ac307d7847d273cbf603b9f253fb1da59eb6ca60d1a465d3e0b9b4ecfbe7.


## 02.10: separate fysiske kilder i initial- og sluttmåling

Initial- og sluttmålingen er nå koblet til separate fysiske TRAIN-/VAL-kilder.
To konkrete feil er rettet: TRAIN-proben brukte samme tilstandsbygger som
kontrollmålingen, og fysisk kontroll sammenlignet målingens egen hash med
hashen til det opprinnelige CONTROL256-utvalget.

Hver rolle binder nå sin egen indeks, indeksmanifest, kildefil og native
sampler. TRAIN256 følger den fryste proben fra de første 4096 native radene.
Kontroll bruker de allerede fryste CONTROL256-ID-ene og eksisterende native
fire-trekks-policy over fysisk VAL. Rekkefølge og gjentatte trekk bevares;
TRAIN-målingen sammenlignes også med den faktiske adapterens trekk.
En felles helper gjenbruker eksisterende tilstandsbygger uten duplisert
provider-/filkobling. Native admission kontrollerer de fryste målekohortene.

274 fokuserte tester består. Tre eksisterende kombinasjoner for avledede mål
utenfor TRAIN-only-omfanget er fortsatt deklarert utelatt. Nye tester bruker
syntetiske klokker, reelle indeks-/samplerkontrakter og eksisterende sesjonskode.
Initial- og sluttmåling rutes til riktig kilde; kilde-/manifestbytte, endrede
trekk og feil under kontrollmålingen avvises eller avbrytes uten endring av
lagret checkpoint, modell eller RNG. Modell-/referanseberegning er mocket;
256 små syntetiske optimizersteg i sesjonstesten er ikke native v38-trening.

PHYSICAL_MEASUREMENT_SOURCE_REVIEW_001 sluttet med exit 0 og uendret kilde.
Manglende faktisk koordinatbinding avvises med
CHRONOLOGICAL_MEASUREMENT_PHYSICAL_PREFIX_REQUIRED. Ingen ekte
målekoordinater er publisert, og ingen ekte data er målt gjennom modellen.

Neste arbeid er avgrenset koordinatproduksjon med eksakte M1-støttetider og
immutabel publisering gjennom eksisterende eiere. Faktisk indeks, ferske
kostnadsvilkår, benchmark og sampler-valg må først kvalifiseres før produksjon
eller måling. Det tidligere brokerspørsmålet er ubesvart. Fullført
normalisering og originale hjelpefasiter skal gjenbrukes.
Native trening er stengt; TEST er forseglet. Ingen læring eller edge er bevist.

Kildekontroll: PHYSICAL_MEASUREMENT_SOURCE_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T033523636474Z.json
SHA256 aeac9a13e5cef9791321f30b51fd4117d59c0814f48b3b48899ca4776701a7b0.


## 02.10: fysisk koordinatprodusent og eksakt M1-støtte

Koordinatprodusenten for fysisk TRAIN/VAL er implementert. Den gjenbruker
fullført preprocessing, målt sampler-valg og eksisterende native eiere til å
publisere full epoch0-rekkefølge, første 4096 rader, TRAIN256 og kontrollens
allerede fryste ID-er. Native anker-chunks gjenbrukes uten å materialisere
hele populasjonens overganger bare for å finne radrekkefølgen.

En konkret mangel er rettet: tidligere kunne oppgitte referansesluttider
være feilaktig tidlige og likevel passere periodegrensen. Nå rekonstrueres
ankeret og alle fire trekk fra hash-bundet M1-klokke, faktisk start-/sluttrad
og referansepolicyens beregningshorisont. Markedspauser inngår som faktiske
klokkehull; 120 steg er ingen maksimal holdetid. Feil kildesplit, endrede
M1-bytes, ugyldige koordinater og feilaktige sluttider avvises.
Måleforbrukeren binder også tilstandsbyggerens M1-fil og manifest.

Produsenten bruker atomisk publisering uten overskriving og streng lesing av
staged bytes gjennom de eksisterende koordinatvalidatorene. COMPLETE.json
kommer sist. Feil før fullføring gir ingen ferdig måleautoritet; gyldige
delartefakter og feilstaging bevares for retention-eieren. Sampler-admission
er delt mellom produsent og forbruker, uten en ny valgalgoritme.

287 fokuserte tester består; tre eksisterende avledede målkombinasjoner
utenfor TRAIN-only-omfanget er fortsatt deklarert utelatt. Publisering fra
syntetiske kilder er kontrollert ende til ende, med native radrekkefølge også
over ufullstendige sampler-chunks, klokkehull, observasjonsgrense, avbrudd og
konkurrerende publisering. Dette er kode-/kontraktbevis, ikke markedslæring.

PHYSICAL_COORDINATE_PRODUCER_REVIEW_001 sluttet med exit 0 og uendret kilde.
Ekte fullført preprocessing når produsentens avvisning av manglende målt
sampler. Ingen ekte koordinater, modellkjøring, fit, broker-kall eller
TEST-lesing ble utført. Faktisk kostnads-/indekskvalifisering, benchmark,
sampler-valg og fersk initial-/sluttmåling gjenstår. Det tidligere
avgrensede brokerspørsmålet er ubesvart; native trening er fortsatt stengt.
Uavhengig klargjøring av initialiserings- og målebindingene kan fortsette.
Ingen læring, kostnadsjustert edge eller train/serve-paritet er bevist.

Produsent: gx1/scripts/materialize_unified_exit_chronological_coordinates_v1.py.
CLI krever --design, --normalization-result, --labels-result, --selected-sampler
og --output-dir. Uten --publish utføres bare inputkontroll. Ekte kjøring skal
bindes i en egen plan og kjøres gjennom eksisterende capped producer; den er ikke startet.

Kildekontroll: PHYSICAL_COORDINATE_PRODUCER_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T035119774721Z.json
SHA256 4c4fd4de4efe394a03f590480e95c021d4c923f23b09a1f935631d386e246e9b.


## 02.10: fersk optimizer-/schedulertilstand og kildebundet initialmåling

To feil i fersk tilstand og gjenbruk av initialmåling er rettet.

Gjenoppretting kontrollerte tidligere tom optimizerhistorikk, men kunne
likevel laste andre parametergrupper, læringsrate eller scheduler-innstillinger.
EMA-/schedulerfeil kunne dessuten oppdages etter at modellen var endret.
Nå sammenlignes lagret optimizer, scheduler og EMA-metadata med de ferske
komponentene recipe-en faktisk konstruerte, før innlasting muterer noe.

V38-initialmålingen binder nå hele recipe-ens eksisterende kildeinventar,
inkludert eierens filhash og filidentitet. Det samme inventaret kreves før
avgrenset læring; en endret mål-/beregningsfunksjon kan ikke gjenbruke målingen
bare fordi hovedmodellens fil og vekter er like. Innføringen gjelder fysisk
v38; den historiske rutens eksisterende evidens beholdes. Ingen ny inventareier
eller alternativ læringssløyfe er lagt til.

149 fokuserte tester består; tre eksisterende avledede målkombinasjoner
utenfor TRAIN-only-omfanget er fortsatt deklarert utelatt. Ni korrupte
optimizer-/scheduler-/EMA-varianter avvises uten endring av modell, optimizer,
scheduler, EMA eller RNG. Fysisk initial-/sluttmåling lagrer kildebindingen,
og læringsadmission kaller kontrollen før videre behandling av kohortene.

FRESH_STATE_SOURCE_REVIEW_001 sluttet med exit 0 og uendret kilde. Kontroll-
harnessen bruker repoets faktiske native inventar med 159 oppføringer,
og avviser en endret referansekilde med NATIVE_PREFIX_MEASUREMENT_SOURCE_CHANGED.
Modusmarkøren i denne isolerte kontrollen er syntetisk; dette er ikke en
kjørbar recipe eller full native admission. Tester av restore og sesjon bruker
små syntetiske modeller. Ingen ekte data er målt gjennom modellen.

Faktisk kostnads-/indekskvalifisering, benchmark, sampler-valg, koordinater og
fersk native initialisering gjenstår. Det tidligere avgrensede brokerspørsmålet
er ubesvart. Før mer kodearbeid må neste konkrete blokkering påvises; beståtte
tester og fullført preprocessing gjenbrukes. Native trening er stengt og TEST
forseglet. Læring, kostnadsjustert edge og train/serve-paritet er ikke bevist.

Kildekontroll: FRESH_STATE_SOURCE_REVIEW_001/EVENTS/SOURCE_REVIEW_20261002T040826067925Z.json
SHA256 50683882ad77e1b737e7b72c013ca97f3849564e6303ae057dcda3b47abe6539.

## Lesekontroll uttrykkelig godkjent — 02.10.2026

Operatøren godkjente 02.10.2026 den tidligere klargjorte lesekontrollen:
«Ja kjør lesekall». Tillatelsen gjelder bare COST_TERMS_REVALIDATION_001,
maksimalt ett GET for OANDA practice-kontovilkår og ett GET for XAUUSD-vilkår.
Den hash-bundne planen og operatoren gjenbrukes. Ingen retry, redirect,
transaksjonsoppslag, ordre, handel eller spending. Trening og TEST er stengt.

Neste steg er én capped audit-kjøring og kontroll av renset terminal evidens.
Godkjenningen er ennå ikke brukt; ingen nye broker-kall er utført.

## Lesekontroll fullført; målte finansieringsvilkår endret — 02.10.2026

Den autoriserte engangskontrollen er fullført med exit 0: nøyaktig to
GET-kall til OANDA practice, null ordre og null nye transaksjonsoppslag.
De 258 historiske fyllene og finansieringsobservasjonene er bevart.
Kontovilkårene er uendret. Instrumentets finansieringsrate endret seg fra
-0,054 til -0,0569 for LONG og fra +0,0282 til +0,0323 for SHORT.
GSLO-minsteavstand endret seg fra 5,32 til 1,5; fryst no-GSLO-policy består.

Tillatelsen er brukt opp. Ingen flere broker-kall er autorisert.
Kostnadseieren avviser de nye vilkårene fordi gammel finansieringsrate er
hardkodet i produsent, policyvalidator og parameterautoritet. Neste rettelse
skal binde finansieringskostnaden til den eksakte nye vilkårsevidensen med
samme vedtatte metode: negative renter blir kostnad, positive kreditter
klippes til null, og faktisk veggklokketid brukes. Ingen endring av slippage,
kommisjonsbevis, risiko, mål, TEST eller trening. Ny prospektiv policy er
ikke historisk kostnadsfasit eller lønnsomhetsbevis.

result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/COST_TERMS_REVALIDATION_001/EVENTS/RESULT_20261002T045413208824Z.json
SHA256: 8426c51ebf69481a0a78659889bcbfc81d61910f44897a73d63663bd624d061a
terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/COST_TERMS_REVALIDATION_001/EVENTS/TERMINAL_20261002T045413225232Z.json
SHA256: 3f31510bfe224c6a27a92fa2d274ad22da423cbe53d3852d1b2eea406e14e8e4

## Dagens vilkår bundet i ny prospektiv kostnadspolicy — 02.10.2026

De to godkjente OANDA practice-lesekallene er fullført. Ingen ordre,
transaksjonsoppslag eller nye broker-kall er utført etter engangskontrollen.
258 historiske fyll og lagrede finansieringsobservasjoner er bevart.

Dagens instrumentvilkår er -0,0569 for LONG og +0,0323 for SHORT.
Kostnadskoden hadde hardkodet de gamle ratene. Eksisterende eier binder nå
rater til hvert artefakts eksakte broker-evidens og bruker samme metode:
negative renter belastes, gunstige kreditter klippes til null.
Policy, komponentfakta og parameterautoritet bruker samme bundne tall.
Gamle artefakter kontrolleres fortsatt mot sine opprinnelige kilder.

CURRENT_TERMS_POLICY_001 publiserte en ny, separat prospektiv kostnadspolicy
for 01.06.2011–01.07.2026. Årlig finansieringskostnad er 0,0569 LONG og
0 SHORT. Kommisjonsgrunnlag, slippage 2 bps per utførelse, sensitivitet
1/2/4 bps, no-GSLO-policy og null ekstra risikostraff er uendret.
Policyen er forhåndsbundet før native måling og bruker ekte før-TEST-quotes.

36 fokuserte tester består. Kontrollene dekker endrede fortegn/rater,
avvisning av gamle eller underrapporterte kostnader, atomisk publisering
og gammel evidens. Gamle ekte policybytes og den nye publiserte autoriteten
er strengt lest med kildeverifisering. Kilde var uendret under kontrollen.
Etter de to godkjente GET-kallene var nettverk sperret i policykontrollen.

Dette kvalifiserer en prospektiv beregningspolicy, ikke historisk
finansieringsfasit eller økonomisk edge. Neste avhengighet er nye faktiske
TRAIN/VAL-økonomi-/indeksbindinger til denne autoriteten, deretter målt
sampler, koordinater og fersk initialmåling. Ingen indeksbygg eller
modellmåling er gjort i denne bølgen. Native trening og TEST er stengt;
engangs broker-tillatelsen er brukt opp.

result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/CURRENT_TERMS_POLICY_001/EVENTS/RESULT_20261002T050033032316Z.json
SHA256: 7cff69e025586e4b06fc8889713ce51509c97936972c29e859428f0b7ebf0498
terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/CURRENT_TERMS_POLICY_001/EVENTS/TERMINAL_20261002T050033050080Z.json
SHA256: c038e2b2e11dc8ecee3bb0e6a07be80d9b5c3f567dd1101a12127f567a13097b

## Faktisk økonomi-/indeksbygg klargjort — 02.10.2026

Faktiske TRAIN/VAL-økonomibindinger og native indekser er klargjort i
NATIVE_ECONOMIC_INDEX_001. Ferdige input og ny prospektiv kostnadsautoritet
gjenbrukes. Kapitalkravet forblir den eksisterende fryste 10%-metoden;
identitet bindes til faktisk TRAIN-parquet, fryste TRAIN-rad-ID-er og
eksisterende lineage fra preprocessing. Ingen ny fit eller rateseleksjon.

En konkret feil i indeksbyggeren er rettet før kjøring: failed staging
skal bevares for retention-eieren. Feilen er reprodusert, og 30 fokuserte
indeks-/økonomitester består. Alle tre publiseringsruter i samme eier
bevarer nå feilede forsøk. Selve indeksene er ennå ikke bygget.

Planen bruker producer 10G/swap512M, CPU0–7 og én numerisk tråd.
Ingen modell, optimizer, normaliseringsfit, broker, TEST eller handel.
Kilde fryses under kjøringen; ingen relansering av forbrukt plan.

plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NATIVE_ECONOMIC_INDEX_001/PLAN.json
SHA256: c55a035824f5379146ef06bf593938defdc4a4a422cdb1e40c3a986ad2d63763
operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NATIVE_ECONOMIC_INDEX_001/OPERATOR.py
SHA256: 2623968215f39279f242a746243d1f0d52ea3d8055f85a97f2ff0500805fb045


## Indekser verifisert; faktisk featurekilde koblet — 02.10.2026

Indeksbygget NATIVE_ECONOMIC_INDEX_001 er fullført med exit 0 og uendret
kilde. Uavhengig kontroll bekrefter filhasher, manifest, alle 652 552 TRAIN-
og 70 880 VAL-rader, eksakt foreldreindeks og ingen økonomisk sluttgrense.
Nye kostnadsbindinger gjenbrukes; ingen modell eller optimizer er kjørt.

Den neste forbrukeren hadde en konkret mismatch: gammel lifecycle-leser
brukte den tidligere M1-kilden, mens nye indekser og normalisering bruker
komplett før-TEST-M1. Faktiske tidsstempler passer ikke ved indeksens
radposisjoner. Benchmark ble derfor ikke startet med feil kilde.

Eksisterende lifecycle-eier kan nå laste pris og ferdig featureflate fra
indeksens bundne preprocessing. Både fysisk native komponentbygging og
benchmark bruker denne samme ruten. Historiske lifecycle-episoder beholdes;
ingen episodefasit fabrikeres eller ny featureberegning/normalisering gjøres.
79 fokuserte tester består, inkludert koblingen i begge forbrukere og
avvisning av endrede filidentiteter, TEST-split og forskjøvede klokker.
De nye lastetestene bruker syntetiske kilder; reell kildeinnlasting er ennå
ikke kvalifisert.

Neste jobb er INDEX_FEATURE_SOURCE_REVIEW_001, producer 10G/swap512M, for
streng innlasting av de faktiske TRAIN-/VAL-kildene. Kilde fryses under
jobben. Deretter gjenstår målt sampler, koordinater og fersk initialmåling.
Ingen trening, refit, broker-kall, TEST eller live/paper er åpnet.
Læring, generalisering, økonomisk edge og train/serve-paritet er ubevist.

index_result: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NATIVE_ECONOMIC_INDEX_001/EVENTS/RESULT_20261002T060119720832Z.json
SHA256: 1fe10ce8874911f80d9c0d13ba2ba90e3da4d2c2b80fc5b416ef78011ce0a32d
index_terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NATIVE_ECONOMIC_INDEX_001/EVENTS/TERMINAL_20261002T060119740507Z.json
SHA256: 626db8da298880980459d6b04eebe9e7b905163ac75600ef54d74de1fefe01c6
publication_review: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/NATIVE_ECONOMIC_INDEX_001/EVENTS/PUBLICATION_REVIEW_20261002T060703917118Z.json
SHA256: 0a192198d9cbf96d0325939d68b48b2f40c7f08ff01d754959952568512c685f
source_review_plan: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/PLAN.json
SHA256: 094f0cb2bed71d2653328185e51642765b71c74f6401c2b0ef83eac0e2d55848
source_review_operator: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/OPERATOR.py
SHA256: 9b53f72571492c7f95305bb4e7995c44e5ebeb0a19f3e00888c8688f18b2a604

## Featurekildereview fullført; samplerbenchmark er neste - 05.10.2026

Restartkontrollen fant at status-JSON fortsatt omtalte
INDEX_FEATURE_SOURCE_REVIEW_001 som planlagt, selv om runtime hadde resultat
og terminal fra 02.10. Jobben er fullført med exit 0, uendret kilde, null
TEST, 254 felt og eksakte klokker for 652 552 TRAIN- og 70 880 VAL-Entry-rader.
Ingen modellforwards, optimizersteg eller normaliseringsfit ble kjørt.
Ingen sampler ble valgt.

Resultat: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/EVENTS/RESULT_20261002T062117624344Z.json
SHA256: f7fbe63ad88230c6a7da65d76384f60276c17ad12895bd3ce00c5b4388cbf23b

Terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/INDEX_FEATURE_SOURCE_REVIEW_001/EVENTS/TERMINAL_20261002T062117642659Z.json
SHA256: 2ffd811ee77d821dda0324ccca1889e4086bc004195663d75a10aa8803758fac

Planen er konsumert. Neste arbeid er én forhåndsregistrert workload-matchet
samplerbenchmark. Se RESTART_POINT_20261005.md.
