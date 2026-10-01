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
- Pilotens Entry-admission krever pretest_test_guard. V38 har i stedet en
  ekte, completion-bundet TEST-forsegling med eksisterende metadata-validator.
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

## Neste konkrete grense

Kvalifiser M1-kilden og koble inn eksisterende validator for den faktiske
TEST-seal-hendelsen. Ikke ommerk native-pair-data som det gamle pre-TEST-formatet.
Deretter må øvrige eksisterende eiere bruke samme bundne vinduer og separate
fysiske koordinater: Entry-child, M1-views, summary-fit, normalisering,
native indeks/trening/måling og frossen TARGET-funksjon. Call-site-sveip har
funnet kalenderkrav også der; disse er dokumenterte ufullførte migreringer,
ikke fjernet ved å lempe den første porten. Normer/indekser fra v37 kan ikke
brukes med 254 v38-felt.

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
