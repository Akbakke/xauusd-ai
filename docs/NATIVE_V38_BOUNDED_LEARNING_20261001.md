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

## Neste konkrete grense

Bind fysisk TRAIN-normalisering, konkrete TRAIN-/kontrollrader, fersk
initialtilstand og faktisk økonomisk måleregel før én avgrenset native måling.
Eksisterende prefix-eier krever felles parent-datasett, senere kontroll og
full tidsstøtte før cutoff for både targets og Exit-states. En tilfeldig
tidlig radliste er ikke nok. Feature-/policyfit må også være før kontrollen.
Gjenbruk eksisterende eiere og tidligere låst forsøksbudsjett; gamle
normaliserings-, indeks-, metadata- og modellartefakter er ikke v38.

Kildegjennomgangen bekrefter at native initial-/sluttmåling eksplisitt har
economic_rollout=false. Porteføljeøkonomi med alle valgte handler, åpne
posisjoner og kapital-/overlappskontroll må derfor bindes separat. AVWAP-vekt
er prisoppdateringsaktivitet, ikke omsatt volum. Source review ligger i runtime
SOURCE_REVIEW_20261001T153932Z.json. Ingen ny produksjonskodefeil ble bekreftet
i dette avgrensede utsnittet; dette er ikke en full revisjon av all aktiv kode.

Normalisering og native trening er fortsatt stengt inntil konkret scope er
bundet. Teknisk input-PASS er ikke læring eller lønnsomhet. Makro, senere
uavhengig generalisering, train/serve-paritet og offline driftskvalifisering
er fortsatt separate ufullførte delmål.
