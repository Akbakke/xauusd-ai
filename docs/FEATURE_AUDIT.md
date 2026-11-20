# Aktuell feature- og konsumrevisjon

Verifisert 20.11.2026 i `/home/andre2/src/GX1_CURRENT`, `work/gx1-current`.
Dette dokumentet oppsummerer ferdige kontroller. `NEXT_RUN_POLICY.json` er
arbeidsstatus; `CURRENT_HANDOVER.md` er overleveringen. En teknisk PASS beviser
ikke bedre retning, generalisering eller økonomisk edge.

## Dekning og modellkobling

CURRENT har 1402 Python-filer og 20251 funksjonsdefinisjoner. Hele kilden er
AST-inventert med import-/kallgraf og dynamiske koblingskandidater. Featureeierne
har 396 funksjoner i 24 moduler. Dette er en eksplisitt nevner for gjennomgangen,
ikke påstand om bevist oppførsel for hver historisk funksjon.

| Modellflate | Feltposter |
|---|---:|
| Signal | 318 |
| Kontinuerlig kontekst | 132 |
| Kategorisk kontekst, session_id | 1 |
| M5/M15/H1/H4/D1, 256 hver | 1280 |
| Sum | 1731 |

Alle signal-/kontekst-/MTF-indekser rutes én gang til de åtte spesialistene.
Alle 132 tidsaliaser beholdes; den generiske snapshotbanen bruker øvrige 186
signalfelt. Alle 61 D1/makrofelt beholdes både i signal og kontekst via
`session_regime_encoder`. Den niende kontekstfamilien er ikke en niende
spesialistencoder. Ingen utilsiktet urutet aktiv feature ble funnet. GC/MBP er
bevart separat og inngår ikke i gjeldende TA-profil.

## Uavhengige tall og klokker

Bevisrot: `/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001`.

- `FULL_NATIVE_NUMERIC001`: alle 256 felt på alle fem tidsrammer, samt lokale
  ikke-makrofelt; 1609 råfeltposter. Hele beholdte rader: M5 389579, M15 129916,
  H1 32575, H4 8590, D1 1669. Finite-/ukjentmasker, diskrete domener, kausale
  koblinger, alle 27 arrayheaders/bytebindinger og faktisk normaliseringsaritmetikk
  er kontrollert. Fullført exit0 på 5078,675 sekunder.
- `FULL_MACRO_NUMERIC001`: 61 uavhengig beregnede makroformler og erklært
  tilgjengelighet, 122 feltposter/47528638 celler over 389579 lokale rader.
  Samlet er alle 1731 råfeltposter kvalifisert. Hele de eksplisitt bundne
  offentlige kildecontainerne ble hashet, også bytes etter forseglingen;
  etterforseglede observasjoner/tilgjengelighet inngikk ikke i feltberegningen.
  Ingen native TEST-fil ble lest. En komplett historisk vendor-/revisjonsledger
  er fortsatt ikke bevist.
- `FROZEN_TRAIN_STATS003`: uavhengig rekonstruksjon av hele input-TRAIN før
  målmasker, vindusunioner, kvartiler/skala, domener og alias-eierskap. Alle
  1730 skalaer og float64-ekstrema er bitlike. 1729 sentre er bitlike; ett
  navngitt D1-center er numerisk identisk +0/-0, dokumentert av
  `D1_STATS_DIAG001`. Den uendrede lagrede representasjonen brukes til kontroll
  av faktisk transform. Ingen fit-API, parameteroppdatering eller nye
  modellparametre er brukt. Originale mislykkede auditkvitteringer er bevart.

TRAIN har 354550 issues i 2021–2025; utviklings-VAL har 34934 issues i første
halvår 2026. Native TEST fra 01.07.2026 er forseglet. Retningsruten bruker native
M5 med lukket M15/H1/H4/D1, uten M1-data eller M1-metadata.

## Faktisk kjøring og ubeviste grenser

Den gjenbrukte ferske initialbaselinen kjørte én modell og 6086 forwards over
hele TRAIN/utviklings-VAL, uten optimizersteg. Initiallogits var lik ny TRAIN-
prior. Dette beviser kjøring gjennom profilen; nullstilte sluttvekter gjør det
uegnet som per-felt sensitivitetsbevis. Den kvalifiserte mekanikkprøven beviser
kald bitlik gjenopptakelse, forventet auxiliary detach og positiv samlet M5-
backbone-gradient etter første head-oppdatering. Den beviser ikke nytte eller
gradient fra hvert felt. Alle 53 bundne aktive kildehashene er uendret.

Konsumkart og bevisgrenser finnes i `CURRENT_FEATURE_CONSUMPTION_CLAIMS001.json`;
komprimerte funksjons-/avhengighetsregistre finnes via
`CODE_AND_FEATURE_INVENTORY_ARCHIVE.json` under bevisroten. Datert
`FEATURE_AUDIT_INVENTORY_20261017.json` beholdes uendret som historisk kildebevis,
med Git-gjenoppretting for senere pensjonerte kilder. Det er ikke dagens inventar.

Ingen per-felt økonomisk nytte, uavhengig senere generalisering, handelsregel,
live/paper-resultat eller profitt følger av disse kontrollene. Ukjente utfall og
0,3 ATR som nettoendringsproxy beholder sine erklærte begrensninger.
