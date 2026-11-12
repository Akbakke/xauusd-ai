# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_TRENDLINE_ANCHOR_SPAN_RAW_INPUTS_QUALIFIED_NORMALIZATION_PENDING.
Målet11.11 er aktivt: avklar og rett målte svakheter i indikatorrepresentasjon,
sammensetting og retningslæring. Korte runder er bevisst feilsøking, ikke kapasitetsdom.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.

CPU-inputkontrollen er COMPLETE/CONSUMED, exit0; ingen kjøring står åpen.
Alle tidligere engangsplaner er konsumert. Ingen ny numerisk-/modell-/treningsplan
innrømmet. Training/native_training/native_launch/full_epoch/full_val er false.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion stengt; GC på pause.
Ingen endring av Windows-treningsoppgaver.
## Konkret rettelse: trendlinjens andre anker

[Reparasjon og kvitteringer](docs/TA_TRENDLINE_ANCHOR_SPAN_REPAIR.json).
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_IDENTITY_SOURCE_20261112_001.
Kildegrunnlag3af1a341da26473fb6ddb0791b5c2e3bed682450.

Første ankertid var allerede implisitt bevart: age=t−(anchor1+lookback).
Tidligere omtale av «manglende ankere» var derfor for grov. Andre ankertid manglet.
Et syntetisk vitne viste ulike ankerpar(10,30)/(10,50), men identiske verdier i alle
31publiserte felt, inkludert projeksjon, helning, alder og antall berøringer.
Dette beviser et informasjonstap i slotrepresentasjonen; frekvens og prediktiv
kostnad på markedstape er fortsatt UNKNOWN. Hele OHLC-historien påstås ikke identisk.

Minste rettelse er implementert i eksisterende trendline_registry_v1.py:
- geomline_above_anchor_span_bars
- geomline_below_anchor_span_bars

Span=anchor2−anchor1 i native barer. Sammen med eksisterende alder kan begge ankertidene
rekonstrueres relativt til raden. Bare allerede bekreftede pivoter brukes. Ingen ny
terskel, ID-allokering, carry-state, slotvelger eller livssyklus. Null ved fravær
skilles av eksisterende active_count; warmupNaN bevares. Span er uendret ved berøring
og rolleflip. Full internline_id, eksplisitt byttemarkør og ankerpriser er ikke lagt til.

Den felles produsenten og navnetuplene fører feltene til lokalM5 og alle5nativeTF,
med chart_geometry_encoder som eier. Alle8genuine familier,5TF og makrokontekst beholdes.
RegisterkontraktV7→V8; gammel carry avvises.33registerfelt, tidligere31 bevares.
TAinput signal315→317, ctx_cont132, ctx_cat1, hverMTF253→255.
Rå overflate signal2725→2727/ctx2542/cat1; chartfamilien MTF39→41.
Registry-navneSHA badbb1f7, MTF-navneSHA ec41b616, fullTAprofil5332e582.
Eksakte breddevakter i compact/geometry/momentum/tabular er oppdatert; ingen vakt svekket.
Gamle bundne051/source050-/auditplaner er ikke relabellet til ny semantikk.

## Hva som er verifisert

SOURCE_QUALIFICATION.json samler kilde-/testbindingene; kun syntetisk kvalifisering.
LEGACY_FIELD_PARITY.json sammenligner V7fraGit med V8: alle31opprinnelige felt er
float32-byteeksakte på1340syntetiske rader(1200randomwalk,70hold,70fail).
Begge slotter skiller nå de kolliderende ankerparene. Promotion/touch/rolleflip,
prefixkausalitet, eksisterende chunk/one-shot og V7carry-avvisning er kontrollert.
Lokal rå/aligned pakking og alle5faktiske TF-vindusslicere bevarer felt og lukket-bar-grense.
FullTAutvalg, syntetisk normaliseringsprojeksjon, gather og factorybinding er kontrollert.
Factory-/dualtestene bruker eksisterende testdobler; ingen faktisk ny modellbaseline påstås.

185unike fokuserte tilfeller er dekket på tvers av avgrensede runder:
171først(168PASS/3FAIL), deretter11PASS(3rettede+8berørte nye), deretter5TAtilfeller
(4PASS/1FAIL) og2PASS(1rettet+1søstertilfelle). Beståtte tilfeller ble gjenbrukt.
Feilene var gamle feltbredder og en foreldet factory-testdobbel. Doblene aksepterer
nå eksplisitt de tre eksisterende decoderflaggene og krever false; produksjonsfactory
uendret. Originale feilkvitteringer beholdt. Audit4GiB/512MiB,CPU0–7/pids64,1tråd.
Uavhengig kildekontroll er bundet til immutable kvalifiseringskvittering i
INDEPENDENT_SOURCE_REVIEW_002.json; første metadatakvitterings hashsyklus er korrigert
uten å endre kilde eller testresultater.

## Tidligere læringsfunn som fortsatt gjelder

[Samlet analyse](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json) og
[siste faktiske læring](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json).
Den faktiske læringsrunden brukte4096TRAIN-rader/256updates,1.–21.mai2024,
med separateTFaktivitetshoder. RESULTaca65dc4, kilde7dcddb37. Den brukte den gamle
31felt-overflaten; nye spanfelt er ikke målt i en faktisk modell eller læringsrunde ennå.

| M5 side, samme brukte TRAIN | INITIAL | Aktivitetshoder FINAL |
| --- | ---: | ---: |
| S-AUC | 0,546185 | 0,551352 |
| F-AUC | 0,546125 | 0,553147 |
| S-Brier | 0,247999 | 0,248027 |
| F-Brier | 0,248788 | 0,248218 |

AUC måler rangering. Ingen entydig M5gevinst mot originalshort-final ble vist.
Aktivitetsregresjon87,23% mindre enn original var et tapsregnskap, ingen87%edgegevinst.
Totaltap1,878003→1,884324. Alle8familier/5TF ble tidligere bekreftet konsumert;
mekanisk tilkobling og parameteroppdateringer er ikke bevist familienytte.

Full paired avlesning av begge checkpointene er COMPLETE/CONSUMED:
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_BOUNDARIES_20261112_001.
RESULT95f1fc32/TERMINAL87551f40/TRAIN_FUSIONba1ea384, kildef170a91c uendret.
Alle4096rader/endepunkt, originale outputs/batchtap og tidligere INITIALh0 byteeksakte.
PAIRED_BOUNDARY_REVIEW893e1962 og M5_RANK_REVIEWcf7716ff viste at sourceendringene
hjelper M5rangering samtidig som de bidrar til overkonfidens i andre kanaler.
S/D1FLAT Paktivitet88,49→93,35%; F/M15-H1 PUP ender69,94/70,88% mot52,08/53,97%observert.
Ingen generell frys, indikatorfjerning eller valgt endpointblanding er begrunnet.
Ingen generell konstantinput-/normaliseringsfeil eller mean+last-kansellering ble vist.

## Ikke gjort og neste steg

[Fysisk inputkontroll](docs/TA_TRENDLINE_ANCHOR_SPAN_INPUTS_REVIEW.json) er COMPLETE:
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001.
Kilde db667784 uendret; RESULT8878d17a/TERMINAL9414bc5d. Plan88916106 er konsumert.
21fokuserte tester, import og ren preflight PASS; capped audit4GiB/512MiB,CPU0–7,
én tråd,pids64. Worker276,73s/controller277,20s; peakRSS1,30GB.

Alle31gamle trendlinjefelt er float32-byteeksakte på73118fysiske rader:
lokal29782, M5/M15/H1/H4/D1=29702/9959/2571/744/360.
Det er2266658sammenlignede gamle feltverdier uten avvik. Alle seks rowhashes og fulle
availability-clock-hasher matcher original fold1-normalisering e24a2b8d.
To nye felt per flate gir146236nye verdier i seks små sidecars; originale caches urørt.
Spanfeltene er finite, heltallige, positive ved aktiv linje og0ved fravær;
rekonstruerte første/andre ankere er kausale. Begge sider varierer på alle seks flater.
Lokal replay brukte original2010-enriched opprinnelse; MTF brukte2009-native,
begge med frosne parametre og fersk state. Bare nødvendig registryhistorikk er replayet.

Råfeltene er kvalifisert; den nye komplette modellinputen er ikke ferdig bundet.
Ingen normaliseringsfit, targetverdier, modell, checkpoint eller trening i denne runden.
Eksisterende normalisering/checkpoints er ikke relabellet; gammel baseline er ikke ny baseline.

Neste steg er navnebundet normaliseringsutvidelse: tilpass kun de12nye feltstatistikkene
på disse seks populasjonene, behold alle gamle statistikker og remapp deklarerte
enheter eksplisitt. Ny feature-view/gather og fersk faktisk initialbaseline kreves før
kort læringssammenligning. Ingen ny numerisk plan er innrømmet nå.
Fullhistorikk/30–40epoker er senere skalering. Senere generalisering, familienytte,
rettelsens prediktive gevinst og økonomisk edge må fortsatt måles. Målet er aktivt.
