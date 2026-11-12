# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_ANCHOR_SPAN_COMPARISON_SOURCE_PREPARED.
Målet11.11 er aktivt: avklar og rett målte svakheter i indikatorrepresentasjon,
sammensetting og retningslæring. Korte runder er bevisst feilsøking, ikke kapasitetsdom.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.

Både input- og normaliseringskontrollen er COMPLETE/CONSUMED; ingen jobb står åpen.
Alle tidligere engangsplaner er konsumert. Ingen ny data-/modell-/treningsadmisjon.
Training/native_training/native_launch/full_epoch/full_val er false.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion stengt; GC på pause.
Ingen endring av Windows-treningsoppgaver.

Brukerens fokuskorreksjon12.11: støttearbeidet har blitt for omfattende i forhold til
direkte prediktiv måling. Generisk utbygging er stanset. De to konkrete koblingsfilene
entry_anchor_span_feature_inputs_v1.py og entry_anchor_span_feature_view_v1.py
er nå koblet inn i eksisterende binder/gather. 37unike fokuserte tilfeller er
dekket: først28PASS/9FAIL, deretter9rettedePASS; feilene gjaldt syntetiske
kildereferanser og DataFrame/backing-deling. Originale testkvitteringer beholdt i
MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_001/BRIDGE_TESTS*.
Dette er metadata/syntetikk, ikke faktisk fysisk gather eller modellbaseline.
Ingen modell/trening er startet. NEXT_RUN_POLICY sin execution_contract_20261112
låser forsøket til ett spørsmål, én endring og én fast kort sammenligning.
Neste implementeringskontroll skal gi konkret forsøks-preflight eller eksakt
blokkering; ingen ny forberedende arbeidsstrøm åpnes.
Forberedelsen har60minutter effektivt arbeid fra neste implementeringsstart;
ved grensen rapporteres konkret blokkering og nødvendig avvik før videre utvidelse.
Gjør bare nødvendig kobling og matchet fersk INITIAL for én fast kort sammenligning
(samme4096TRAIN-rader/256updates og eksisterende mål/oppsett der gjenbruk er gyldig).
Mål M5-rangering og sannsynlighetskalibrering; behold totaltap/alleTF i vurderingen.
Ingen flere generiske støtteverktøy eller gjentatte diagnosekampanjer. Hvis spanfeltene
ikke gir målt gevinst, gå tilbake til det observerte sammensettings-/kalibreringsproblemet.
Konkret kvalifisert plan kreves fortsatt før ny kjøring; teknisk PASS er ikke edge.

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
Inputkontrollen brukte ingen normaliseringsfit, targetverdier, modell, checkpoint eller trening.
Eksisterende normalisering/checkpoints er ikke relabellet; gammel baseline er ikke ny baseline.

[Normaliseringsutvidelsen](docs/TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_REVIEW.json)
er COMPLETE/CONSUMED, exit0. ROOT
/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001.
Kilde8582cbfb uendret; PLAN07653339/RESULT4476c63b/TERMINAL568bed0a.
Worker6,77s/controller7,65s; capped audit4GiB/512MiB,CPU0–7,pids64,én tråd.
96unike fokuserte tester er dekket: først93PASS/3FAIL, deretter8berørte retester PASS.
De3feilene var historiske fixtures som matet nye2727navn til gamle2725profiler;
fixtures krever nå eksakt gammel navneSHA. Gamle profiler og planer er uendret.
Obligatoriske model-native kontrakt-/repositorygater besto før commit.

Nøyaktig6fit-kall for2nye felter hver;0gamle felt refittet.
6532gamle flatefelt/45724statistikkverdier er bevart byteeksakt etter navn;
2542aliaser er remappet uten endrede felt, masker, domener eller støtte.
Ny units-profilaccae7df remapper bare signalens51value/mask-par etter navn;
ctx/GC-enheter og fit-prefix er uendret. Gammel fysisk lineage er eksplisitt forelder,
ikke relabellet cache. Proveniensen binder sidecarresultat8878d17a.
Alle nye observerte spanverdier bevarer streng rekkefølge gjennom asinh og gjenfinnes
eksakt som heltall etter invers. Største flyttallsavvik er under0,000010bar.
M5spans er sparsomme: center[0,0],scale[6,5], eksisterende median-positive-deviation.
Øvrige10nye felt bruker ordinær IQR; ingen ny binærregel, terskel, klipping eller maskering.

Artefakter: EXTENDED_BASE_NORMALIZATION.json SHA71e201b6/contract58e9e9bb;
TA_INPUT_NORMALIZATION.json SHA02eca0cc/contract6c882086;
NEW_FIELD_STATISTICS.json SHA06d2a112; NEW_UNITS_PROFILE.json SHA050f93ad.
Eksisterende fullTA-projeksjon validerer signal317/ctx132/MTF255.
Uavhengig JSON-/filhash-/statistikkbytekontroll PASS, uten ny fit eller array-dekoding.

Faktisk feature-view/gather må nå bindes til de uendrede foreldrearrayene pluss
de kvalifiserte span-sidecarene og ny normalisering. Ingen full rebuild er nødvendig.
Ny komplett modellinput og fersk faktisk initialbaseline er fortsatt ukvalifisert;
ingen modell, checkpoint, targetverdier eller trening ble brukt i normaliseringsjobben.
Fersk faktisk baseline kreves før en avgrenset læringssammenligning.
Ingen ny numerisk plan er innrømmet. Fullhistorikk/30–40epoker er senere skalering.
Senere generalisering, familienytte, rettelsens prediktive gevinst og økonomisk edge
må fortsatt måles. Målet er aktivt.

## Avgrenset sammenligning klargjort12.11

Konkret gather og navnebasert INITIAL-import er koblet til eksisterende kort-runner.
14parametertensorer utvides;2314nye parametere settes0. Gamle lærte verdier
kopieres etter felt-/modulnavn; nye normaliserings-/rutingbuffere beholdes.
En manglende remapping av persistent input_norm_alias_signal_indices ble
fanget i kildekontroll og rettet før faktisk modellbruk.
101fokuserte initial/runner/campaign-tilfeller PASS;22berørte importtilfeller
PASS etter aliasrettelsen (én ny avvisningsgrense). Tidligere gatherbevis gjenbrukes.
Metadata-preflight PASS; ingen markedsarray, checkpoint eller modell dekodet.
Aktuell ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_001.
Dette er samme ene forsøk i execution_contract_20261112, ingen ny forsøksserie.
Runtime skal måle fersk INITIAL på alle4096rader før første backward/update.
Toleranser fastsatt før kjøring: logits atol2e-6/rtol1e-6, batchtap atol5e-6;
akser/targets må være byteeksakte. Etterpå samme256updates, alle4096FINAL-rader.
Opprinnelig aktivitet-INITIAL og RNG gjenbrukes som kilde, med fersk tom AdamW.
Én ny scope i eksisterende campaign/capped-run/materializer; samme vakter og grenser.
Ren kilde, obligatoriske gater og konkret policyadmisjon gjenstår før launch.
Ingen ny modellbaseline, trening eller prediktiv gevinst er påstått.
