# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og neste retning

TA_DIRECTION_EDGE_CAUSE_FIXED_BENCHMARK_INPUTS_COMPLETE.
Målet er fortsatt å avklare og rette svak retningslæring. Den avtalte sammenligningen er
FULL_OBJECTIVE mot M5_DIRECTION_ONLY, samme modell/genuine inputs/INITIAL/RNG/rekkefølge.
TRAIN4096 over mai–september2024 og senere4096 over oktober–desember2024 er låst.
Tre passeringer per arm; måling0/1/3, pass3 primært, ingen checkpointseleksjon.
M5-koeffisientene beholdes; tom M5-støtte gir cursorframgang uten optimizersteg.

M5-only tap og checkpointbinding samt separat senere-adapter er implementert;73tester besto.
Senere inputeier/metadata har42beståtte tester og faktisk preflight før engangsadmisjonen.
Fysisk jobb er COMPLETE/CONSUMED; ingen relaunch. Kilde304bf944 var uendret.
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_OBJECTIVE_LATER_INPUTS_20261112_001.
RESULTf5f8a0e6, TERMINALdde568a6, COVERAGE307f930b. Dekning binder FEATURE_VIEW2b15bcfe.
Alle31gamle felt på44114flaterader var byteeksakte:1367534celler, null avvik.
4096lokale og20480faktiskeMTF-vinduer besto. Tid313,26s, peakRSS1,315GB.
Kun to nye spanfelt ble produsert. Ingen normaliseringsfit, gammel cacheendring eller modellkjøring.

Neste konkrete arbeid: koble eksisterende paired restore/fit/eval til de to tapsvariantene,
den kvalifiserte senere-adapteren og de tre passeringene. Gjenbruk beståtte kontroller;
ingen ny inpututvidelse eller gradientdiagnostikk. Eksakt native admisjon/INITIAL og selve
læringssammenligningen gjenstår. Ingen prediktiv gevinst er målt i denne bølgen.
Global training/native_training er false. Root eier policy, launch og samlet kontroll.

Korte runder er bevisst feilsøking, ikke en dom over fullhistorikk-kapasitet.
Kvalifisert full TRAIN-historikk og 30–40 epoker er senere skalering.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Åtte genuine familier, alle fem TF og D1-konteksten beholdes.

## Ferdig kontroll: to trendlinje-spanfelt

[Samlet bevis](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json), anchor_span_matched_comparison.
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_002.
Kilde8c9c2303; RESULT0ddae87d; TERMINALf6f08a15; sammenligninga60178f7.
COMPLETE/CONSUMED, exit0, remaining_invocations=0, relaunch_allowed=false.
Windows boot644; kontroller avsluttet og GX1AnchorSpanShortLearning002 deaktivert.

Samme4096TRAIN-rader,256updates,B16,queries,targets,loss,AdamW og INITIAL-RNG som originalen.
Fersk INITIAL besto alle4096rader/256tap før første update. Akser/targets byteeksakte.
Største logitavvik9,54e-7; største tapsavvik2,39e-7, innen forhåndsfastsatte toleranser.
Én modell/optimizer,768forwards,256backward/updates; opprinnelige RNG-endepunkter matcher.
Worker1133,13s; fit425,04s. Kilden var uendret. Maskinvarevaktene besto.

| Samme brukte TRAIN | Gammel aktivitet-FINAL | Span-FINAL |
| --- | ---: | ---: |
| M5 S-AUC | 0,55135167 | 0,55135589 |
| M5 F-AUC | 0,55314706 | 0,55313224 |
| M5 S-Brier | 0,24802656 | 0,24802690 |
| M5 F-Brier | 0,24821835 | 0,24821891 |
| Totaltap | 1,88432418 | 1,88435934 |

M5-rangering endret seg svært lite i hver sin retning. Begge Brier/log-score ble svakere.
Side-Brier/log ble svakt dårligere i alle10task/TF-celler; aktivitetsmålene var blandede.
Totaltapets regresjon fra INITIAL1,87800258 består. Ingen konsistent prediktiv gevinst.
Fire lagrede TRAIN-captures ble skåret med eksisterende metode; gamle publiserte skårer
ble gjenskapt eksakt. Ingen ny modell, checkpoint-dekoding, forward eller fit i review.
Dette er brukte TRAIN-data; AUC er rangering, ikke handelsgevinst eller bevist senere edge.

## Rettelser og gjenbruk

001 feilet før første forward/update og er fortsatt konsumert.
To importfeil ble rettet i99605dc3: feature-view skal binde S-delmodellen, og recipe-hash
skal bruke sin eksisterende eiers linjeskiftformat. Korrigert test gjenskapte feilen først;
24berørte importtilfeller besto etter rettelsen. Fersk faktisk INITIAL er nå også kvalifisert.
002 gjenbrukte kontrakt/runner med ny engangsidentitet, etter egen ren preflight og admisjon.
Ingen gammel engangsplan ble relansert; gamle feilkvitteringer og checkpoints er bevart.

[Span-reparasjon](docs/TA_TRENDLINE_ANCHOR_SPAN_REPAIR.json): andre ankertid manglet;
første var allerede implisitt i alder. To kausale native-bar-spans er lagt til.
Det syntetiske representasjonstapet var reelt; denne korte sammenligningen viste ingen gevinst.
TA-bredder317/132/1/255; rå signal/ctx/cat2727/2542/1.
[Fysisk input](docs/TA_TRENDLINE_ANCHOR_SPAN_INPUTS_REVIEW.json) er konsumert:
31gamle felt byteeksakte på73118flaterader, seks kvalifiserte span-sidecars.
[Normalisering](docs/TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_REVIEW.json) er konsumert:
seks fit for bare to nye felt;45724gamle statistikkverdier byteeksakte.

[Tidligere læringsfunn](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json) og full paired
avlesning viste at source/globalcontext-endringer hjalp M5-rangering samtidig som andre
kanaler ble overkonfidente. Ingen generell frys eller indikatorfjerning er begrunnet.
Mekanisk konsum av alle familier/TF beviser ikke prediktiv familienytte.
Hovedårsaken, senere generalisering og økonomisk edge er fortsatt uavklart. Målet er aktivt.

Eksisterende batchrekkefølge-kontroll20261110_002 (RESULTb11a1932) brukte samme4096
fordelte TRAIN-rader/256B16-grupper per arm og512updates totalt. Fletting bedret
totaltap1,869982→1,848745, men forverret M5 S/F-Brier fra0,247566/0,248458
til0,247834/0,248634. Eldre INITIAL/hoder og annet TRAIN-utvalg begrenser overføring
til dagens løp. Ikke start en ny generell flettingskontroll; gjenbruk den målte avveiningen.
