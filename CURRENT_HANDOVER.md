# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og avtalt omfang

TA_DIRECTION_EDGE_CAUSE_ANCHOR_SPAN_REPLACEMENT_ADMITTED.
Målet er fortsatt å avklare og rette målte svakheter i indikatorrepresentasjon,
sammensetting og retningslæring. Korte runder er bevisst feilsøking.
Full kvalifisert TRAIN-historikk og 30–40 epoker er senere skalering.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt.
GC er på pause. Åtte genuine familier, alle fem TF og D1-konteksten beholdes.

Brukerens fokuskorreksjon 12.11 er bundet i execution_contract_20261112:
Ett spørsmål: forbedrer de to trendlinje-spanfeltene M5-rangering og kalibrering?
Samme 4096 TRAIN-rader, 256 updates, B16, rekkefølge, queries, loss, AdamW og INITIAL-RNG.
Mål M5 S/F AUC, Brier/log-loss og behold totaltap/alle TF i vurderingen.
Ingen generisk utbygging, nye diagnosekampanjer eller fullhistorikk-utvidelse.
Ren konkret preflight ble oppnådd etter 1730,63 sekunder innen 60-minuttersrammen.

## Siste faktiske oppstart: FAILED og konsumert

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_001.
Kilde efec26e9; recipe f7ba828f; Windows boot 643.
RESULT 339de7a4 og TERMINAL.json dokumenterer feil før første forward/update:
TIME_QUERY_FOLD1_FIT_STATE_ANCHOR_SPAN_INITIAL_SOURCE_CONTRACTS.
Én modell konstruert, ett INITIAL-checkpoint lest, seks input-sidecars dekodet.
Null forwards, optimizer-konstruksjoner, backward, updates og checkpoint-skrivinger.
Ingen fersk faktisk INITIAL eller prediktiv sammenligning ble produsert.
Kontrolleren avsluttet; GX1AnchorSpanShortLearning001 er deaktivert.
Planen er FAILED/CONSUMED, remaining_invocations=0 og relaunch_allowed=false.
001 er stengt. Global training/native_training er true bare for eksakt erstatning002.
002 er admittert etter ren preflight; faktisk jobb er ennå ikke startet.

Terminalens source_unchanged=false bevares. Separat git-observasjon viste ren
kilde efec26e9 etter terminalen og før rettelsen; terminalflagget er ikke relabellet.
Oppstartsoppgavens UNC-sti ble rettet til eksisterende lokale C:-stagingform før
launch; Windows-prinsipalen ble verifisert som Andre. Dette endret ingen modellkode.

## Minste rettelse etter konkret feil

[Samlet bevis](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json), felt anchor_span_short_learning.
ROOT/IMPORT_CONTRACT_MISMATCH.json avgrenser to feil i INITIAL-importkontrollen:
Feature-view binder S-delmodellen, mens importen krevde identiteten til hele S/F-modellen.
Treningsoppsettets hash bruker campaign-formatets avsluttende linjeskift; importen
brukte modellkontraktenes format uten linjeskift. De numeriske verdiene var identiske.
Rettelsen i market_state_time_query_fold1_fit_state_v1.py gjelder bare disse bindingene.
Syntetiske fixtures hadde gjentatt de samme feilene og er korrigert mot eksisterende eiere.
Korrigert positiv test feilet med samme SOURCE_CONTRACTS før rettelsen.
Etter rettelsen: 24 berørte tilfeller PASS, inkludert avvisning av begge feilvariantene.
Originale red/green-kvitteringer og IMPORT_REPAIR_REVIEW.json ligger i ROOT.
Dette er importkvalifisering; faktisk baseline og læringsvirkning er fortsatt ukjent.

## Gjenbrukbare ferdige bevis

[Span-reparasjon](docs/TA_TRENDLINE_ANCHOR_SPAN_REPAIR.json): andre ankertid manglet;
første ankertid var allerede implisitt i alder. To kausale native-bar-spans er lagt til.
Et syntetisk vitne beviste representasjonstap; frekvens og prediktiv kostnad er ukjent.
TA-bredder er 317/132/1/255; rå signal/ctx/cat er 2727/2542/1.
[Fysisk input](docs/TA_TRENDLINE_ANCHOR_SPAN_INPUTS_REVIEW.json) er COMPLETE/CONSUMED:
alle 31 gamle felt byteeksakte på 73118 flaterader; seks kvalifiserte små span-sidecars.
[Normalisering](docs/TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_REVIEW.json) er COMPLETE/CONSUMED:
seks fit for bare de to nye feltene; 45724 gamle statistikkverdier byteeksakte.
Eksisterende arrays, caches, targets, normalisering og gamle checkpoints bevares.

[Siste fullførte læring](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json) brukte
4096 TRAIN-rader / 256 updates, 1.–21. mai 2024, med gamle 31 trendlinjefelt.
M5 S/F AUC: 0,546185/0,546125 til 0,551352/0,553147.
S-Brier ble svakere, F-Brier bedre; totaltap 1,878003 til 1,884324.
Full paired avlesning er konsumert: sourceendringer hjalp M5-rangering samtidig
som andre kanaler ble overkonfidente. Ingen generell frys eller indikatorfjerning er begrunnet.
Mekanisk konsum av alle åtte familier/fem TF beviser ikke prediktiv familienytte.

## Neste nødvendige steg

Den ene avtalte læringssammenligningen er ikke gjennomført. Brukt001 relanseres aldri.
Erstatning002 er admittert med egen eksakt kilde-/policybinding etter rettelsen.
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_002.
Ren kilde9ddedc5e; RECIPE8af5fa09; PREFLIGHT798e20e1; prepared Windows boot643.
Faktisk provider-metadata, eksakt startup-avvisning under audit og3600/3900/3900-grenser PASS.
Materializer/controller og trygg fresh-boot-oppstart gjenstår; ingen ny kodebølge.
Samme kontrakt/runner gjenbrukes; kun engangsidentiteten og capped-path flyttes til002.
Fokusert scope-/budsjettest PASS; tre JSON-inputbindinger er forberedt uten array-/modelllesing.
De 24 beståtte importtestene gjenbrukes; importkoden er uendret fra rettelsen99605dc3.
Stående lokal0USD-bestilling i GX1_RULES gir autorisasjonen; gammel001-godkjenning gjenbrukes ikke.
Ingen automatisk forsøksserie: samme spørsmål, data, mål og budsjetter beholdes.
Før første update må alle 4096 ferske INITIAL-rader og 256 tap matche gammel aktivitet-INITIAL:
logits atol2e-6/rtol1e-6, tap atol5e-6, akser/targets eksakt.
Ved faktisk kjøring fryses kilde; stabil drift kontrolleres omtrent hvert 15.–30. minutt.
Senere generalisering, familienytte og økonomisk edge er fortsatt ubevist. Målet er aktivt.
