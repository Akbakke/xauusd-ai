# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_FULL_PREFIX_LEARNING_ADMITTED_AWAITING_FRESH_BOOT. Edge-målet er uoppnådd.
Ingen native jobb er startet ved admisjon. Training/native er true bare for den nye
avgrensede planen; full_epoch/fullVAL er false. Gamle planer forblir konsumert. Eksisterende modellarkitektur, inputfelt, fem TF og tap er uendret.

Neste forsøk er nå kildekvalifisert: samme bef35-INITIAL og samme29574TRAIN-foreldre fra
mai–september2024 i to komplette, kronologiske gjennomganger. Alle rader beholdes, også
siste6 og tidligere736komplementrader.3698updates/59148presentasjoner; måling0/1/2 på
uendrede4096TRAIN/4096senere utviklingsrader. Pass2 er primær; ingen checkpointutvelgelse.
Dette isolerer mer repetisjon med dagens features/hoder; det tilfører ingen nye historiske
markedsperioder. INITIAL-backbone arver allerede to fulle pass (3698updates over resetgrenser).
Original FINAL→vektimport→R1 INITIAL er hashkontrollert; alternative FINAL-armer telles ikke.
Hoder har annen treningshistorikk.30–40epoker/fullhistorikk er fortsatt ikke åpnet.

Root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_LEARNING_PREPARATION_20261112_001.
Metadata RESULTf741414d; SOURCE_QUALIFICATION.json binder faktisk kilde og testkvitteringer.
219unike fokuserte cases er dekket:213bestod først;6nye progress-feil ble rettet med én
manglende scopegren, deretter17/17ny+gammelrouting bestått. Ekte kilde-/inputmetadata
består for102eiere, null modell-/checkpoint-/datapayloadkjøringer. NumPy/Torch importeres
av eksisterende kontrakteiere; import alene er ikke modellkonstruksjon. Peer-review bestått.

Minste kildeendring utvider eksisterende benchmark-rute til den eksakte to-passplanen,
inkludert tail6, faktisk radregnskap, INITIAL-paritet før updates og slutt-/feilrapportering.
Det kanoniske numerikkfeltet max_updates1849 gjelder én-pass statekontrakten, brukt her
bare ved nullsteg. Private3698updates har egne porter og gir ingen kanonisk resumeautoritet.

Ny native root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_LEARNING_20261112_001.
RECIPE45ca582a og PREFLIGHT196b7485 er kvalifisert. Tidligere faktisk checkpointkontroll
e34e34d2 gjenbrukes etter fersk bytehash og samsvar i ny recipe, normalisering, akserekkefølge
og modelldigest. Null nye checkpointdekodinger/modeller/arraylesinger i denne preflighten;
begge genuine INITIAL-paneler skal fortsatt være bitlike kontrollen før updates.
Windowsboot647 er forberedelsesboot; native krever en ny fysisk boot. Fersk observasjon
viser ingen prosjektjobb/writer/lås, synlig app eller GPU-beregning. Ingen jobb er stoppet.

Neste: materialiser eksakt campaign, kontroller oppstart/window uten claim, installer ny
bootoppgave og kontroller lediggrensen på nytt før fysisk omstart. Én modell,3698updates,
10500smodell-/12000sinvokasjons-/13800syttergrense; eksisterende20G/512M/250W-vakter.
Ingen automatisk ekstra pass. Kilde/policy fryses under kjøring; prosess/terminal overstyrer
admisjonsprosa. Stabil jobb kontrolleres15–30minutter, uten nye arbeidsspor i ventetiden.

Siste ferdige modellresultat er fortsatt combined-benchmark31868249 og etterkontroll
TRAJECTORY2f0ab03d (docs/TA_COMBINED_TF_DECODER_TRAJECTORY_REVIEW.json). Alle10senere S/F
AUC-punkter steg1→3 i begge armer, men alle forskjeller mellom armenes endringer har0 i
begge dag-/ukeintervaller. M5F combined0,516268→0,526010 er usikkert. Ingen tidlig gevinst
som trenes bort er etablert; AUC er ikke treffprosent eller økonomisk edge.

## Fullført sammenligning

Root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_COMBINED_DECODER_BENCHMARK_20261112_001.
Frossen native kilde a23c9346, RECIPE826d89a3, plan2fd275e9 og fersk fysisk boot647.
RESULT31868249, TERMINAL93cb7f17 og COMPLETION_VERIFICATION482437d0 er kontrollert.
768updates/0skip;2304forwards(768TRAIN+1536eval),2modeller/1optimizer/1checkpointlesing,
0checkpointskrivinger og1evidence-kapsel. Native målt3129,08s. Vakt exit0; peak57Ckjerne/
60Cminne/167,15W/2206MiB. Ingen jobb måtte stoppes for omstart. Originalkvitteringer bevares.

Begge INITIAL-paneler bestod før updates. Akser/targets/aktivitet/M5 bitlike; non-M5-side
maksavvik3,58e-7 innen forhåndsbundet toleranse. Eksponeringsplan og RNG er like kontrollen.
TRAIN hevder ingen representasjonslikhet; LATER kontrollerer begge lagrede M5-representasjoner.
Alle genuine features/8familier/5TF/D1, normalisering, delt encoder/query og tapsaritmetikk
beholdes. Bare eksisterende16sidetensorer/1032parametre er lagt til i denne opt-in-varianten.

RETAINED COMPARISON9f61ae0a er COMPLETE:4arkiver,0modeller/forwards/updates,5,69s.
Gamle kontrollpunkter og alle dag-/ukeintervaller replaysjekket; maksimal proper-score-feil
1,11e-15.70dager/14uker, punktvise utforskende intervaller og gjenbrukte utviklingsdata.
Samlet kilde-/resultatvurdering: docs/TA_COMBINED_TF_DECODER_REVIEW.json.

TRAIN-loss1,832358→1,823358; senere samletloss2,360385→2,360441. Alle10 senere S/F AUC-
endringer har0 innenfor begge95intervaller. M5F AUC0,528606→0,526010; den lille Brier-bedringen
og AUC-endringen er usikre. M15S/F Brier/log bedres i begge intervaller, mens H4S Brier/log
forverres. H1S log forverres også i begge; H1S Brier er usikker med ukegrupper.
M15F meanPup0,615858→0,594687 mot0,491060 i fasit. H1F blir0,624724 mot0,5; D1F stiger
0,638159→0,689883 mot0,575089. Separate sidehoder alene fjernet ikke skjevheten.

## Grenser

Kandidaten er ikke promotert. Ingen ny modellkode er endret etter den kvalifiserte native
kjøringen. Beståtte kilde-/migrerings-/modellkontroller gjenbrukes; policy/docs følges av
fokuserte hooks og diff-/syntakskontroll. Ingen uavhengig generalisering, Entry-kvalitet eller
økonomisk edge er bevist. Korte kontroller dømmer ikke endelig kapasitet; full kvalifisert
historikk og30–40epoker er senere skalering. TEST forseglet; GC på pause; ingen broker/handel.
