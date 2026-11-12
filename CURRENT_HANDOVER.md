# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_COMBINED_TRAJECTORY_COMPLETE_EDGE_UNRESOLVED. Edge-målet er uoppnådd.
Native benchmark og begge etterkontroller er COMPLETE/CONSUMED. Ingen aktiv jobb observert;
global training/native er false. Ingen gammel plan gjenåpnes.

Siste etterkontroll er lagret i docs/TA_COMBINED_TF_DECODER_TRAJECTORY_REVIEW.json.
Root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_COMBINED_DECODER_TRAJECTORY_20261112_001.
RESULT2f0ab03d, TERMINAL5728317e; åtte arkivlesinger, null modeller/checkpoints/forwards/
updates, målt10,16s. Akser/targets/masker og tidligere pass3-resultater er kontrollert.
Alle10 senere S/F AUC-punktverdier stiger fra pass1 til3 i begge armer. Alle10 parede
forskjeller mellom armenes endringer har0 i både dag- og ukeintervall. Ingen tidlig
combined-rangeringsgevinst som deretter trenes bort er etablert ved disse målepunktene.
M5F combined AUC0,516268→0,526010; dag95[-0,002933;0,022629], uke95[-0,004066;0,023779].
Dette er rangering, ikke treffprosent eller dokumentert edge. Ingen positiv innen-arm
AUC-endring er etablert under begge grupperinger. Pass3 forblir primær; ingen checkpointvalg.

Brukerens spørsmål om mer trening: den siste videreføringen brukte12 288unike rader én gang,
fordelt på tre disjunkte delutvalg, fra en allerede trent INITIAL. Dette er ikke tre fulle
epoker. Dagens spaninputs/normalisering er kvalifisert for29 574TRAIN-rader fra1.mai til
29.september2024;29 568er hele B16-rader. Bevares736komplementrader utenfor, gjenstår28 832.
Lengre historikk finnes, men større kalenderprefikser er ikke ferdig bundet til nye spanfelt.

Neste konkrete arbeid er én større eksponeringskandidat med eksisterende kvalifiserte inputs,
fast modell/tapsoppsett og forhåndsvalgt INITIAL, radplan, updatebudsjett og senere evaluering.
Kontroller nye valgte rader med eksisterende akse-/target-/vindusporter. Skill mer unik data
fra flere epoker. Ingen ny native kjøring er admittert nå; ingen modellendring er begrunnet
av denne etterkontrollen. Korte diagnoser er ikke en dom over full trening eller kapasitet.

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
