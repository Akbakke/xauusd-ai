# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_COMBINED_DECODER_COMPLETE_NO_ESTABLISHED_LATER_DIRECTION_GAIN. Edge-målet er uoppnådd.
Native jobb og retained review er COMPLETE/CONSUMED. Controller exit0; oppgaven deaktivert,
aktive markører fjernet og prosessene avsluttet. Global training/native er false. Ingen ny jobb
eller trening er admittert; gamle engangsplaner skal ikke relanseres.

Neste konkrete spørsmål: får combined en tidlig senere retningsgevinst ved pass1 som forsvinner,
eller overføres TRAIN-bedringen heller ikke da? Bruk de åtte eksisterende pass1/pass3 TRAIN/LATER-
arkivene for combined og COVERAGE til én avgrenset lesende forløpskontroll. Gjenbruk scorere,
pass3-regnskap og INITIAL-JSON. Behold pass3 som primær; ingen checkpointutvelgelse.
Parvis tidsforskjell krever de faktisk parede prognosene, ikke subtraksjon av ferdige CI-grenser.
Det trengs separat eksakt auditadmisjon/preflight, ingen ny modell eller trening.

Publisert JSON viser allerede senere combined-loss2,350915→2,411408→2,360441 ved0/1/3.
Det støtter ikke enkel monotont dårligere prognose ved mer trening. Pass1-AUC/meanP og paret
rangeringsforløp er ennå ikke målt. Passene øker også unik datadekning; ikke kall endringen
ren epoch-, repetisjons-, optimizer- eller encoderårsak.

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
