# Gjeldende GX1-systemkart

Rå M1-priser/bid/ask og native feature-eiere
→ TRAIN-eid normalisering og bundne M1/M5/MTF-data
→ Entry: M5 + M15/H1/H4/D1, LONG/SHORT/FLAT
→ Exit: M1 + M5/M15/H1/H4/D1, HOLD/EXIT_NOW
→ samme økonomi-/kostnadseier i trening og full juni-VAL.

Åtte familier og hele det avtalte feature-settet er bevart. Native featureverdier
beregnes på egne lukkede klokker; ingen framtidsdata eller kopiering mellom klokker.
Hjelpehodene støtter representasjonen; rå Entry-Q bestemmer inngang. Exit får
lokal prissti, livstidssammendrag med MFE/MAE og MTF-kontekst.

Én produksjonsvei:
NEXT_RUN_POLICY.json → kilde-/databundet native campaign
→ workload-only GPU-clock-launcher og Windows-controller
→ gx1_capped_run.sh → native kandidatvindu → TRAIN/EMA/full VAL/checkpoint.
Gamle direkte trainer-/fixed-step-/separate-VAL-ruter er avvist i gjeldende runner.

Fullførte bevis: COMPLETED_RUN.json → frosset V40-kilde, siste pointer,
første epochs EMA, VAL_RESULT og analyser. Dette er aldri en ny oppstartsplan.
Handover er kun lesing og har ingen historisk fallback.

Neste profil: VAL 256, åtte CPU-arbeidere, 3 timers VAL, native 12 000 s,
ytre vakt 13 800 s. Market-/metadata-cache, delt sideprissti, batched økonomi,
FP32/TF32 av og laststyrte GPU-klokker er obligatorisk. Treningen er blokkert
inntil risiko-, GPU-, totalfart- og resume-bevis er klare. TEST forblir forseglet.

Importer som heter v12/live kan fortsatt eie offline feature-/livssyklusfunksjoner.
De er avhengigheter, ikke alternative godkjente kjøreveier; ikke slett på navn alene.
