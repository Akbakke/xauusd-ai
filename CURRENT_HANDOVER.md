# GX1 — overlevering 17. september 2026

Kandidaten er implementert og kontrollert. Én native256-prøve er bundet,
ikke startet. Læringsporten er ikke bestått. NEXT_RUN_POLICY åpner bare denne
prøven; training_enabled=false gjelder fortsatt generell trening.

Kode: /home/andre2/src/GX1_CURRENT, work/gx1-current. Data: /home/andre2/GX1_DATA.
Mac er overleveringskopi. Start med ./handover.sh --check eller Linux-scriptet
bash scripts/gx1_handover.sh --check. current_work viser faktisk nåstatus;
COMPLETED_RUN og gamle checkpoint-/VAL-felt er historikk.

Følg VEIEN_VIDERE.md og docs/RESIDUAL_NORMALIZATION_FIXED256_20260917.md.
Kilde fryses under native kjøring. Én agent/én tung jobb. Stabil kjøring følges
etter omtrent15–30 minutter eller sjeldnere. Ingen ny start ved timeout alene.

Signaldiagnosen er fullført:98,18 prosent av Entry-MSE-bedringen var fellesnivå,
bare0,36 prosent LONG–SHORT. Hidden-variasjonen falt til9,93 prosent og lokal/
fused middelradnorm vokste134,6×/155,9×. Inputnormaliseringen var uendret.
Residualnormalisering er en testbar hypotese, ikke en bevist kur. Gjenbruk
REVIEW.json og SAVED_SCALE_CAUSE/RESULT.json under
BASE/NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917; ikke gjenta diagnosene.

Siste trente kausal256-kandidat er avvist: Entry FLAT256/256, dårligere enn
TRAIN-konstanter; Exit HOLD for LONG og EXIT for SHORT, alle fire MSE verre
enn connected256. Checkpoint5,256 steg,epoch0/offset256 under
BASE/NATIVE_CAUSAL_ENTRY_FIXED256_20260917 bevares. RESUMABLE/complete=false
er ikke resume-tillatelse. BASE er
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.

Entry-detach og etterpåklok target-klipping er tidligere rettet. Entry-mål
er første likvidasjonsverdi+(119/120)*Q_HOLD, med negative verdier beholdt og
ugyldig/terminal HOLD=0. Bevar Exit-mål,kostnader,bootstrap og kausalitet.
120 beregningssteg er ingen handelsregel om maksimal holdetid.

TEST er forseglet; mars–mai/juni er utviklingsdata. Ingen fast tapsgrense eller
holdetidsgrense. Alle200 features/åtte familier/tidsrammer bevares. Læring,
generalisering og profitt rapporteres separat. Stående offentlig push gjelder
kode,docs,interne stier og aggregater; aldri rådata,vekter eller hemmeligheter.
