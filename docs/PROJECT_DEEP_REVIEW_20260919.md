# Felles kausal policy-fit-grense

Filnavnet er beholdt fordi entry_causal_m1_target_policy_v1.py refererer C-1.
Gamle cloud-/checkpoint-/rebuildstatusavsnitt er fjernet; de er ingen aktuell
arbeidsautoritet og finnes ved behov i Git.

## C-1. Én eier for TRAINs siste gyldige beslutningsbar

TRAIN-grensen er half-open. Beslutningsbaren som åpner ved train_end tilhører
neste split. Siste M5-beslutningsbar for policyfit er derfor én
ENTRY_DECISION_BAR_SECONDS tidligere.

Eier: gx1/contracts/entry_causal_m1_target_policy_v1.py,
causal_m1_policy_fit_train_end(train_end).
Builder, ranker, preflight og rebuild-kjede skal bruke denne eieren og kreve
samme eksakte klokke/provenans. Ingen unilateral shift, kopiert literal eller
gjettet fallback. En ren datametadata-PASS uten faktisk kobling til eieren er
ikke bevis for samsvar. Fit-/source-/splitidentitet skal feile lukket ved drift.

Dette er en diagnostisk mål-/fitgrense, ikke maksimal native holdetid eller
bakoverrettet embargo. Featurelookback er kausal warmup. Fremtidsutfall er
supervision, aldri modellinput. Hvert utfallsdomene purges på sin egen klokke.

## Bevisgrense

Den beholdte kontrakten og fokuserte testsuiten eier grensene. Teknisk konsistens
på disse feltene beviser ikke v38-læring, generalisering eller lønnsomhet.
Nåstatus, jobs og tillatt omfang eies bare av NEXT_RUN_POLICY.json.
