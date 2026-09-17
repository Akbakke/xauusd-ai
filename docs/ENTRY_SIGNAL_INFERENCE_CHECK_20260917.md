# Bundet Entry-signaldiagnose med korrekt inferenskontroll

## Fullført — ikke relanser

Fire forwards er fullført med guard PASS og null optimizersteg. Resultatet og
CPU-skalaauditen er vurdert; se CURRENT_HANDOVER.md og de nye RESULT/REVIEW-
snapshotene. Oppstartsoppskriften nedenfor er bevart historikk.

## Gjeldende neste arbeid: Entry-signalet, fire forwards

Den rettede signaldiagnosen er separat bundet som
`NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917`. Recipe/campaign er ikke
forberedt, og kjøringen er ikke startet. Samme cachede TRAIN16, samme frosne
initial-/sluttmodeller og korrigerte mål. Fire forwards, null optimizersteg.

Mål hvor Entry-representasjonens variasjon svekkes og om hjelpetapenes gradienter
motarbeider LONG–SHORT-komponenten. Vanlig inferens kontrolleres med uendret
toleranse; gradientavvik rapporteres separat. Ingen læring loves fra denne
eval-diagnosen. Ingen Exit-forward, CONTROL/VAL/TEST eller automatisk utvidelse.

I den nye artefaktmappen finnes PLAN.json, ADMISSION_CHECK.py og
OPERATOR_HANDOVER/PREPARE.py samt ACTIVATE_TEMPLATE.ps1. De to sistnevnte er
tilpassede kopier av de fungerende native operatørene. Etter ren commit/push,
kjør PREPARE.py via eksisterende audit-vakt. Fyll så ny campaign-filhash og
kildecommit i en ny kopi av aktiveringsmalen; verifiser deaktivert task og
ingen aktiv jobb, og aktiver én gang. Les PREPARATION_RESULT/prosesser/receipt
før eventuell ny handling. Gamle planer/operatører skal ikke relanseres.

Se docs/ENTRY_SIGNAL_INFERENCE_VALIDATION_FIX_20260917.md for den testede
rettelsen og begrensningene. Steng brukt scope etter terminalt resultat.

