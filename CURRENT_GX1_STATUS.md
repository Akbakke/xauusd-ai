<!-- GX1_DOCUMENT_CLASS: CANONICAL | short current status -->
# GX1 status nå

BootId 361, 2026-09-11. Kampanjen er deaktivert med null automatiske forsøk.
Ingen kandidatprosess kjører.

V18 passerte kald WSL-start og signert telemetri med ren commit
747111162705a8f7e8b10df4b0ede641cf4431ab. Første smokejobb stoppet før
optimizersteg på UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_DIR_EXISTS: kampanjestarten
opprettet mappen som trenerens bootstrap krevde å opprette selv.
Feilloggen og den uavklarte ACTIVE-markøren er bevart; V18 skal ikke startes igjen.

Rettelsen gir treneren eierskap til den ferske checkpoint-mappen og legger
observatørens statusfiler i kampanjens kjøremappe. Trenerbanen signaliserer nå
til eksisterende guard etter CPU-kontroll av checkpoint og før modellstart.
En bekreftet sirkelavhengighet i neste port er også rettet: 4-stegs referanse
og 3+1-resume gjennomføres før det bevisbundne epochmanifestet opprettes.

32 berørte CPU-tester og reell Windows PowerShell-test passerer.
Brukeren prioriterte direkte fremdrift mot trening. Fjerde fullsuite ble stoppet
etter 683 tester og 14 deltester; dette er delvis verifikasjon, ikke full PASS.
De beståtte målrettede testene brukes for disse konkrete rettelsene. Ingen ny
fullsuite bare fordi endringen skal committes.

Målet er fortsatt én fersk lifecycle-v2 epoch med TRAIN 2025-06-01–2026-05-31
og full juni-2026 VAL gjennom de avtalte smoke- og resumeportene.
Ingen tvungen EXIT ved 512 eller maksimal tradelevetid. TEST er forseglet.
Én eier og én tung jobb om gangen. Eksakt feilbevis og neste steg står i
CURRENT_HANDOVER.md.
