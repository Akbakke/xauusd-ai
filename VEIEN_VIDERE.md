# Veien videre

Repo-oppryddingen etter operatørens benchmarkstopp er fullført og verifisert.
Punktene nedenfor beskriver den gjennomførte bølgen; eksakte bevis står i
docs/REPO_REVIEW.md. Ingen ny benchmark eller trening følger automatisk.

1. Kontroller import-, kall-, test-, dokument- og artefakteierskap før sletting.
2. Rett observerte defekter hos eksisterende eiere; bevar modellsemantikk,
   fryste inputs, genuine features og alle sikkerhets-/læringsporter.
3. Fjern frakoblede kilder, dobbeltstatus og ferdige engangsrapporter fra repoet.
   Bruk Git til gjenoppretting, ikke en ny historikkmappe.
4. Kontroller syntaks, tester, gjenværende referanser, handover og kildeidentitet;
   commit/push innen stående autorisasjon. Rapporter det som ikke er verifisert.

Etter levering kan brukeren gjenoppta læringsløpet i docs/NATIVE_LEARNING.md.
Ny benchmark krever ny kildebundet plan; ATTEMPT_003 er konsumert og terminal.
Full benchmark → sampler/koordinater → fersk initialmåling → 256-stegs prøve
→ separat TRAIN/CONTROL256-review → bare betinget, endelig utvidelse.
Dette er rekkefølge, ikke nåværende launchautorisasjon.

Full makro-B (DFII10, DTWEXBGS, T10YIE, GLD, COT, VIX) består som separat mål.
Historiske dataversjoner/tilgjengelighetsklokke er uavklart for deler av kjeden.
MACRO_CORE erstatter ikke B og har ingen GO/promotion. Ingen nye hentinger eller
forskningsfits er åpnet av oppryddingen.

TEST, broker, live/paper, handel og spending forblir stengt.
