# GX1 — hele frosne Exit-policyen er vurdert

FROZEN_EXIT_TRAIN_POLICY_20260919 er fullført, vurdert og deaktivert.
Guard PASS; null nye optimizersteg; original512/lærer/cursor bevart.
Alle512 kontrafaktiske handler fikk faktiske modell-EXIT. Ingen åpne posisjoner,
sensurering eller ressursavbrudd. Entry er fortsatt FLAT256/256 og faktisk netto0.

|Hele Exit512 på TRAIN256|LONG|SHORT|
|---|---:|---:|
|Gjennomsnittlig netto Bps|−4,1080|−5,1893|
|Bedring mot umiddelbar EXIT Bps|+1,8409|+0,5902|
|Positive måneder|2/9|0/9|

Den forhåndsbestemte øvre/nedre rangeringshalvdelen ga−3,3433/−4,8623 Bps.
Øvre halvdel er bedre i5/9 måneder, men fortsatt negativ. Det foreløpige
hybridresultatet−0,3998 for øvre halvdel var ikke fullpolicy-resultatet.
Bedre Exit enn umiddelbar lukking er målt på TRAIN; profitabel Entry/Exit-strategi,
kronologisk kvalitet og læringsport er fortsatt ikke dokumentert.

1083 forwards og21333 tilstander; selve rollout tok160,51s, native arbeid776,79s.
Kjøring19:14:01–19:29:19 UTC /21:14:01–21:29:19 Oslo2026-09-19; controller
Disabled19:33:30 UTC. Ingen aktiv jobb eller ny kjøreautorisasjon.

Entry lærer fortsatt Q_mu, mens hele forløpet brukte pi512. Cachet sammenligning
bekrefter ulike mål/utfall; dette er et designspørsmål, ikke automatisk en kodefeil.
Policykonsistente fullpolicy-labels og cachet baseline er nå ferdige: alle256
rader/512 sider og409 negative labels er bevart, FLAT0. Tre uttrykkelig bestilte
agentgjennomganger prioriterer én regularisert Entry-selector som avkreftingsprøve.
Entry-Q inngår også i Exit-tokenet; kandidatens valgverdier må holdes separat
fra hele originalfunksjonen. Fit127/check129 er brukt TRAIN, ikke uavhengig VAL.
Følg VEIEN_VIDERE.md for konkret måling og stoppkriterier. Ingen fits er bundet. Kun den avgrensede uttrekkingen nedenfor er tillatt. Kostnader og terskler er uendret. Ikke relanser planen.

Bevis: docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md og de tre nye snapshotene
FROZEN_TRAIN_POLICY_REVIEW, FROZEN_TRAIN_POLICY_COMPLETION og
ENTRY_COMPLETE_POLICY_TARGET_ALIGNMENT under handover_snapshot.
Kun /home/andre2/src/GX1_CURRENT, work/gx1-current. Mac er overleveringskopi.
Tidligere operatørkopi under kildefrys er historikk. Målet er fortsatt aktivt.

Minimal native uttrekking av eksisterende Entry-representasjoner er implementert.
19 målrettede tester bestod under beregningsvakt. Én uttrekking er nå bundet:
NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919,16 Entry-kall,0 fits/0 Exit-rollout,
0 optimizersteg og krav om eksakt original Q og Exit-kontrakt. Native måling er
ikke startet. Fullfør commit/push og eksisterende native klargjøring; følg
VEIEN_VIDERE.md. Ingen forbedret handelsfordel er påvist.
