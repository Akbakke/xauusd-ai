# Neste avklaring: hele Exit-policyens verdi

512-prøven og Entry/Exit-koblingsanalysen er fullført. Exit-forbedringen på
TRAIN bevares. Entry er fortsatt FLAT i alle 256 målepunkter og samlet
læringsport er ikke bestått. Ingen ny kjøring er bundet.
Gjenbruk docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md og lagrede analyser.

Entry trenes på statisk Q_mu. Ett kausalt 512-Exit-valg etterfulgt av
referansepolicy forbedrer utfallet, men sidegjennomsnittene er fortsatt
negative. Det avkrefter ikke betinget handelsfordel og begrunner heller ikke
direkte targetomskriving. Hele den lærte Exit-policyen er fortsatt umålt.

Neste avgrensede spørsmål er om hele den frosne Exit-policyen forbedrer
utfallet med uendrede kostnader. Avklar separat TRAIN-omfang i eksisterende
native evaluator, begge kontrafaktiske sider, observasjonsgrense før CONTROL
og verdsetting av åpne posisjoner. Gjennomgangen har påvist at separat
cohort, observasjonsgrense og sensureringsgrunn krever en avgrenset
kontraktsutvidelse i eksisterende native motor. Dagens TRAIN-målecohort tillater ikke
rollout; bevar measurement_only-sperren. Bind modell, data, budsjett og
vurderingskriterier før eventuell jobb. Ingen ny runner eller trening.
Et positivt kontrafaktisk resultat er ikke en kausal Entry-strategi.

Fire Bps av inngang/utgang kommer fra valgt 2 Bps slippage per utførelse.
De 258 bevarte fillene og den direkte kildekjeden mangler beslutningsquote
koblet til ordre og fill. Brukeren er spurt etter eventuelle separate logger.
Undersøk oppgitt materiale når det finnes. Mangelen hindrer empirisk
kostnadskalibrering, men er ikke alene grunn til å stoppe signalundersøkelse
under uendrede kostnader. Fill=fullVWAP beviser ikke null latenstidsslippage.
Ingen nye handler, papirhandler eller kostnadsreduksjon for å få PASS.

Ingen automatisk mer trening, brede søk eller terskeltuning. TEST, full epoch,
full VAL og live/paper/spending er stengt. Kronologisk kvalitet og samlet
økonomi er fortsatt ubevist. Målet er aktivt.
