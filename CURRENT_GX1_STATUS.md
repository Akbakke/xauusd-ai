<!-- GX1_DOCUMENT_CLASS: CANONICAL | short current status -->
# GX1 status nå

Status etter BootId 360, 2026-09-11.

**Kampanjen er deaktivert mens den verifiserte telemetri-rettelsen bindes til ny kildeversjon.**
Windows-oppgaven har null automatiske forsøk, S4U-eier Andre og én oppstartstrigger.

V17 var bundet til ren commit `2de8ff8af31a9d61567de2c799923cf1b821735a`.
Den kanoniske omstartsbanen førte til BootId 360 kl. 15:11:08.5Z. Første
WSL-/kampanjeinspeksjon passerte. Kontrolleren stoppet kl. 15:12:42.6196258Z
på et tomt `Compare-Object`-resultat i telemetrikontrollen, før ACTIVE eller GPU-start.
V17 skal ikke startes på nytt med den gamle kontrolleren.

Rettelsen bevarer en array også når sammenligningen gir null eller ett objekt.
Den reelle Windows-testen under streng modus og hele den faktiske telemetribane
passerer nå, inkludert transport, sertifikat, oppgave og signert avlesning.
Sluttkontrollen passerer med 4666 tester og 14 deltester; kompilering,
shell-syntaks og Git-diffkontroll passerer også.

Den gamle femårs epoch 1 er historisk bevis, ikke et direkte resume-punkt for
lifecycle-v2. Lifecycle-v2 har ingen tvungen EXIT ved 512 eller maksimal
tradelevetid. Det videre målet er én fersk epoch med TRAIN 2025-06-01–2026-05-31
og full juni-2026 VAL gjennom eksisterende guard-, smoke- og resumeporter.
TEST er fortsatt forseglet.

Neste steg er å binde den rene etterfølgeren til eksisterende data, modell,
checkpoint og kampanjeplan; installere og kontrollere de eksakte kildefilene;
og samle de ti bevisrollene før den kanoniske oppstartsbanen aktiveres.
Arbeidet har én eier og én tung jobb om gangen. Detaljer og eksakt feilevidens
står i `CURRENT_HANDOVER.md`.
