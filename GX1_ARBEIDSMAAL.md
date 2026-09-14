# Gjeldende GX1-mål — 2026-09-14

Få Entry og Exit til å lære riktig før større trening, og mål kostnadsjustert
netto Bps ærlig. Prioriter selvstendig Entry-signal, selektivitet og tydelig
Exit-læring. Ingen bred jakt på regler, EMA-varianter eller nye modeller.

Brukerens siste presisering: ett år er ønsket første større læringsforsøk av
hensyn til effektivitet. Anbefalt TRAIN er 2025-06-01 til 2026-06-01 eksklusiv,
etterfulgt av hele juni 2026 som VAL. Bruk CURRENT-bundne data og native campaign.
Ettårsdekning bindes som et datoutvalg på eksisterende full TRAIN-indeks.
Full parenthistorikk, opprinnelige child-ID-er, prisbaner, features og alle
normaliserings-/foldartefakter beholdes. Bare hvilke TRAIN-rader som teller
i en ny epoch avgrenses. Ingen ny child-/summary-datakjede er nødvendig.
Ettårsoppskriften er ennå ikke aktivert for hele epocher. Femårsvekter
som initialisering skal merkes som videre kalibrering, ikke som en modell som
bare har sett ett år. Juni er utviklings-VAL, ikke et nytt uavhengig holdoutbevis.
TEST forblir forseglet. Ingen full femårs-epoch startes nå.

Intradag er ønsket stil; M5 er ingen pålagt handelsvarighet eller eneste tillatte
fokus. Behold alle 200 features, åtte familier og samarbeid mellom tidsrammer.
Høyere TF skal bidra målbart, ikke dominere bare fordi vi antar at den hjelper.
Robust kvalitet er viktigere enn antall handler. Etterprøvbare registreringer
av inngangsdata, modellverdier, beslutninger og senere utfall skal skille hva
modellen visste ved beslutningen fra etterfølgende vurdering. Mer adaptiv ML/RL
må begrunnes med bedre resultater på senere perioder uten læringslekkasje.

Risiko er avklart: «Ingen fast grense». Ingen fast tapsgrense eller maksimal
holdetid. Exit sammenligner forventet videre nettoverdi med gjennomførbar lukking.
Bevar fysisk pris-/kostnadsregnskap, split-censoring og etterprøvbar verdsetting
av åpne posisjoner. Ingen statistikk bare for lukkede vinnere kan kalles samlet
lønnsomhet. Risikobindingen står i docs/RISK_OBJECTIVE_20260914.json.

Gjeldende målt status:

- Første femårs-epoch og hele juni-VAL er bevart. Epoch 2 ble stoppet på
  checkpoint 315, epoch_index 1, offset 320, totalt 19 908 optimizersteg.
  Juni analyseres bare med første epochs uforanderlige EMA.
- Av 5 508 valgte juni-handler ble 2 227 lukket og 3 281 avkortet ved månedsslutt.
  Full-policy netto Bps fra denne kjøringen er ikke tilgjengelig.
- Native v4-kontroll 34fd8997 fullførte 16 → 32 nye steg med guard PASS.
  Faktisk checkpointovergang og resume er kjørt; numerisk resume-likhet er ikke bevist.
- Native kontroll 7054ec8b fullførte 16 steg på boot 419 med guard PASS,
  trainer/observer/controller exit 0. Ingen trainer kjører; Windows-task er deaktivert.
  På de undersøkte Entry-rutene er Q/Exit-gradientene frakoblet, mens forecast
  fortsatt lærer. Første batchs fremoververdier er uendret i eksisterende loggmål.
- Samme batch viser HOLD-target mean 32.1241 Bps, hvorav arvet frozen bootstrap
  er 32.0953 Bps. Faktisk reward har mean 0.0288 og absolutt mean 1.7169 Bps.
  Dette viser avhengighet av gammel verdilærer, ikke at all fortsettelsesverdi er feil.
  Selektivitet, kalibrering og lønnsomhet er fortsatt ikke dokumentert.

Minste rettelse er implementert med 48 beståtte målrettede CPU-tilfeller:
en eksplisitt ny v4-initialisering av bare Exit-verdi-
utgangen fra lukk-nå-baseline 0, konsistent i online, target, EMA og relevant Adam.
Markedsencodere, Entry-Q og øvrig tilstand beholdes; gamle checkpoints røres ikke.
Null er lærerens startpunkt, ikke en holdegrense eller bevist optimal verdi.
Før større læring sammenlignes 32 uavbrutte native optimizersteg med 16+16 steg
over fysiske booter fra samme baseline og ettårsutvalg. Et uforanderlig snapshot
ved steg 32 kan måle den eksisterende hele-juni-VAL-profilen (256/8/10800).
Dette er rapportering av kapasitet/paritet; snapshotet kan ikke velge checkpoint,
flytte early stopping eller fremstilles som en fullført epoch. Historiske
EMA-steg bindes med eksakt offset til den bevarte overgangspointeren.
Alle nødvendige målebevis gjenstår til de faktisk er kjørt og godkjent.

Eneste kodebase: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Start/gjeninntreden: bash scripts/gx1_handover.sh --check; les CURRENT_HANDOVER.md
og NEXT_RUN_POLICY.json. Full trening er fortsatt av. Bruk bare native campaign
via gx1_capped_run.sh med TRAIN-batch 16, VAL-batch 256, åtte CPU-arbeidere,
tre timers VAL-vinduer, FP32, eksisterende optimaliseringer og maskinvarevakter.
Historiske kildekopier er avhengigheter og dokumentasjon, aldri reservekjørevei.

Underagenter er uttrykkelig autorisert. Bare én tung jobb samtidig; root eier
integrasjon, verifikasjon og oppstart. Ingen gjentatte beståtte tester uten ny
relevant endring. Kontroller stabil langkjøring omtrent én gang i timen.
Stående autorisasjon gjelder. Oppdater handover ved vesentlig endring og
commit/push ferdig arbeid. Ingen live-/papirhandel, spending eller TEST-bruk.
