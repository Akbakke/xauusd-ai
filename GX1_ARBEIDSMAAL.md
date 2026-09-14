# Gjeldende GX1-mål — 2026-09-14

Brukerens nyeste mål: Undersøk og rett konkrete læringsproblemer før stor trening.
Prioriter selvstendig Entry-signal, selektive M5-innganger med målbar høyere-TF-
kontekst og tydeligere Exit-læring. Bruk observerte TRAIN-utfall som lærer, uten
å gjøre etterpåklok optimal Exit til live-input eller kopiere svake Q-verdier som
fasit. H4/M15-samarbeid skal måles, ikke antas. Ingen bred regel-/modelljakt.
Deretter korrekt checkpointovergang, GPU256-paritet, samlet fart og resume.
Siste rettelse beskytter Entry-spesifikke representasjoner fra Q/Exit-token-tilbakeføring;37 målrettede tilfeller består. Neste native-vindu16 måler faktisk pris-/kostnadsreward versus arvet bootstrap.
Første avgrensede rettelse er nå verifisert: Exit-verdi relativt til kjent
gjennomførbar lukking, med konsistent kostnadsforankring tilbake til Entry.
143 unike målrettede CPU-tilfeller består; faktisk kalibrering er ikke bevist.
Økonomiovergang og begrenset native TRAIN-måling er nå implementert med
79 ytterligere målrettede kontrakttilfeller. Faktisk overgang/måling er ikke kjørt.
Neste handling er bundet v4-recipe og kontrollforsøk16/32 optimizersteg, deretter
rapporterende native GPU256/totalfart/resume. To underagenter er nå autorisert.
Native TRAIN-kontroll16→32 er nå fullført på34fd8997; ingen full epoch er startet. Læringskalibrering og GPU256/totalfart/resume-likhet gjenstår. Første native kalibreringsforsøk
på ee4693e2 feilet i ytre readiness-kontroll før guard/session/optimizer.
Oppstartskoblingen til godkjent16/32-scope rettes; gammelt forsøk bevares.
Neste forsøk må bruke ny kildebinding og fersk fysisk boot.

RISIKO AVKLART: Brukeren har svart «Ingen fast grense». Ingen fast tapsgrense
eller maksimal holdetid. Exit styres av forventet videre nettoverdi etter kostnad.
Beslutningen er bundet i docs/RISK_OBJECTIVE_20260914.json og NEXT_RUN_POLICY.json.
Øvrige måle-/treningsporter gjenstår; arbeidet fortsetter fra eksisterende rettelser.

Siste dobbeltkontroll: docs/PRETRAINING_DOUBLECHECK_20260914.md. Vi er fortsatt
ikke klare for en ny epoch. Prioriter læringssignal/selektivitet og målt Entry-
svikt, deretter korrekt overgang og de avtalte GPU-/fart-/resume-portene.
Ingen bred regeloptimalisering eller ny arkitektur er begrunnet.

Nytt aktivt mål etter brukerbeskjed: Følg den prioriterte anbefalingen konkret mot
positiv kostnadsjustert netto Bps. Entry må dokumentere selvstendig kvalitet mot
observerte priser; en god Exit skal ikke være bevis for riktige innganger.
Bevar samarbeidet, men skill markedsprognose fra Exit-avledet handelsverdi.
Eksisterende prognosemål skal undersøkes før nye modeller eller hoder bygges.
Målt uavhengig Entry-kvalitet og den eksakte treningskoblingen er dokumentert i
del 2E av prosjektgjennomgangen. Årsaken til svakt signal er fortsatt uavklart.
En eksplisitt, inaktiv v3-kandidat for markedsverdibasert økonomi er implementert
og verifisert med 42 målrettede CPU-tilfeller; se CURRENT_HANDOVER.md.
Native v3-evaluering har nå separat åpen markedsverdi og en kontroll av én
posisjon om gangen; 57 målrettede tester besto. Kronologisk nettoverdi er nå
koblet til eksplisitt checkpoint-policy v5 med patience 5; 23 nye målrettede
tilfeller besto. Risikovalg er avklart øverst. Ingen ny kampanje er aktivert;
produksjonsovergang, GPU-paritet, samlet fart og resume samt Entry-kvalitet
gjenstår. Dette dokumenterer ikke lønnsomhet.

Brukerpresisering 2026-09-14: Kartlegg og mål hele den relevante Entry/Exit-kjeden,
åtte familiers og tidsrammenes samarbeid, selektivitet/FLAT, risiko, data og drift.
Kvalitet fremfor antall handler. Vurder også alternative regler/modeller/metoder,
men gjør ingen unødvendig koding eller omskriving. Gjenbruk ferdige analyser,
bevar én agent og én tung jobb, og lever prioriterte, etterprøvbare forslag.
Første samlet leveranse: [docs/PROJECT_REVIEW_20260914.md](docs/PROJECT_REVIEW_20260914.md)
med nye målebevis i handover_snapshot/PROJECT_AUDIT_METRICS_20260914.json.
Forslagene er ikke en aktivering av neste trening eller valg av numeriske risikogrenser.

Tren GX1 mot positiv kostnadsjustert netto Bps med hele feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

BRUKERBESTEMT STOPP: Full juni-VAL etter første femårs-epoch er ferdig. Ingen videre epoch før Entry/Exit, MAE/MFE, månedsslutt uten lukking og mulige Entry-filtre er analysert og forklart. Brukeren har samtidig godkjent lengre kjøreøkter, større VAL-batcher og mer parallell beregning, bare med bevart kvalitet. Disse tiltakene skal måles og verifiseres målrettet; ikke start nytt treningsløp som del av analysen.

Windows-oppgaven GX1RandomAccessCampaignV2 er fjernet etter eksport til privat arkiv; native treningsprosess 723 er avsluttet. Automatikken hadde rukket å starte epoch 2 før stoppbeskjeden kom. Siste lagrede pointer er checkpoint 315, epoch_index 1, 19 908 optimizersteg / batch-offset 320. Første epochs uforanderlige EMA-snapshot (19 588 steg) og alle tidligere kilder er bevart. Den avsluttede kjøringen brukte frosset kilde 03592fe6 i /home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40. Videre arbeid skjer bare i GX1_CURRENT.

Full VAL utførte 57 845 748 tilstandsvurderinger. Av 11 016 hypotetiske LONG/SHORT-forløp ble 7 472 lukket av modellen og 3 544 avkortet ved månedsslutt. Entry valgte 4 180 LONG, 1 328 SHORT og ingen FLAT; bare 2 227 av de 5 508 valgte handlene ble lukket, mens 3 281 ble avkortet. Full-policy netto Bps er derfor ikke autoritativt tilgjengelig. Ikke presenter positiv statistikk bare for lukkede vinnere som hele modellens lønnsomhet.

Sluttresultatet er bevart lokalt i trade_review_20260914/VAL_RESULT_EPOCH_1.json og i den frosne native sesjonen. Analyse skal skille faktiske lærte Exit-resultater fra hypotetisk likvidering ved månedsslutt, og skille Entry-retning/timing fra Exit som slipper tidligere gevinst.

Én agent og én tung jobb. Kontroller én gang i timen, etter brukerens presisering 2026-09-13. Endre bare observerte blokkeringer og uttrykkelig bestilte tiltak. Stående autorisasjon gjelder. Bevar frosne kilder, fullførte resultater og lagret fremdrift.

Ressurser: 20 GiB RAM, 512 MiB swap, 128 oppgaver, CPU 0–18; 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Eksakte bindinger står i [COMPLETED_RUN.json](COMPLETED_RUN.json). Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.

Further verified finding: SHORT HOLD has zero running financing/risk reward. With split-end censoring and no economic terminal, indefinite zero-reward HOLD can dominate voluntarily realizing a loss under the implemented objective. This is an objective-level incentive; more epochs alone are not a demonstrated remedy. Clarify intended economic holding/risk constraints before altering the objective. Full reasoning and evidence are in docs/ENTRY_EXIT_REVIEW_20260914.md. Preserve the stop.

Eneste videre arbeidskopi er /home/andre2/src/GX1_CURRENT, branch work/gx1-current. NEXT_RUN_POLICY.json beskriver avtalt fartprofil og beviskrav; handover-kontrollen viser blokkert til GPU-, totalfart- og resume-bevis er klare. Native start har egne bundne kampanjeporter; direkte håndheving av de nye bevisrollene må kontrolleres før neste kjøring. Tidligere kodekopier er historiske, ikke alternative kjøreveier.
