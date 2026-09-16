# Grunnregel fra brukeren — 2026-09-12

Prioriter å få modellen trent og måle resultatet. Gjør aldri unødvendig omfattende oppdateringer.

- Endre kode bare når en konkret, observert blokkering hindrer det avtalte treningsløpet. Navngi blokkeringen og gjør den minste nødvendige rettelsen i eksisterende kode.
- Ingen forebyggende refaktorering, nye rammeverk, ekstra rapportløp eller utvidelse av oppgaven uten brukerens uttrykkelige ønske. Allerede fungerende og verifisert arbeid skal gjenbrukes.
- Kjør bare målrettet verifikasjon som er nødvendig for den faktiske endringen. Ikke gjenta beståtte smoker, trening eller fullsuiter uten ny relevant feil.
- Én agent og én tung jobb om gangen. Kontroller kjøringer som varer i flere timer én gang i timen, gjerne sjeldnere når tilstanden er stabil (brukerpresisering 2026-09-13). Aldri minuttvis polling eller statusprat uten en konkret ny feil eller et nært forventet sluttpunkt. Lokal automatisk sikkerhetsvakt håndterer hyppige temperatur- og prosessmålinger uten modellbruk.
- Bruk tokens konservativt: vent når jobben må beregne, ikke fyll ventetiden med nye analyser, dokumentasjon eller sideoppgaver. Ikke del venting opp i minuttvise modellrunder. Gjenbruk kjent kontekst og beståtte resultater; les bare nødvendige filer og korte loggutdrag. Unngå hele manifest-, kildekode- og loggdumper.
- Før en kodeendring skal den konkrete blokkeringen og minste nødvendige rettelsen kunne forklares kort. Hvis arbeidet ikke bringer avtalt trening eller resultatmåling videre, skal det utgå. Bruk eksisterende løsning fremfor nye lag, rammeverk og generell opprydding. Når nødvendig verifikasjon består, fortsett treningen; ikke utvid testen eller endringen uten ny relevant evidens.
- Gi korte statusmeldinger ved vesentlig fremgang, feil eller resultat. Ikke gjenta en uendret status bare fordi en målfortsettelse eller timer aktiveres.
- Stående autorisasjon gjelder nødvendige handlinger innen avtalt oppgave. Ikke be om samme godkjenning på nytt.
- Bevar fullførte treningsresultater og aktiv kjøring. Ikke endre en frosset kilde eller starte kjøringen på nytt for dokumentasjon, opprydding eller spekulative forbedringer.
- Skill lokale driftsgrenser fra maskinvareprodusentens spesifikasjoner. En overskredet lokal temperaturgrense er ikke alene bevis på overoppheting eller utilstrekkelig maskinvare.

Gjeldende treningsmål og status står i `GX1_ARBEIDSMAAL.md`. Denne brukerregelen gjelder også arbeid i prosjektets eksterne Linux-repositorier.

## Gjeldende arbeidssted og opprydding — 2026-09-14

Bruk bare /home/andre2/src/GX1_CURRENT, branch work/gx1-current. Start med
bash scripts/gx1_handover.sh --check. Les CURRENT_HANDOVER.md og NEXT_RUN_POLICY.json.
COMPLETED_RUN.json er bevis fra avsluttet kjøring, aldri en ny oppstartsplan.
Neste trening er blokkert til risiko-/holdemålet, GPU-paritet, samlet fart og
resume-likhet er dokumentert. Bruk bare native campaign via gx1_capped_run.sh.
Ingen fallback til gamle smoker, separate VAL-kjørere, små batcher eller korte vinduer.
Gamle kildekopier som beholdes utenfor denne arbeidskopien er kun bundet historikk
eller nødvendige kjøremiljøer. Ikke bruk dem som en alternativ kjørevei.
Brukeren har uttrykkelig bestilt opprydding; slett bare dokumentert overflødig
innhold og bevar modell-/dataavhengigheter, fullførte resultater og checkpoints.

## Nyeste brukerprioritet — 2026-09-16

Native134/global8162 er stoppet ved brukerens beskjed08:13:52UTC. Ufullstendig
VAL viste bareHOLD; GPU var lite utnyttet. Ingen automatisk videre VAL/epoch.
Mål og rett konkret fartshinder, og vurder aktuell ONLINE mot EMA før videre
læring.25 målrettede rollout/provider-tester består for en minimal JSON-hash-
endring i VAL-økonomi. Paret faktisk CPU-fart/hash-paritet består:økonomitrinn1.32656x; målteCPU-trinn1.13905x. Full native totalfart etter endringen er ikke målt. Se nyeste
CURRENT_HANDOVER.md; tidligere startinstrukser ovenfor er historikk.

## Læringsmåling ferdig — 2026-09-16

Bred paret CPU-kontroll95→134 på1024 ulike faktiske TRAIN-Entries/4096 native
overganger er ferdig;12måneder. Prognoser for25–120min bedres, Entry/Exit-
verdilæring bedres ikke på samme mål.1021/1024 Entry-lærervalg tilsvarer første
M1-likvidasjon. Alle inputs/targets/outputs er cachet, ikke gjenta jobben.
Se nyeste CURRENT_HANDOVER.md og BROAD_TRAIN_LEARNING_REVIEW_20260916.json.
Neste er én konkret flertrinns-backuphypotese med frossen policy, i eksisterende
eiere, uten fast holdetidsgrense. Ikke lov ny full epoch før faktisk læring
og tekniske porter er dokumentert. Ingen modell-/EMA-/tapsvekt-/regel-jakt.
