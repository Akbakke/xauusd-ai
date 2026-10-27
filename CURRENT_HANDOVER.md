# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo er /home/andre2/src/GX1_CURRENT, branch work/gx1-current på
gx1-3090-lan, WSL Ubuntu-22.04. Root i denne tråden eier integrasjon, policy
og launch. Bruker har avvist oppdrag fra andre chatter. Den særskilt bestilte
uavhengige retningsagenten har fullført; ingen ny delegering er nødvendig.
Les GX1_RULES helt ved første arbeid. Kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er eneste arbeidsstatus; originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

Bps-supervisjon er implementert som opt-in i eksisterende native retningsrute,
men ennå ikke kvalifisert. DIRECTION_BPS_SUPERVISION_SYNTHETIC_PLAN_001 er
engangsadmittért for 144 syntetiske kontroller. Kilde og policy fryses ved commit.
CPU audit 4G/512M, én tråd, 64 tasks, 180 s worker / 210 s outer.
Syntetiske heads/gradienter/AdamW/resume og filrundtur er tillatt i eget output;
ingen ekte payloads, full native-konstruksjon/forward eller trening er tillatt.
Bps-head er initialt null med bevart RNG; den deler udetechet native hidden.
Skala er egen prefiks-RMS; nuller deltar i bps-loss. Original BCE er default.
Måltilførsel, modell, loss, recipe, feature-view, state og rapport er koblet.
Ingen ny horisont/epokeendring eller genuin kjøring er admittért.

DIRECTION_MINUTE_TARGET_SYNTHETIC_20261027_002 er COMPLETE/claimed/consumed.
Original worker og capped outer avsluttet med exit 0. Alle 36 fokuserte
syntetiske tester bestod; worker 4,612 s, outer 5,124 s. Kilde og policy var
uendret; cgroup/eier er borte, originallåsens identitet er kontrollert og ledig.
Forsøk 001 bevares som FAILED før teststart og relanseres aldri.
Eksisterende prisfasiteier støtter eksplisitte minutter 1–60, inkludert 10,
med uendret 1/5/15-default og egen schema-identitet for andre horisontvalg.
Testene dekker gammel syntetisk paritet, oppgang/retur og speilvendt fall,
klokker, felles entry-opphav, null/manglende/cutoff og feil horisontakse.
Ingen ekte payloads, native-kall, fit eller optimizersteg i denne enheten.
Native head/loss/state er uendret. Ingen ekte utvidet fasit er materialisert.
Se docs/DIRECTION_MINUTE_TARGET_QUALIFICATION_20261027.json og bundne originaler.
Begge engangsautoritetene er lukket; ingen trening er admittért.

DIRECTION_BPS_ALLOCATION_20261027_001 er COMPLETE/claimed/consumed.
Original worker og capped outer avsluttet med exit 0. Worker brukte 14,0709 s:
9 lagrede TRAIN-NPZ, 54 proper-score-kall og 362 rangkall. Ingen fit, native
konstruksjon, forward, reverse, optimizersteg eller rådatasettlesing.
CPU audit 4G/512M, én numerisk tråd, 64 tasks, 180 s worker / 210 s outer.
Kildecommit 31540e604705e1eda47849a25480889804efa91a og policy var fryst;
originalinputs og reproduserte native-/prior-skårer ble kontrollert.

Root har bare lest kilde og JSON-aggregater/metadata, ikke payloads.
Originale terminaler og hashbindinger er verifisert. Cgroup og opprinnelig eier
er borte; den identitetskontrollerte originallåsen er ledig. Ingen jobb kjører.
Alle autoriteter for denne engangsenheten er lukket. Oppfølgingen er PAUSED.
Ingen engangsplan relanseres. Originaler og checkpoints bevares.

DIRECTION_BPS_ALLOCATION_ACTUAL_ROOT_REVIEW_001.json binder originalene.
DIRECTION_BPS_ALLOCATION_VERIFIED_TERMINAL_001.json er metadatakonvolutten.
Aggregert repo-kopi: docs/DIRECTION_BPS_ALLOCATION_RESULT_20261027.json.
Kilde: gx1/scripts/diagnose_entry_direction_bps_allocation_v1.py.
15 fokuserte syntetiske kontroller bestod før den genuine enheten og gjenbrukes.

## Hva den nye målingen avklarer

Den forhåndsdefinerte hypotesen var at 0 < absolutt avkastning <= 1 bp utgjør
over halvparten av både gyldige retningsceller og målt sluttmodell-BCE.
Denne hypotesen er ikke støttet. På hele TRAIN er celleandelene:

- 1 minutt: 39,17 %.
- 5 minutter: 20,20 %.
- 15 minutter: 11,84 %.

Alle de fire opprinnelige prefiksene ligger også under 50 % ved hver horisont.
På første prefiks, 47 602 foreldre, utgjør disse cellene 26,2642 % av den
opprinnelige B16-koeffisientmassen og 26,2694 % av frossen, batchvektet BCE.
Grensen for «små bevegelser» skal ikke endres etterpå for å redde hypotesen.
Dette beviser verken at små bevegelser er støy eller at størrelsesmål gir edge.

Den korte sluttbatchen har 2 foreldre / 6 gyldige celler; cellekoeffisienten er
8 ganger en full batch med 48 celler. Frossen batchgjennomsnitts-loss er
0,6930733897 mot globalt cellegjennomsnitt 0,6930726337. Dette er ikke historiske
treningslogits eller optimizergradienter og beviser ikke at siste steg skadet
generaliseringen. Ingen batchrettelse er innført.

Alle faste bps-bånd er rapportert for native, prior, ET252 og R1. Native-rangering
blir ikke jevnt bedre ved større faktiske bevegelser. Båndene velges etter kjent
utfall og kan ikke bli et handelsfilter. Sparse bånd har begrenset bevisverdi.
Ingen ny prediktor, bedre retningsbeslutning eller økonomisk edge er dokumentert.

## Beholdt resultat fra fast terminal-readout

F_TERMINAL_READOUT_20261027_001 er komplett og konsumert. Samme frosne modell,
128D hidden, 47 602 prefiksforeldre og 17 288 senere foreldre ble brukt.
Tre faste lineære fits konvergerte og forbedret TRAIN-fit ved alle horisonter.

Ved 5 minutter steg TRAIN-AUC fra 0,515888 til 0,539421.
Senere AUC gikk fra 0,503021 til 0,503360, delta +0,000339.
Parvise 99 %-intervaller: UTC-uke [-0,009623; +0,010859] og
UTC-dag [-0,010497; +0,010879]. Begge inkluderer null.
Senere logloss ble dårligere ved alle tre horisonter; ved 5 minutter
0,693514 til 0,700082. Kandidathodet innføres ikke.

Dette svekker enkel slutt-head-underfitting som løsningen i denne folden.
Det avkrefter ikke bedre backbone-læring, flere epoker eller ikke-lineært signal.
Se docs/DIRECTION_FROZEN_READOUT_RESULT_20261027.json og de bundne originalene.
Ikke relanser eller utvid forsøket uten en ny konkret begrunnelse.

## Mål og uavhengig gjennomgang

docs/DIRECTION_TARGET_PATH_REVIEW_20261027.json inneholder fem hypoteser,
kildelinjer, motargumenter, avgjørende tester og stopp, oppdatert med begge funn.
Agent direction_target_independent_review_20261027 fullførte et lesende mandat.
Ingen fortegnsfeil er funnet i den undersøkte mål-/modell-/losskjeden.

10 minutter er eksplisitt med i målutformingen. Rapportpunktene er
1/5/10/15/30/60; foreslått full fasitkurve er hvert minutt 1–60 fra samme M5-entry:
R_h = 10000 * (mid(D+h)-mid(D)) / mid(D), D = M5 bar_start + 5 minutter.
Den målte native-modellen har fortsatt bare 1/5/15-utganger.
Native 10/30/60 og fulle 1–60-utganger er ikke implementert eller målt.
Selve prisfasiteieren støtter nå domenet; ekte fasitparitet gjenstår.

Endepunktsretning er gyldig: tidligere endepunkter kan se oppgangen selv om
et senere endepunkt returnerer til entry. Dagens BCE gir derimot samme
supervisjon for +0,1 og +50 bps; mid_return_bps velges bort i treningsadapteren.
Observerte nuller utelates og lærer ikke FLAT. Dette er en påvist målbegrensning,
ikke bevist årsak til svak retnings-AUC. Små-bevegelse-dominans er nå svekket.

Modellens harde trigger er z_h = b_h + w_h·hidden > 0 for UP, ellers DOWN.
Featurefølsomhet, prediktiv verdi, markedsårsak og handelsordre er ulike spørsmål.
Gatevekter forklarer ikke kausalt markedet. Ingen handel er åpnet.

60 likevektede horisonter gir 45/60 av vekten etter 15 minutter ved komplette
masker. Det er et fremtidig designvalg, ikke årsaken til dagens tre-horisontresultat.
60 marginale sannsynligheter er ingen felles prisbane og gjenoppretter ikke
intraminuttrekkefølge. Senere short krever en ny kausal beslutning.

## Neste minste steg

Bruker presiserte at +0,1 og +50 bps ikke skal gi samme læringsinformasjon.
Prioriter et separat signert bps-mål ved uendrede 1/5/15-horisonter.
Minste kandidat er tre betingede middelverdiutganger i bps fra samme hidden,
med prefiksfryst RMS-normalisert kvadratfeil i tillegg til uendret BCE.
Konkrete formel, masker, gradientkontroller, initialisering, referanser og
stopp står i docs/DIRECTION_TARGET_PATH_REVIEW_20261027.json.
Dette er nå implementert som opt-in; syntetisk kontroll er admittért.
Genuin baseline/gradient/resume-kvalifisering og sammenligning gjenstår.
Middelverdi er ikke et usikkerhetsmål eller en handelsbeslutning.
Kvadratfeil kan overvekte sjeldne store feil; de faste bps-båndene rapporteres,
men brukes ikke til etterpåvalgte filtre eller et nytt terskelsøk.

Først må mid_return_bps faktisk nå dette tapet, med eksakt samme foreldre/
klokker og kun egen prefiksmoden fasit. Behold P(UP)-semantikken og ta med
observerte nuller i bps-komponenten uten å lage en FLAT-fasit av dem.
Ny funksjon krever fersk baseline/state/resume-kvalifisering. Ikke relabel gamle
checkpoints. Ingen genuin numerisk enhet eller native-trening er admittért her.

Bps-supervisjon, horisontutvidelse og én-mot-tre epoker er separate faktorer
med samme observasjoner, kronologi og eksponering. Ingen automatisk 30 epoker.
Eksakte nye minutter krever kvalifisert TRAIN-M1-kilde og avgrenset admisjon;
aldri interpolasjon eller selvstendig åpning av blandet 2026-Parquet.
Ikke gjenta den avsluttede små-bevegelse-målingen.

## Bevarte bevis og grenser

Fire terminal-uttrekksenheter er komplette. Opprinnelig åtte-enhets
0U-versus-terminal-studie er fortsatt uferdig: seks av åtte uttrekk fullført.
To own0U-enheter er utsatt. Den gamle planen gir ingen ny autoritet.

Firefoldanalysen på 67 112 samme senere foreldre viste native AUC5 = 0,503735
mot ET252 = 0,513092, med negative parvise uke-/dagintervaller korrigert for seks
kontraster. Alle 118 249 TRAIN-foreldre hadde bitidentiske 2 622 lokale felt/masker.
Full numerisk MTF-paritet eller felles upstreamfeil er ikke avklart.
DIRECTION_1_60_CAUSE_SYNTHESIS_008 bevarer disse bevisene.
TRAIN-only biasjustering forbedret 5m proper score uten å endre AUC.
GC-OFI viste sterk samtidig GC-respons og svak fremtidig XAU-sammenheng i
avgrenset analyse; det avviser ikke alle inputenes fremoversignal.
Endrede parametre avkrefter helt stillestående modell, ikke uproduktiv læring.

M5 Entry, åtte spesialistfamilier og D1/makro beholdes. M5/M15/H1/H4/D1 er i
faktisk Entry-rute; 61 D1/makrofelt eies av session_regime_encoder.
TRAIN mai 2024–desember 2025. CONTROL januar–mars 2026 er lukket.
Juni 2026 er gjenbrukt utviklings-VAL. TEST fra juli 2026 er forseglet for
tilpasning/utvalg; forward-journalen gir ikke denne oppgaven TEST-autoritet.
Ingen Exit-trening, broker, live/paper, ordre, spending eller promotion.
Arbeidsmålet er aktivt og uferdig. Teknisk PASS er ikke edge.
