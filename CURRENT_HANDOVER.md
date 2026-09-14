# GX1 overtakelse — 2026-09-14

Gjeldende arbeidskopi: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Kjør bash scripts/gx1_handover.sh --check. Scriptet viser verifisert historisk
resultat, lagret checkpoint, faktisk prosessstatus og hvorfor neste løp er blokkert.
Det starter ingenting og har ingen gammel reservevei.

## Fullført kjøring og resultater

Full femårs-TRAIN epoch 1 og full juni-2026 VAL er ferdig. Brukerstopp gjelder.
Windows-kampanjeoppgaven er fjernet; ingen videre epoch startes automatisk.
Epoch 2 startet før stoppbeskjeden og rakk 320 lagrede steg: checkpoint 315,
epoch_index 1, totalt 19 908 steg. Første epochs EMA ved 19 588 steg er uforanderlig.
Siste kjøring var fra frosset kilde 03592fe6; denne skal ikke redigeres.
COMPLETED_RUN.json binder nøyaktige kilder, checkpoints og hele resultatfilen.
Invokasjon 7 ble manuelt avsluttet; den skal ikke omtales som en ytre PASS.

VAL: 57 845 748 tilstandsvurderinger / 467 371 forwards. 7 472 av 11 016
hypotetiske sideforløp ble lukket; 3 544 nådde månedsslutt med HOLD.
Entry valgte 4 180 LONG / 1 328 SHORT / 0 FLAT. Av 5 508 valgte handler ble
2 227 lukket og 3 281 avkortet ved månedsslutt. Full-policy netto Bps er utilgjengelig.
Hypotetisk månedssluttslukking av åpne valgte handler gir −312,90 Bps per handel;
dette er diagnostikk, ikke lært Exit eller porteføljeavkastning. Verste netto MAE −1 369,27 Bps.

## Hva analysen betyr for neste steg

72,42 % av valgte månedssluttåpne handler hadde først +20 netto Bps og senere
under −20 uten lukking; rundt 89 % hadde største MFE før verste MAE.
Høyeste 10 % Q-margin ga fortsatt −218,62 Bps i samme hypotetiske sluttberegning.
Entry-MAE-hodet gjelder 95 minutter og har ingen direkte beslutningsmyndighet.
Exit mottar faktisk akkumulert MFE/MAE. Entry-Q-MSE var 2,66 % svakere enn konstant baseline.
SHORT-HOLD har null løpende finansiering og risikostraff. Innen dagens mål kan
ubestemt HOLD derfor dominere frivillig tapsrealisering. Flere epocher alene er
ikke dokumentert som løsningen. Avklar ønsket økonomisk holdetid og risikoramme,
og gjør så den minste nødvendige endringen i eksisterende økonomimål.
Ingen terskel, risikogrense eller modellmål er foreløpig endret.
Hele analysen og presiseringene står i docs/ENTRY_EXIT_REVIEW_20260914.md.

## Én vei for neste kjøring

NEXT_RUN_POLICY.json krever VAL-batch 256, åtte CPU-arbeidere og 10 800 sekunders
VAL-vinduer, 12 000 sekunders native budsjett og 13 800 sekunders ytre vakt.
Bevar market-cache, metadatacache, delt prissti for sider, batched økonomi,
FP32 uten TF32 og den laststyrte GPU-clock-launcheren. Alle features/familier/TF-er beholdes.
Målt CPU-trinn: +17,36 % med åtte arbeidere/batch 128, +24,65 % med batch 256;
identiske tilstandsbytes, fem repetisjoner, CPU 0–7. Dette er ikke GPU-/totalfart.
Før trening kreves risikoavklaring, faktisk GPU-paritet på 256, målt samlet fart
og eksakt resume-likhet. Profilen er ikke aktivert; ingen ny kampanje er laget.
Eksisterende kilde-/data-/checkpoint-/guardporter gjelder i tillegg.

30 epocher med patience 5 er fortsatt sluttmålet når dette er klart. TEST er
forseglet, ingen live/papir. Én tung jobb og timesvis modellobservasjon.
300 W / 85 C kjerne / 80 C minne / 12 GiB VRAM / 20 GiB RAM / 512 MiB swap.

## Opprydding og bevaring

28 gamle arbeidskopier er fjernet. Gjeldende kilde er GX1_CURRENT.
Elleve historiske kilde-/runtimekopier er beholdt fordi checkpoint- og datahistorikken
binder dem, eller fordi de eier Git/kjøremiljøet. De er ikke alternative oppstartsveier.
Ingen treningsdata, modellvekter, fullførte resultater eller checkpoints er slettet.
Gamle oppstartsnotater, ferdigbrukte forberedelsesbindinger og alternative
treningsgrener i den felles kjøreinnpakningen er fjernet fra gjeldende kilde.
Importerte moduler med gamle navn beholdes når feature- eller modellkoden bruker dem.

Gjenoppretting av fjernet kildemateriale: /home/andre2/GX1_ARCHIVE/CLEANUP_20260914/ALL_REFS.bundle.
Unike lokale endringer er lagret privat utenfor aktivt prosjekt.
Kvitteringer står i handover_snapshot/WORKTREE_CLEANUP_20260914.json,
CLEANUP_CODE_REMOVALS.json og TASK_RETIREMENT_20260914.json.
Arkivet skal ikke brukes som en alternativ treningskilde.

Mac-arbeidsområdet er også ryddet: 85 gamle toppnivåfiler/-mapper, 798 filer
og 252 889 123 byte er fjernet fra aktivt prosjekt etter kontrollert arkivering.
Hele trade_review_20260914 med VAL-resultat og analyser er beholdt.
Privat arkiv: /Users/andrekildalbakke/GX1_ARCHIVE/CLEANUP_20260914.
På Mac brukes bare ./handover.sh (--check eller --verbose); den leser GX1_CURRENT
via SSH og starter aldri trening. Scriptets versjonerte kilde er scripts/macos/gx1_takeover.sh.
