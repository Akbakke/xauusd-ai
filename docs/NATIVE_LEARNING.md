# Native læring — gjenværende metode

Nåstatus, utført arbeid og aktuell hindring står bare i CURRENT_HANDOVER.md
og NEXT_RUN_POLICY.json. Dette er metodekrav, ikke en launchordre eller
historikk over daglige forsøk. Bruk eksisterende eiere og genuine resultater;
ikke relanser ferdige inputs, kapasitetstester eller engangsplaner.

## Før en modell kan startes

Aktuell ren/committet/pushet kilde, fysiske inputs, valgt målt sampler,
preprocessing og TRAIN-/CONTROL-koordinater må være eksakt bundet i den
kanoniske recipe/campaign. Et separat finite native-vindu og eksisterende
cgroup-/host-/GPU-vakter må faktisk godkjenne kjøringen.

Fysisk omstart krever at alle prosjektjobber, writers, låser og GPU-beregninger
på maskinen er ledige. Etterpå verifiseres ny Windows-boot, WSL, kilde og
artefakter. En terminal eller lav temperatur alene er ikke slik dokumentasjon.

Fersk nullstegs ONLINE/TARGET skal bruke samme nåværende funksjon og delte
moduler. Historiske checkpoints, constructor metadata eller identiske
vekthasher erstatter ikke genuine initial-/funksjonsparitetsmålinger.
Endret ONLINE-funksjon krever en ny faktisk initialbaseline.

## Omstart, mellomlagring og effektivitet

Operatøren ba 09.10 om hyppige trygge segmentgrenser og mellomlagring. Bruk
samme native campaign, checkpoint-eier og vedvarende inputs/cacher. Omstart
kommer etter varig checkpoint/terminal og dokumentert maskinvid ledighet.
Kontrolleren returnerer REBOOT_REQUIRES_MACHINE_WIDE_IDLE_REVIEW ved denne
grensen. Agenten kontrollerer Windows/WSL-prosesser, writers, prosjektlåser
og GPU før eksisterende prepare-reboot/confirm-reboot og fysisk SSH-omstart.
Neste Windows-oppstart fortsetter bare det allerede bundne finite vinduet.

Kildekontroll: deterministic_fp32 lagrer hver 64 optimizersteg, ved tidsgrense
og ved avgrenset slutt. Checkpointet bevarer ONLINE/TARGET, optimizer, EMA, scheduler, RNG,
epoch-order, neste batch og fremdrift. Eksisterende to-slot-eier bruker
temporær fil, fsync, atomisk rename, hash og atomisk aktiv peker. Diskcacher
gjenbrukes; RAM-/GPU-cache må lastes på nytt. Ved krasj kan arbeid etter siste
fullførte checkpoint gå tapt. Gjeldende resume-evidens og dens avgrensning står i CURRENT_HANDOVER.md.

Operatøren valgte 09.10 eksisterende 12000s (3t20) etter å ha opplevd heng
etter 12 timer. Behold denne driftsgrensen; ingen forkorting er nødvendig.
Tidspause vurderes etter hvert fullførte optimizersteg og skjer først etter
varig checkpoint. Windows-oppgavens grense er 4t slik at native 3t20,
ytre guard 3t50 og terminalbokføring får plass. Smoke skal måle step-/checkpoint-tid,
kald gjenlasting og samme neste batch/tilstand etter resume, og kontrollere
at siste batch, lagring og terminal får plass innen eksisterende ytre vakt.
Rapporter beregningstid og lagre-/laste-/oppstartstid separat.
Ingen kalenderjobb avbryter aktivt arbeid. Omstart beviser ikke løst BSOD-årsak.

## Gjeldende M5/MTF/MBP-læring

Prosjektmålet i GX1_RULES.md er M5 Entry med lukket M15/H1/H4/D1-kontekst,
alle genuine indikatorer og de eksisterende26månedene med MBP-1. Åtte
spesialister og niende D1/makro-kontekst beholdes. Alle ruter, masker,
formler, tidsrammer og faktisk konsum gjennomgås før ny modellkvalifisering.

Den native epoken som er bestilt17.–18.10 erstattes ikke av en kort CPU-
studie eller en gammel256-stegsmetode. Små mekanikkprøver kvalifiserer
mekanikk; de avgjør ikke om en fullt trent Transformer kan ha verdi.
En epoke må omfatte hver kvalifiserte TRAIN-rad nøyaktig én gang, med faktisk
partial siste batch, varige checkpoints og det eksisterende finite vinduet.
Ingen automatisk ekstra epoke, HPO eller seleksjon på evalueringsutfall.

Operatøren valgte18.10 eksplisitt M5-prisforløp og liten MAE før MFE som
hovedmål, med lengre horisonter som støtte. Entry-target, tap, handling og
evaluering må bindes til dette før launch. Et h20-terminalmål måler20observerte
sesjoners nettoavkastning og er ikke lenger hovedmålet. En targetendring krever én navngitt
kontrakteier med kausale executable BID/ASK-utfall og deklarerte kostnader,
ikke en gjettet horisont/vekt eller hindsight-valgt optimal exit. MFE/MAE,
terminalavkastning, timing og handelsfrekvens rapporteres separat.

I den eksplisitte `m5_primary_observed`-emisjonen styres den fysiske
populasjonen av gyldig M5-path og genuine input-/historikkrav. Komplette
legacy-M1-mål og unionen av37hjelpemål er ingen primærgate. Fire M5-
prognoser K1/5/12/24 og seks LONG/SHORT1/5/20d-støtteceller har hver sine
observasjonsmasker. Manglende target lagres0 bakfalsemaske; det er ingen
observasjon. Inaktive hoder og lengre støttehorisonter får ikke redusere
primærpopulasjonen. Legacyemisjon beholder sine egne komplette målkrav.
Endret populasjon krever ny faktiskTRAINklokke, ferskTRAINregistrytilpasning
og gjeldende immutable kontraktbindinger før inputadmission.

Første reelle smoke skal bevise samme fysiske batch gjennom alle familier,
GC-value/mask-par, alle MTF, normalisering og modell, samt fersk initialbaseline
og train/serve-paritet. Entry-only-fasen må bevise Exit-grad=None og uendrede
Exit-vekter. Bruk allerede beståtte uendrede kontroller; en ny funksjon krever
ferske funksjonsbevis. Nøyaktig nåstatus og eventuelt uavklart målvalg finnes
bare i CURRENT_HANDOVER.md/NEXT_RUN_POLICY.json.

## Bevarte datakrav og økonomi

Alle genuine featurefamilier, ordnede felt, M1/M5/MTF-klokker, targets og
normaliseringseiere består. Geometri, reference-policy og cutoff hentes fra
de faktiske bundne eierne, ikke gjettede defaults eller gamle dokumenttall.
Normalisering tilpasses én gang på hele fysisk TRAIN; ingen VAL/TEST-refit.

Original target-M1 og nyere komplett input-M1 er ulike kilde-/artefaktroller.
Gjenbrukt uendret-kilde-/algebrabevis er ikke en ny raw quote-måling.
Ukjente gap forblir ukjente/right-censored; ingen carry, imputering eller
nye forskningsperioder. Ingen fast tapsgrense eller maksimal holdetid.

Senere økonomi må inkludere alle valgte handler, åpne posisjoner, kostnader
og én-posisjonskapasitet mot forhåndsvalgt samme-risiko alltid-LONG.
Lokale prospective kostreceipts er ikke historisk kostfasit.

Regler: GX1_RULES.md. TEST, broker, live/paper, handel, spending og promotion
er stengt. GC og andre forskningsarmer åpnes ikke av denne metoden.
