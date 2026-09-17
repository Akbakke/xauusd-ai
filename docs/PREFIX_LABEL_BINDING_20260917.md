# Native fasitbinding — 2026-09-17

Konkret blokkering: native Entry-dataset leste fortsatt ti policyavhengige
hjelpefasiter fra parentfilen, selv etter at nye prefix-fasiter var klare.
Disse gamle feltene var beregnet med policyer fittet gjennom kontrollperioden.

Minste rettelse er én eksplisitt binding i eksisterende EntryV10CtxDataset.
Den kontrollerer kilder, frosset design, policyopphav, kohort, original rad-ID,
tid og fasitens slutttid før ti kolonner erstattes i minnet. TRAIN og CONTROL256
bindes på separate instanser. Utenfor bundet kohort stopper getitem før input
bygges. Ugyldig binding endrer ingenting. Standardmodus er beholdt.

21 syntetiske CPU-kontroller består på første forsøk, under capped audit4G.
Ekte dataset/getitem leser riktige nye verdier; alle inputtensorer, de37 andre
aktive fasitene og låst radrekkefølge er uendret. Tester avviser feil kildehash,
design, policyopphav, rader, klokker, datogrense, datatype og verdidomene.
185 andre toppnivådefinisjoner og15 øvrige datasetmetoder er AST-uendret.
Ingen modell-/tapsfunksjon er endret. Originale data er uendret. Inaktive gamle
diagnostikkfelt er uttrykkelig ikke oppfrisket.

Dette er syntetisk integrasjonsbevis. En faktisk native forsøksoppskrift har
ennå ikke brukt bindingen. Prefix-normalisering, ferske modell-/lærer-/EMA-/
optimizertilstander og TRAIN-kilde for den låste senere kontrollen gjenstår.
Ingen fit, fasitberegning, modell-forward, optimizersteg, VAL eller TEST er kjørt.
Ingen ny læring, generalisering eller lønnsomhet er dokumentert. Ikke gjenta
ferdig fasitmaterialisering eller disse beståtte testene uten ny relevant feil.

Generalisering må måles etter låst forsøksoppsett. Mars–mai er fortsatt gjenbrukt
utviklingsdata; nye vekter gjør ikke perioden urørt. Ingen hendelsesregler,
tapsvektsøk eller justering mot kontrollutfall inngår i denne rettelsen.

Maskinbevis: handover_snapshot/PREFIX_LABEL_BINDING_SYNTHETIC_20260917.json.
Artifactrot: GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/
PREFIX_LABEL_BINDING_SYNTHETIC_20260917; original kilde, JUNIT og logg bevart.
