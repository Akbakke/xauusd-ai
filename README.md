# GX1

Målet er en fullautomatisk XAUUSD-bot med M5 Entry og kausal, lukket
M15/H1/H4/D1-kontekst. Kvalitet før antall: bullish LONG, bearish SHORT og
minst mulig MAE før gunstig MFE. Entry lærer prisforløp fra ett minutt til
mange timer; dager er støtte. LONG/SHORT/FLAT og avståtte muligheter skal
kunne etterprøves. M30 vurderes når tilført verdi kan måles.
Arbeidet nå er offline datakvalifisering og læring; handel er ikke lansert.

Først brukes de eksisterende26månedene med GC MBP-1 sammen med XAUUSD-priser.
Alle genuine indikatorer skal kontrolleres individuelt og følges gjennom
samtlige tidsrammer til faktisk modellkonsum. Åtte spesialistencodere og
niende D1/makro-kontekst samarbeider i samme native modell. Denne bølgen
bygger og kvalifiserer Entry; Exit vurderes eller trenes ikke. Separate
SMC-/MBP-modeller eller ensemble krever målt merverdi i samme tidsordnede
out-of-fold-oppsett før de eventuelt erstatter hovedsporet.
MBP-1 bidrar med observerte trades, BBO, order counts, flow, volumprofil og
footprint. Manglende dybde, aggressor eller historikk konstrueres aldri.

Eneste arbeidsrepo er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Les AGENTS.md og GX1_RULES.md før arbeid. Start med:

```bash
bash scripts/gx1_handover.sh --check
```

Skriptet viser det varige prosjektmålet, faktisk siste terminal, nåstatus og
kjøregrenser. CURRENT_HANDOVER.md samler siste gjort/ikke gjort/hindring/neste
steg. NEXT_RUN_POLICY.json eier eksakte bevisbindinger og arbeidsstatus.
Flere agenter får avgrensede oppgaver; én integrator og én capped tung jobb.

En kort regel-/CPU-/smoketest gir ingen generell konklusjon om prosjektets
muligheter. Kontroller data, TRAIN-fit, senere generalisering og nettoøkonomi
hver for seg. Originalkvitteringer og checkpoints bevares; Git eier historikken.
Ingen TEST-tilpasning, live/paper, broker, ordre, spending eller promotion.

Dokumentasjon: DOC_INDEX.md. Eiere: SYSTEM_MAP.md.
Datakrav: docs/DATA_CONTRACT.md. Metode: docs/NATIVE_LEARNING.md.
Indikator-/dekningsrevisjon: docs/FEATURE_AUDIT.md.

Kilde, kontrakter og fokuserte tester versjoneres. Rådata, vekter,
hemmeligheter og kjøringsoutput holdes utenfor repoet. Bruk
scripts/gx1_capped_run.sh og eksisterende maskinvarevakter. Stabile jobber
observeres omtrent hver15.–30.minutt eller sjeldnere; trygg omstart skjer
bare mellom fullførte jobber etter dokumentert maskinvid ledighet.
