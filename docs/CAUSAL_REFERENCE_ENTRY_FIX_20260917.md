# Kausal Entry-fasit — 2026-09-17

En konkret Entry-fasitfeil er påvist og rettet: max(observert HOLD-utfall,0)
valgte første handling med kunnskap om framtiden. Dette var en feil i den
tidligere «coherent reference»-rettelsen. Korrekt forventning under den
allerede bundne kausale referansepolicyen er (119/120)*HOLD-utfallet; negative
utfall beholdes. Første gjennomførbare likvidasjonsverdi legges til som før.

På eksisterende TRAIN256 løftet feilen Entry-fasiten med 8,6852 Bps i snitt.
Dette er en dokumentert fasitskjevhet, ikke bevis for at den forklarer all
svak læring. Ingen framtidslekkasje i modellinputs er påvist av denne testen.
33 målrettede syntetiske tester består. Ingen ny faktisk modell-forward,
targetcache eller trening er kjørt. Exit-HOLD-fasit og tidligere resultater
er uendret og bevart; ingen lærings- eller økonomiport er bestått.

Neste er én eksplisitt avledet TRAIN-baseline med de lagrede prediksjonene,
HOLD-målene og kanonisk første likvidasjonsverdi. Gamle Entry-targets må ikke
gjenbrukes eller omdøpes til korrekt fasit. Ingen ny trening er åpnet før
sammenlignbare targets og bindinger er kontrollert. Ingen CONTROL/VAL/TEST.
Se docs/CAUSAL_REFERENCE_ENTRY_FIX_20260917.md. Native-tasken er avsluttet;
ingen aktiv modelljobb. NEXT_RUN_POLICY.json angir tillatt forberedelse.

## Bevis og minste rettelse

Samme observerbare tilstand med to like sannsynlige framtider gir HOLD +10
eller −10 Bps og første likvidasjon −1 Bps. Gammel pathwise max gir +4 Bps
forventet Entry-target for begge sider. Kausal referanseforventning gir −1.
Dette gjelder uavhengig av dato eller markedsregime, ikke en tilpasning til
spesielle handler. En maks over realiserte utfall er ikke en maks over
betingede forventningsverdier.

| TRAIN256 | Gammelt Entry-middel | Kausalt rekonstruert middel | Kunstig løft |
|---|---:|---:|---:|
| LONG | 3,8569 | −3,7500 | 7,6068 Bps |
| SHORT | 1,8035 | −7,9601 | 9,7636 Bps |

105/256 LONG- og 151/256 SHORT-HOLD-utfall var negative. Tabellen bruker
likvidasjon rekonstruert fra lagrede float32-targets i float64 og er en audit,
ikke en ny bitlik native targetcache. Den første auditens unødvendige antakelse
om at all første likvidasjon er negativ feilet; første markedsbevegelse kan
gi positive verdier. Feilet operatør er bevart, og revidert audit måler dette.

Kun eksplisitt reference-Entry bruker nå policyforventningen i felles eier
for TRAIN og evaluering. Legacy critic-max, Exit-HOLD-mål, reward, gamma,
kostnader, kausale inputs, terminal/bootstrap, features og arkitektur bevares.
Diagnostikkens tekst skiller multi-step-remainder fra ren lærerbootstrap.
Referanseverdien er Q_mu/V_mu under fast policy, ikke optimal handelsverdi
eller dokumentert profitt under modellens senere greedy-policy.

## Kontroll og videre arbeid

33 berørte CPU-testtilfeller består under eksisterende audit-cgroup. De dekker
symmetriske framtider, negative utfall, terminalmasker, ugyldige policybindinger,
eksakt TRAIN/eval-paritet og uendrede Exit-targets. Ingen fullsuite gjentas.
Originale checkpoints, mål, cacher og fullførte målinger bevares.

Den gamle start-/sluttmålingen krever eksakt identiske targets. Den må ikke
omgås: lagret modelloutput kan gjenbrukes, men korrekt Entry-target må bindes
til kanonisk likvidasjon og uendret HOLD-mål. En avledet baseline skal merkes
som avledet, aldri som en ny native måling. Neste omfang er bare TRAIN-baseline-
forberedelse; native forward/trening krever egen konkret plan i gjeldende policy.

Maskinbevis: handover_snapshot/CAUSAL_REFERENCE_ENTRY_FIX_20260917.json.
Ingen generalisering eller positiv kostnadsjustert økonomi er dokumentert.
