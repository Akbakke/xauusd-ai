# Frosset juni-VAL: kandidaten forkastes — 2026-09-17

Den frosne kandidatens TRAIN-forbedring overføres ikke robust til juni.
Exit-feilen forverres på begge sider i alle fem kalenderukene. Samlet målt
økonomi er negativ når kostnader og åpne posisjoner inkluderes. Ingen mer
omfattende trening, modellpromotering eller nytt fit på dette utvalget åpnes.

## Sammenligningen

256 forhåndsvalgte Entries, samme radidentiteter, mål, lærere, kostnadsregnskap
og originale checkpoint. Begge native evalueringer er fullført med guardPASS,
trainer0/observer0 og null optimizersteg. Samtlige Entry-targets, Exit-targets
og Exit-komponenter er kontrollert eksakt like. Exit-feil gjelder state0 per
Entry; faktisk native økonomi følger hele de valgte forløpene.

| MSE | Baseline | Kandidat | TRAIN-konstant |
|---|---:|---:|---:|
| Entry LONG | 116,4429 | 26,6648 | 65,6995 |
| Entry SHORT | 30,2332 | 32,3406 | 31,6672 |
| Exit LONG | 1408,8260 | 1625,0201 | 1413,7663 |
| Exit SHORT | 1410,8005 | 1553,6087 | 1416,1471 |

Entry LONG har også bedre sentrert MSE35,90→25,83. Entry SHORTs sentrerte
feil26,58→26,64 bedres ikke. Lærerregret faller5,7285→0,9612Bps, mens faktisk
økonomi er negativ. Bedre tilpasning til læreren er dermed ikke markedsfasit.

Exit-kandidatens LONG-middel er+9,1003Bps mot referansens−5,0737; SHORT−5,2287
mot+5,0157. Korrelasjon er henholdsvis0,0354 og−0,0063. Også sentrert Exit-MSE
blir verre: LONG1382,70→1424,12 og SHORT1384,55→1448,66. Svikt er derfor ikke
bare en konstant kalibreringsfeil. Ved state0 velger kandidaten HOLD i226/256
LONG og51/256 SHORT; referansens positive videreverdi gjelder119 og137.

| Uke | Exit LONG MSE før→etter | Exit SHORT MSE før→etter | Kandidatens merkede Bps per Entry |
|---|---:|---:|---:|
|23|757,35→898,54|759,82→792,81|−20,95|
|24|2134,09→2403,34|2144,95→2364,41|−2,20|
|25|1931,49→2123,43|1920,26→2079,31|−22,23|
|26|944,77→1146,79|947,07→1103,27|−24,20|
|27|1146,46→1564,41|1158,36→1318,40|+6,80|

## Hele økonomien

Baseline velger256FLAT. Kandidaten velger212LONG/15SHORT/29FLAT, altså227
handler i den uavhengige mulighetsvurderingen.223 er lukket; fire er åpne ved
splitgrensen og er verdsatt til observert eksekverbar likvidasjonsverdi etter
kostnader. Samlet−4031,4227Bps over256 muligheter gir−15,7477Bps per mulighet.
Dette er ikke en porteføljeavkastning. LONG bidrar−4530,2968Bps; SHORT+498,8742.
En separat, forhåndsdefinert én-posisjonsreplay utfører24 handler,23 lukkede og
én åpen:−1305,0322Bps i fast notional, uten rentesrente.203 signaler hoppes over
mens posisjonen er åpen. Ingen kunstig Exit eller maksimal holdetid innføres.

## Usikkerhet og konklusjonsgrense

Paret dagkluster-bootstrap med26 kalenderdager og5000 trekk gir95%-intervall
for Exit-MSE-forverring LONG[55,58;392,26], SHORT[37,46;251,27]. Kandidatens
merkede middel har intervallet[−42,76;+10,84]Bps. Dette er beskrivende usikkerhet
på gjenbrukt juni-VAL; overlappende handelsforløp og avhengighet mellom dager
gjør det ikke til bevis for fremtidig forventningsverdi. Uke27 er en kort uke.
Ingen robust handelsfordel er dokumentert, og gjennomsnittet alene åpner ikke
videre trening. TEST er ikke brukt.

## Neste arbeid og proveniens

Skjev Exit-verdsetting og svak overføring er målt. Én entydig rotårsak er ikke
bevist. Før en ny læringsendring: kontroller at eksisterende TRAIN-cache og
native VAL bruker samme representasjons-/målsemantikk. Gjenbruk artefaktene;
ingen nye fits, mål, markedsfiltre eller terskeltilpasninger mot juni.

Baseline-kilde89c og kandidat866e2e33 skiller kun i presisjonsvakt/logg i
kjørekoden; AST-likhet utenom vaktfunksjonen er verifisert. Kandidatens første
forsøk på boot450 stoppet før rollout på0,00048828125Bps batchavvik med like
handlinger. Absolutt grense0,001Bps, relativ0 og eksakt handlingslikhet ble
kontrollert i12 tester; gammel0,03Bps-feil avvises. Ny måling på boot451 bestod
batch-, cache- og CPU-pipelinekontroll. Vekter og targets ble aldri endret.

Maskinrapport: [FROZEN_NATIVE_COMPARISON_20260917.json](../handover_snapshot/FROZEN_NATIVE_COMPARISON_20260917.json).
Originalt resultat ligger under NATIVE_REFERENCE_POLICY_20260916_REFERENCE/
FROZEN_READOUT_GENERALIZATION_20260916/NATIVE_COMPARISON_20260917; SHA
c8e4fd44bfbac8719b4c44369d9cbfadcd1cae9c395ac087f2a5c8af142bb54e.
OPERATOR.py reproduserer analysen fra lagrede resultater uten modell-forward.
