# Entry-utfall atskilt fra lærerens estimat — 2026-09-16

Eksisterende TRAIN-cache dekker verifiserte Entry-ankre for 275 av 512 Entries
(53,71 %), fordelt over alle tolv måneder. De øvrige 237 mangler ankernære
forløp i disse cachene. Ingen ny materialisering eller modellkjøring ble brukt.
Ankerkontrollen tok 26,12 sekunder på CPU under audit/4 GiB/512 MiB-swap.

## Identitet og økonomisk avgrensning

Kildens stibygger gir lengde min(state_index + 1, tail_capacity). Lengde 1
identifiserer derfor tilstand 0. Alle 275 slike online-tilstander ble dessuten
sammenlignet eksakt med samme Entry-eiers læreranker: lokale/MTF-inputs,
kontekst, levetidssammendrag, gyldige stilengder og handlingsmasker. Alle har
fem tilgjengelige etterfølgende overganger; ingen duplikate Entries.

Målingen summerer første likvidasjonsverdi og diskonterte observerte relative
rewards, uten lærer-bootstrap. Kildens identitet er fysisk HOLD-reward +
gamma * neste likvidasjonsverdi - nåværende likvidasjonsverdi. Dette er
kostnadsført treningsnytte, ikke rapportert udiskontert kontant-PnL eller
strategiprofitt. Alle de 275 inngangene inngår ved samme femte beregningssteg;
ingen etterpåvalg av beste utgang. Fem steg er en diagnose, ingen handelsgrense.

## Hva utfallene faktisk viser

| På 275 Entries | LONG | SHORT |
|---|---:|---:|
| Gjennomsnittlig observert nytte, Bps | -4,28658 | -7,46095 |
| Positive observerte utfall | 67 | 43 |
| Positive lærermål ved Entry | 24 | 15 |
| Ikke-positivt lærermål, senere positivt utfall | 48 | 34 |
| Referanse96 korrelasjon med observert nytte | -0,07705 | 0,28226 |
| Kandidat96 korrelasjon med observert nytte | -0,07818 | 0,28301 |

Positive enkeltutfall beviser ikke at læreren skulle ha forutsett dem. Et
betinget verdiestimat kan være negativt selv om enkelte realisasjoner vinner.
Gjennomsnittet er fortsatt negativt for begge sider. Bare LONG i mars 2026
har positivt månedsmiddel i dette utvalget. Ingen lønnsom kausal regel er påvist.

| Entry-MSE mot samme observerte nytte | LONG | SHORT |
|---|---:|---:|
| Før95 | 110,43174 | 110,40629 |
| Referanse96 | 109,88829 | 109,62631 |
| Forkastet klippekandidat96 | 109,90244 | 109,61936 |
| Konstant lik samme utvalgs sidemiddel | 107,08131 | 110,65141 |

Det finnes en liten forbedring fra95 til96 på disse allerede trente radene,
og SHORT har noe tilstandsavhengig rangering. Vi skal derfor ikke si at all
betinget læring er fraværende. LONG slår ikke den beskrivende konstantbaselinen.
Alle tre modeller velger FLAT på alle 275. Kandidaten gir ingen relevant
forbedring over referanse96; tidligere separat TRAIN-forverring gjelder fortsatt.
Konstantbaselinen er tilpasset de observerte utfallene i samme utvalg og er
ikke en validert modell eller handelsstrategi.

Maskinrapportens beslutning NO_OBSERVED_ENTRY_LEARNING_IMPROVEMENT_PROVEN betyr
at læringsporten ikke er bevist, ikke at de små MSE-forbedringene over er null.
Dette er TRAIN, ikke en holdout. Utvalgsforskjell må beholdes: lærerens LONG/
SHORT-middel er -5,103/-6,625 Bps blant de 275, mot -5,714/-5,854 blant de 237
uten ankercache. Månedstall og alle tre armer finnes i maskinrapporten.

## Neste beslutning

Behold treningsstoppen og gjenbruk målingene. Vi har nå skilt observerte utfall
fra lærerestimater; flere like kontroller skal ikke gjentas. Neste arbeid må
velge og begrunne én konkret korreksjon i lærer-/verdimålkjeden med disse
bevisene. En foreslått målendring må forklare hvilken forventet handlingsverdi
den estimerer, hvordan kausal policy skilles fra etterpåvalg, og hvorfor den
retter den målte skjevheten. Positive etterpåutfall, konstantkalibrering eller
vilkårlig tvungne fem steg er ikke tilstrekkelig begrunnelse. Ingen ny trening,
lærerrefresh, optimizerreset, terskel-/modell-/tapsvektsøk eller full VAL åpnes.

## Bevis

Autoritativ kilde: /home/andre2/src/GX1_CURRENT ved 8691899457d9dfdd2405d3d37e759a25fddc7576.
Resultatmappe: BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/ENTRY_ANCHOR_CACHE_AUDIT_20260916.
RESULT.json SHA256: 22fcb932cd106175e01add52910cbdf279616b7a90ba2d30b12304a0f2c9e304.
OBSERVED_OUTPUT_SUMMARY.json SHA256: 90de1ac8c75b9a0fe8456682938766ecd41c2bf2c277b500c332a64634026e7a.
Kopier ligger under handover_snapshot/EXIT_PRIVATE_CLIP_20260916/.

Oppsummeringens første forsøk stoppet på et felt som manglet i den eldre
førstebatch-JSON-en. Den rapporterer nå lærerfeltet som finnes i alle batcher;
ingen input, mål eller prediksjon er endret. Originaloperatoren er bevart.
Ingen GPU, backward, trening, nye data, VAL, TEST eller ordre.
