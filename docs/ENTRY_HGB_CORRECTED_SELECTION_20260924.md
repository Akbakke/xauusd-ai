# Korrigert HGB-kapasitetsvalg — 24.09.2026

## Konkret avklaring

Mønsterforsøket med1311 inputfelt hadde arvet iterasjonstall valgt uten sine235
mønsterfelt. Dessuten var reparert indre purge/iterasjonsvalg ikke kjørt.
Den isolerte full-refit-kontrollen og mønsterkontrollen dokumenterte dette
eksplisitt; de var ikke en komplett kontroll av den rettede kjeden.

Denne ene kjøringen fullførte eksisterende `fit_hgb` med korrekt indre
tidsdeling og påfølgende full-fold-refit. Ingen produksjonskode eller native
modell ble endret. Ingen nye tester eller regresjonssuiter ble kjørt.

## Låst sammenligning

- Samme313399 TRAIN-rader og1311 felt; faktisk inputarray, tider og ATR er
  SHA-like forrige mønsterforsøk. Tidligere full TRAIN-prefix-kontroll av de235
  mønsterfeltene er gjenbrukt.
- Samme h12/ATR, seed0, læringsrate0,1, minste blad20, indre20% og purge289.
- Eksisterende ramme1–300 trær; valget bruker bare den indre kronologiske
  TRAIN-kontrollen. Ingen senere årsutfall brukes til kapasitetsvalget.
- Fire opprinnelige senere TRAIN-år, ikke juni-VAL. Ingen VAL-dataset lastet,
  nye juni-beslutninger eller TEST-tilgang.
- Åtte indre fits og åtte full-fold-refits. Tidligere paired predictions er
  referanse; ingen refit av en ekstra baseline.
- Samme argmax(LONG,SHORT,FLAT=0). Samtlige valgte h12-markeringer er med.

Native source, vekter, checkpoint og Exit-policy er urørt.

## Resultat

| Senere TRAIN-periode | Valgte | Netto bps per valgt | Valgte trær LONG/SHORT |
|---|---:|---:|---:|
| juni2022–mai2023 |1334|−1,754|1/1|
| juni2023–mai2024 |16196|−5,468|1/3|
| juni2024–mai2025 |491|−7,804|1/1|
| juni2025–mai2026 |33|+0,241|1/1|
| Samlet |18054|−5,246| |

Alle258048 tidspunkter er paret. Etterberegningen kontrollerte **bitlike**
LONG-prediksjoner, SHORT-prediksjoner, handlinger og netto markeringer mot
forrige1311-felts forsøk: null avvik i hver kolonne. Valgte iterasjonstall
er identiske med de tidligere arvede tallene.

Netto per mulighet: −0,367049bps. Paret forskjell fra forrige forsøk:0.
48 kalenderblokker,2000 resamplinger,seed0, intervaller justert for fire
kontraster innen denne avlesningen:
- Mot forrige forsøk:0, intervall[0;0].
- Mot FLAT:−0,367049, intervall[−0,777472;−0,067394].
- Mot alltidLONG:+5,083702, intervall[4,577271;5,541605].
- Mot alltidSHORT:+5,596093, intervall[4,977090;6,150206].

Bedre enn to tapende konstantstrategier er ikke positiv økonomi.
Videreføringsporten feilet:
`STOP_THIS_H12_ATR_SELECTION_VARIANT_NO_RETUNING_OR_PROMOTION`.

Dette avklarer at arvet kapasitet ikke forklarer resultatet for denne faste
konfigurasjonen. Det avviser ikke alle tekniske kombinasjoner eller modeller.
Ingen ny juni-tilpasning, native trening eller utvidelse følger automatisk.

## Omfanget av økonomien

Historiske close-fill h12-mål, inklusive spread, pluss arkivert scenario:
2bps per utførelse,0 kommisjon,5,4% årlig LONG-finansiering og0 SHORT.
Faktisk klokketid brukes. Ingen bekreftelse av aktuelle meglervilkår.
Hypotetiske overlapp er inkludert; tallene er ikke en kapitalført portefølje,
neste-open-utførelse eller native Exit-livsløp. h12 er et målevindu, ikke en
ny maksimal holdetid. Alle periodene er gjenbrukt utviklingsdata.

## Hva som allerede er prøvd

Gjenlesingen av ENTRY_DIRECTION_FIRST_20260923.md bekrefter at direkte
native retningslæring også er gjennomført:
- L1-forecast:512steg, ingen horisont bestod.
- BCE på forecast-representasjon:512steg, ingen horisont bestod.
- BCE på Entry-fusjonen:512steg, ingen horisont bestod.
Totalt1728 native optimizersteg inklusive de tidligere kontrollene.
Direkte retningsfeedback er derfor ikke et uprøvd neste tiltak.

Neste arbeid må knyttes til ny konkret evidens. Bevar de avsluttede forsøkene
og gjenbruk deres caches. Ingen ny tapsvariant, økt kapasitet eller annen
målehorisont begrunnes av at dette forsøket feilet.

## Drift og sporbarhet

Kilde: GX1_ENGINE, audit/v9-premiere-20260905,
b4de26b9e8062efd0b9e6cd535be21b01562c035, clean og frosset under kjøring.
Rot: `/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/HGB_CORRECTED_SELECTION_20260924`.

Terminal rc0 kl.15:38:39UTC /17:38:39Oslo;1423,25sekunder totalt.
Ingen timeout. Producer10GiB,512MiB swap, én numerisk tråd og eksisterende
eksklusiv lås. Ingen CUDA. Etterberegning rc0 under audit4GiB.
RESULT SHA d31ed03920d531315e697c30ff2c2aeca05321c68d8445c33d7af616245a1d09.
TERMINAL SHA aa666d253ac7d97b67ab4bc37f054257d64a91b00f7455d6db517bcc2796ff7a.

PLAN, START, INPUT_BINDING, fold-/side-rapporter, RESULT, TERMINAL,
COMPLETED_REVIEW, metrics, script og logg er bevart. Private radprediksjoner
forblir på Linux. Ikke relanser. Målet om økonomisk nyttig Entry er ikke oppnådd.
