# GX1 arbeidsmål — oppdatert 26.09.2026

Målet er en ærlig XAUUSD-bot som tar retning på den tidsskalaen der retningen faktisk
finnes, og som slår relevante baselines etter kostnad. Målet er aktivt og ikke oppnådd.

## Operatørvedtak 26.09.2026

- **Én kodebase:** `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
  `audit/v9-premiere-20260905` (GX1_ENGINE) er arkivert; det som har verdi derfra slås inn
  her. Aldri to parallelle spor igjen (se AGENTS.md).
- **Retning på dager–uker, ikke 95 minutter.** Målt i
  [docs/DIRECTION_TIMESCALE_20260926.md](docs/DIRECTION_TIMESCALE_20260926.md): trenden er
  ~1,6 % av bevegelsen per 95-minutters vindu og blir dominerende først på uker–måneder;
  svingninger fortsetter med sannsynlighet 49–51 % når de sees. Retningsmålet hadde et
  hardt tak på 8 timer (96 M5-barer).
- **Beslutningsklokke:** M5, eller en høyere tidsramme (H1/H4/D1) hvis målingene viser at
  den er bedre. M5/M1 brukes til timing av inngang i trendens retning med lav MAE.
- **Datasett bygges på nytt** når det nye målet er definert og målt; det er autorisert.

## Suksesskriterier, i rekkefølge

1. Walk-forward på eksisterende features med ukeshorisont: retning *utover drift*, målt mot
   alltid-LONG / kjøp-og-hold valgt før hver periode, per år, med ærlig antall uavhengige
   uker. Myntkast er ikke referansen på lange horisonter.
2. Bare ved robust resultat: ny målkontrakt, rebuild av datasettet og native trening.
3. Økonomi med alle valgte handler og åpne posisjoner, BID/ASK og kostnader, risikojustert
   mot kjøp-og-hold. TRAIN-fit, senere generalisering og samlet økonomi rapporteres hver for
   seg. Konstant bias, all-FLAT/all-HOLD og bedre hjelpeprognoser alene er utilstrekkelig.

## Bevares

Alle features, alle åtte familier, alle tidsrammer og kausale inputs. Gjennomførbar
BID/ASK-økonomi og kostnader. Ingen fast tapsgrense eller maksimal holdetid; en
beregningshorisont er ikke en handelsregel. TEST er forseglet; ingen live/paper eller
spending. Ingen modell loves å være lønnsom «evig».

Én agent og én tung jobb om gangen, gjennom `scripts/gx1_capped_run.sh` og eksisterende
vakter. Ingen blind trening, søk eller forebyggende refaktorering; mål før du bygger.
Stående publiseringsautorisasjon gjelder ferdig kode, dokumentasjon og aggregater; rådata,
vekter og hemmeligheter publiseres aldri.
