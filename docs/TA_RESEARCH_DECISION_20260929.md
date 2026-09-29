# A/B/C — forskningsplanen er avsluttet, 29.09.2026

**Ingen av armene ga GO til ny native trening eller utførelsesforskning.**
Den avtalte planen er gjennomført til dokumenterte beslutninger eller eksplisitt
kildebegrensning. Målet om en lønnsom bot er fortsatt ikke oppnådd.

| Arm | Hva som er avsluttet | Beslutning |
|---|---|---|
| A: sju D1-felt, ridge/HGB |15 årsfolds per horisont; kost-, finansierings-, risiko- og inferensinstrumentene reparert | Begge INKONKLUSIV |
| B:12 makrofelt fra seks kilder | Manifestbundet kildeport; ingen godkjent full historisk versjonskjede for GLD/COT | IKKE MÅLT — KILDEBEGRENSNING |
| C: én kombinasjon av fem celler |1 649 muligheter på juni2025–juni2026; aktive/passive kostnader og fyllingsutvalg målt | INKONKLUSIV, negativ observert netto |

## Hva vi faktisk lærte

[A](TA_A_RESULT_20260929.md) gir ikke dokumentert merverdi mot de nødvendige
baseline-armene. Etter finansieringsproxy ga h20 ridge14,50 %, HGB59,21 % og
samme risiko-LONG/konstant61,04 % samlet over måleperioden. Ridge/HGBs
gjennomsnittlige meravkastning mot LONG var−0,669/+0,018 bps per intervall.
Simultane grenser var henholdsvis[−4,241;2,903] og[−2,147;2,183].
Dette er ulikt et bevis for at indikatorer aldri kan gi informasjon.

[B](TA_B_RESULT_20260929.md) har ingen målt prognoseeffekt. ALFRED-transporten
timed ut på forskningsmaskinen; Mac kunne lese skjemaet. Den avgjørende
begrensningen er manglende bundet historisk publikasjon-/versjonsbevis for
GLD og COT. Dagens historiske sluttserie med et konstruert fast lag oppfyller
ikke kontrakten. En firekilders erstatning ble ikke kjørt.

[C](TA_C_RESULT_20260929.md) viser hvorfor brutto fortsettelse ikke alene er
en strategi. Aktiv inngang hadde+1,465 bps mid-bevegelse per valgt mulighet,
men−2,547 netto ved1 bps per utførelse og finansiering. Den passive modellen
reduserte kostnadene, men valgte et svakere utvalg: mid−0,751 bps blant
berørte ordre mot+9,536 blant utførbare ordre uten berøring. Samlet passiv
netto var−1,397 bps per valgt mulighet og−14,91 % av initialkapital.
Simultan grense mot FLAT[−4,480;1,686] gir INKONKLUSIV etter den fryste regelen.

A har96 og C72 endepunkter i hver sin deklarerte korreksjonsfamilie.
Begge bruker gjenbrukt utviklingshistorikk; korreksjonen opphever ikke gammel
seleksjon eller VAL-gjenbruk. B får ingen konstruert inferens uten data.

## Neste operatørbeslutning

Anbefalingen er å beholde training_enabled=false og avslutte dette prisbaserte
forsøksløpet uten nye terskel-/modell-/indikatorsøk. Ingen v37-build, optimizer,
native trening, full native VAL, TEST-utfall, live/paper eller spending følger
av disse resultatene. Alle native featurefamilier og checkpoints bevares.

Den konkrete mulige gjenåpningen er en **databeslutning for full B**:
kan et navngitt arkiv dokumentere originale og reviderte GLD-/COT-verdier med
faktisk tilgjengelighet? Først når dette er bevist og manifestbundet gir det
mening å løse transporten, fryse felles A/B-populasjon og registrere en ny fit.
Ingen leverandør er valgt, ingen kjøp er gjort, og ingen merverdi er lovet.

C ga ikke den avtalte GO til ordrebok-/utførelsesforskning. Resultatet kan
begrunne en framtidig særskilt hypotese om fyllingsutvalg, men er ikke en
tillatelse til å optimalisere passive grenser på den samme perioden.
A ga ikke grunnlag for å innføre en ny native mål-/horisontkontrakt nå.

## Tre evidensklasser

- **Målt:** A/C på deklarerte perioder, kostnader, observerte nettoresultater,
  usikkerhet/styrke og Cs utvalg ved quote-berøring. B-kildeadgang er målt,
  men B-modellverdien er umålt.
- **Bevist teknisk konsistent:** de reparerte forskningsinstrumentene,
  fokuserte syntetiske kontroller, kilde-/manifestbindinger, terminale
  kvitteringer og artefakter; Cs berøringer og kontantregnskap er
  etterkontrollert mot cache/kilde. Dette beviser ingen lønnsomhet.
- **Ikke undersøkt:** faktisk utførelse, urørt framtidig generalisering,
  full B med gyldige dataversjoner og native læring under et eventuelt nytt mål.

Fullførte målinger skal ikke relanseres. Den operative statusen og neste
beslutningsgrense står i CURRENT_HANDOVER.md og NEXT_RUN_POLICY.json.
Rådata, modellvekter og detaljerte signal-/prisfiler publiseres ikke.
