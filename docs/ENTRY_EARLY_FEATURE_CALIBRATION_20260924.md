> Oppdatert: regenereringen er fullført med rc0 24.09 kl. 16:55:43 UTC.
> Se [fullført inputrettelse og neste sammenligning](ENTRY_EARLY_INPUT_COMPARISON_20260924.md).
> Avsnittet om neste avgrensning nedenfor beskriver den daværende planen.

# Tidlig featurekalibrering og lukket bargrense — 24.09.2026

## Ferdig parameterpakke

Eksisterende eiere har tilpasset nivåer, trendlinjer og seks volatilitetsklokker
på eksisterende historikk fra 01.01.2019 kl. 23:00 UTC til 28.03.2022 kl. 22:00
UTC, halvåpent. Sluttgrensen er siste felles dagsgrense før første indre
modellkontroll 29.03.2022 kl. 00:15 UTC. Ingen sluttgrense er valgt fra PnL.

229371 M5-rader. Registry indre grense 04.08.2021 kl. 22:00 UTC er avledet av
samme 80-prosentdeling og eksisterende dagsklokke. Kjøringen brukte den
eksisterende producer-vakten: 10 GiB, 512 MiB swap, én numerisk tråd,
CPU 0–7 og 1800 s tidsramme. Ferdig rc0 kl. 16:29:40 UTC, 389,00 s.
Kjørekilde: 94e75f9954d9c3ee7e83e7666563849b96befb50.

Parameterpakken er validert av eksisterende eiere og ligger før første
indre kontroll. Dette er ikke en regenerert featureflate eller Entry-læring.

- Registry contract SHA: 155cf4da0a7523a5485c2cd6748f610270f5eae85b1d67fd48c7bda4d3ce20a4.
- Volatilitet manifest SHA: 27fd89ccfb8f19b2df3f2899ef32b086a6649680df213dbdb25f243de59455f0.
- D1 nivåstøtte: 131 fit-hendelser og 31 kontrollhendelser. Valgte grener
  har 16/115 henholdsvis 10/21 hendelser og består eksisterende støttekrav.

Det første korte vinduet fra juni 2021 hadde bare 212 D1-barer, 30 fit- og
seks kontrollhendelser. Ingen terskel oppfylte grenstøtten. En avgrenset
replay uten terskel-fit lokaliserte dette; M5/M15/H1/H4 hadde støtte.
Derfor ble hele den allerede tilgjengelige tidligere historikken brukt.
Støttekrav, terskeleiere og sluttgrense ble bevart. Den innledende tuple/dict-
feilen i kjøreskriptet stoppet før noen tilpasning og er bevart separat.

## Ytterligere feil: åpningstid var brukt som datagrense

Den gamle volatilitetskalibreringen sluttet deklarert 31.05.2026 kl. 23:55 UTC,
men valgte ferdige barer etter åpningstid. H4/D1-populasjonene er rekonstruert
med nøyaktig samme OHLCV-hash som de lagrede parameterfilene:

| Klokke | Rader | Siste åpning UTC | Siste lukking UTC |
| --- | ---: | --- | --- |
| H4 | 7736 | 31.05.2026 22:00 | 01.06.2026 02:00 |
| D1 | 1290 | 31.05.2026 22:00 | 01.06.2026 22:00 |

H4 SHA: 763f22b3440bfe21baf1d4fca5af36cb595eced7c9a2c93f08a8507f948ddb47.
D1 SHA: 9928f707bf808f4a8d9241176d06cf4fcf10668c9766ad0f2f193f3f5392debf.

Begge brukte dermed priser fra juni. Den tidligere rene datokontrollens
juni-PASS var for svakt; også juni påvirkes av denne forbehandlingsfeilen.
Det viser ikke at feilrettingen vil gjøre resultatene positive.

## Minste rettelse og kontroll

Volatilitetseierens manifest-fit velger nå bare barer som er lukket ved
sluttgrensen. Den underliggende fitteren avviser også en bar som lukker senere.
Ingen runtime-featureformel, modellarkitektur eller handelsregel er endret.

Forskningsporten beholder deklarert fit-slutt og registrerer i tillegg en
konservativ øvre grense for når gamle volatilitetsinputs kunne være tilgjengelige.
Gamle manifester mangler siste observerte fit-bar; grensen utledes derfor fra
den største klokkens lukking. Den kan være strengere enn nødvendig ved hull i
kilden, og er ikke et oppdiktet observert tidspunkt. En felles lukket dagsgrense
endres ikke av denne kontrollen.

23 målrettede kontroller bestod under audit-vakt. De dekker alle seks lokale
fit-klokker, H4/D1-filtrering før tilpasning, datobinding/hash, historisk juni-
overlapp og eksisterende syntetisk ende-til-ende-mekanikk. Ny ekte pakke passerer
første indre grense; gammel pakke avvises ved alle fire årsgrenser og juni 2026.
Ingen tester eller parameterpakke dokumenterer bedre Entry.

## Neste avgrensning

Native signalflate har 241 felt. Berørte eiere leverer 33 nivå-, 31 trendlinje-
og tre squeeze-felt; 174 øvrige signalfelt og 71 ctx-felt beholdes dersom
avhengighet og paritet er bekreftet. MTF har 190 felt per klokke, hvor de samme
eiergruppene leverer 67. Gruppene inkluderer også enkelte uendrede råfelt;
gruppetilhørighet betyr ikke at hver verdi endres.

Et separat forskningsuttrekk skal først reprodusere gamle lokale eierutdata
på alle eksisterende TRAIN-rader. Deretter regenereres de aktuelle blokkene
og MTF fra den nye pakken. Alle gjenbrukte MTF-felt og eksakte ctx-aliaser
skal være bitlike. Ingen native manifest skal gis en usann ny proveniens,
og native TRAIN-vindukontrakten skal ikke svekkes. De 235 eksisterende,
tape-baserte mønsterfeltene har uavhengige, faste beregninger og egen bevart
prefix-kontroll.

Regenereringsskript er klargjort, ikke kjørt ved denne rapporten. Ingen ny
Entry-fit, native rebuild, TEST, live/papir, push eller utgiftsbruk. Målet er
aktivt og ikke oppnådd.

## Evidens

/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/FEATURE_CALIBRATION_EARLY_20260924

WITH_EXISTING_PREHISTORY inneholder fullført plan, parameterpakker, RESULT,
START, TERMINAL og logg. COMPLETED_CALIBRATION_REVIEW.json binder verifisert
kilde og ekte gammel/ny datogrense. LEGACY_HTF_FIT_AVAILABILITY.json binder
de eksakte gamle H4/D1-populasjonene. REGENERATION_SCOPE.json binder omfanget.

Den første fem-klokke inventeringen nådde sin 30 s tidsramme uten resultat.
En redusert H4/D1-inventering trengte korrekt utelukkelse av tomme resample-
bøtter før hashene stemte. Den avsluttede, hash-identiske kontrollen tok
1,85 s og utførte null parameter-/modellfits. Tidligere forsøk er bevart.
