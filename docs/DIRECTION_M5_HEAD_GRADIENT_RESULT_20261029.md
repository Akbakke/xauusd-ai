# M5-retningsmodell: fem minutter og øvrige 59 horisonter

Den fullførte prisbaserte M5-modellen ble trent med gjennomsnittlig BCE over alle observerte, modne mål med ikke-null retning. Evalueringen har fem minutter som primærmåling. Dette er en målsetningsforskjell som bør undersøkes; ingen av de 60 målhodene fjernes.

En forhåndsavgrenset CPU-kontroll brukte de 256 allerede lagrede TRAIN-foreldrene, 16 i hver gruppe, det ferdigtrente checkpointets to retningshoder og den lagrede fasiten. Ingen modellforward, optimizersteg, nye markedsdata, Q1-evaluering, TEST eller handel inngikk. For hver gruppe beregnet kontrollen analytisk gradienten av den faktiske uvektede BCE-en ved den felles skjulte representasjonen, og delte den i bidraget fra fem minutter og bidraget fra de øvrige 59 minuttene. En syntetisk kontroll sammenlignet beregningen med PyTorch autograd, også ved manglende, like og umodne utfall.

| Måling | Resultat |
| --- | ---: |
| Modne, observerte ikke-null celler | 15 095 |
| Femminuttersceller | 256 (1,696 %) |
| Norm, femminuttersbidrag | 0,0043881 |
| Norm, summen av øvrige 59 | 0,0502263 |
| Forhold mellom de to normene | 0,08737 |
| Cosinus mellom bidragene | 0,01376 |
| Grupper med negativ / positiv cosinus | 4 / 12 |

Gradientene er satt sammen på tvers av foreldre; normaliseringen gjøres med antall kvalifiserte celler i hver 16-foreldersgruppe. Resultatet viser et svakere femminuttersbidrag i denne representasjonen og nesten vinkelrette samlede bidrag. 59 andre hoder er imidlertid ikke en likestilt enkeltreferanse, og positiv eller negativ cosinus i enkeltgrupper kan ikke alene fortelle om læringen blir bedre eller dårligere.

Målingen bruker evalueringsprediksjoner etter siste oppdatering og er ikke et opptak av gradientene ved treningens faktiske steg. Den beregner gradienter til representasjonen, ikke til modellparametrene; tidligere lag kan endre både retning og størrelse. De 256 foreldrene er et diagnostisk utvalg, ikke hele treningshistorikken. Det finnes derfor ikke bevis her for gradientkonflikt, at endret vekting vil hjelpe, generalisering eller økonomisk edge. En matchet test av femminuttersprioritet må endre bare vektingen, beholde alle 60 mål og sammenligne på de samme senere utviklingsradene. Allerede gjenbrukt Q1 kan bare gi utviklingsevidens.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_HEAD_GRADIENT_20261029_002/RESULT.json`, SHA-256 `f06e5bf47076812c5d5679309fac6bd1ee885434b751f345862cfc531905f251`. Den første engangsplanen ble konsumert etter at JSON-lagringen av en NumPy-talltype feilet; korrigert plan 002 bestod en egen serialiseringstest før gjennomføring. Summene, cosinus, normidentiteten, inputbindingene og terminalens SHA ble etterkontrollert.

RSI14 ble tidligere uavhengig tallkontrollert mot publisert cache på M5/M15/H1/H4/D1, og MACD-linje/-histogram samt ret_1 ble uavhengig gjenskapt for et avgrenset TRAIN-utvalg på alle fem tidsrammer. Samleren krever at hver tidsruterad er lukket senest ved M5-beslutningen. Dette viser ikke at alle verdier, tidsperioder eller fysisk ankomsttid er kvalifisert, eller at indikatorene gir prediktiv verdi. Modellens faktiske prisprojeksjonsrute hadde svært liten målbar innvirkning i den separate, tidligere gjennomførte intervensjonen.
