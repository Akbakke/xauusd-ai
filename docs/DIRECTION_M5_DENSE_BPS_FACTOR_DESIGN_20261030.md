# M5: egne sammenligninger for bps og mer trening

Den nye forskningsvarianten beholder den eksisterende M5-entryprofilen, åtte spesialistfamilier, D1/makro, lukkede høyere tidsrammer, RSI14, MACD-linje/-histogram, ret_1 og alle 60 korrigerte retningsmål. M1 og tick er ikke entryinput. En egen nullinitialisert utgang lærer signert midtprisendring i bps på alle 60 horisonter; den deler eksisterende representasjon. Ett enkelt tap er summen av vanlig 60-hoders retnings-BCE og 60-hoders bps-MSE, der hvert bps-mål skaleres med egen TRAIN-RMS. Dette viderefører den eksisterende bps-metodens relative tapsvekt 1, uten tapsvektsøk. Observerte nullbevegelser er med i bps-tapet, men ikke i retnings-BCE. Fremtidige mål er aldri inputs. Beregningshorisontene er ikke holdetid eller Exit-regler.

## To faktorer, samme grunnlag

Før faktisk kjøring bindes tre separate forløp til samme kildeversjon, egne friske felles initialverdier, frø, foreldrerader, rekkefølge, features og korrigert fasit. Den ene armen trener ren retning ett pass, den andre samme retning med bps-tap ett pass. En tredje fortsetter den første modellens eksakte checkpoint gjennom ett ekstra pass med uendret rent retningstap. Dermed sammenlignes bps mot ren retning ved lik eksponering, og to pass mot ett pass uten endret mål. De ekstra bps-parametrene nullinitialiseres uten å flytte den delte RNG-tilstanden; faktisk identitet i modellens felles tilstand og de 256 første prediksjonene må måles før oppdateringer.

En eventuell sammenligning bruker de samme forhåndsbundne senere Q1 2025-utviklingsradene, primært fem minutter, alle 60 horisonter og parret dag-/ukeblokk-usikkerhet. Q1 er gjenbrukt utvikling og kan ikke være uberørt bekreftelse. En lønnsom handelsedge følger ikke av disse skårene. Se også `DIRECTION_M5_FIVE_PRIORITY_RESULT_20261030.md` for de eksisterende referansene.

## Faktisk status

Kilden for 60 bps-hoder og den nødvendige TRAIN-RMS er kvalifisert. En fysisk privat M5-kontroll brukte 32 TRAIN-foreldre og fire oppdateringer: 256 startprediksjoner og felles modell-/buffer-/RNG-verdier var identiske, alle 60 bps-hoder fikk gradient, bps-signalet nådde den felles projeksjonen og resume var byteidentisk. Kvitteringen er verifisert, og private vekter gjenbrukes ikke. Dette viser at læringsveien virker teknisk, ikke at den forbedrer retning.

To separate ettpassoppskrifter er kildeforberedt og passerte en metadataforkontroll med identiske 64 kildeeiere, input, mål og treningsrekkefølge. En separat to-pass videreføring krever den første signmodellens eksakte terminal og samme kildeeiere. Kilde-/testsjekk og kontroll på ren commit gjenstår før en kampanje åpnes. Ingen av de nye fullfitene eller senere sammenligningene er utført. Se `DIRECTION_M5_SIGN_BPS_EXPOSURE_PROTOCOL_20261030.json`; TEST, broker, handel, Exit og promotion er stengt.

Den nye private kampanjen skal håndheve brukerens 250 W-konfigurasjonsgrense i den signerte treningsvakten. Ved temperaturinngrep kan den eksisterende vakten velge en lavere grense; ingen automatisk retur til 160 W er bestilt.
