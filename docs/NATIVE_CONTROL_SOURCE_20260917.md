# Native kontroll fra fysisk TRAIN — 2026-09-17

Eksisterende evaluator krevde fysisk juni-VAL og5508 rader. Den kunne derfor
ikke representere den frosne mars–mai-kontrollen. Minste rettelse bruker samme
factory, rollout, evaluator og resultatvalidering med eksplisitt fysisk TRAIN
og logisk gjenbrukt utviklingskontroll. Ingen separat runner er lagt til.

CONTROL256 hentes bare fra den hashbundne DESIGN-filen. Opprinnelige parent-
og child-ID-er og tidsvindu kontrolleres. Ved ny indeksbinding må alle fysiske
kolonner være identiske med designens indeks, inkludert priser og successor-
counts; bare normaliserings-/kildeavhengige identitetshasher kan bindes om.
En TRAIN-factory uten avgrenset kontroll kan ikke starte full rollout. Senere
normalisering og ferske vekter er fortsatt separate, nødvendige bindinger.

35 unike syntetiske tester består:23 nye og12 eksisterende avgrensede VAL-
kontroller. Første forsøk hadde én feil metodenavn i testen; den ble rettet,
original logg beholdt. Bare feilen og to nye integrasjonskontroller ble kjørt
igjen. En gjenstående juni-kontroll i resultatleseren ble deretter rettet;
kun berørt resultat-/resume-kjede ble gjentatt. Syntetisk pause/gjenopptak og
direkte kjøring gir samme handler/økonomi; det er ikke markedsevidens.

Avvisning av reseleksjon, kilde-/dato-/parent-drift, TEST og falsk full-VAL-
merking er kontrollert. Default juni5508 beholdes. Native tilstander fungerer
også for originale ID-er over5508. Beregninger for pris, lifetime-summary,
MTF og modell er uendret; endrede definisjoner er eksplisitt AST-kontrollert.

Ingen faktiske modell-forwards, optimizersteg, nye fasiter eller nye fits.
Læringsporten er fortsatt uavklart. Neste er fersk native initialisering og
binding av allerede ferdige prefix-normaliseringer, fasiter, rekkefølge og
cutoff. Kildeintegrasjonen er ufullstendig og ingen launch er åpnet.

Maskinbevis: handover_snapshot/NATIVE_CONTROL_SOURCE_SYNTHETIC_20260917.json.
