# GX1 data contract

Dette dokumentet beskriver beholdte datakontrakter, ikke kjøreautorisasjon.
Gjeldende input-/recipebindinger står bare i NEXT_RUN_POLICY.json.

## Native kilder og tilgjengelighet

Entry bruker native M5, Exit native M1, XAU_USD, komplette MBA-candles i UTC.
Priser, request-/generasjonsbevis, timeframe, intervall, ordnede kolonner,
innholdshash og parent/overlap bindes i immutable manifester.
Bid/ask er obligatorisk; mid-only-substitusjon og syntetisk gapfyll er forbudt.
Ugyldig prisgeometri, kryssede quotes, reversert/duplisert klokke, ikke-endelige
verdier og uforklart fravær feiler lukket. Fravær er ikke en oppfunnet bar.

Høyere tidsrammer må være lukket før featureberegning og beslutning. Beregn
formlene på candles på riktig klokke; resample aldri ferdige indikatorverdier.
Tidsstempel, datatilgjengelighet og beslutningsklokke er forskjellige roller.

## Features og normalisering

Feltorden, dimensjoner, skjemaer og de åtte familiene eies av
gx1/contracts/entry_model_native_signal_v1.py og feature-/arkitektureierne.
Ingen dokumentliteral, top-k-ranker eller ambient flagg kan endre lærte inputs.
M1 og M5 deler formel-/feltkontrakter, ikke kopierte verdier.

V4 MTF-cache binder uforanderlige float32 per-bar- og skalarmatriser med
feltorden, kilde, filstørrelser og hasher. Slicing/rebinding bevarer bytes.
Deklarerte eksakte aliaser følger eieren; andre eksakte duplikater feiler.
Featurewarmup bruker faktisk lukkede barer på hver klokke, inklusive D1,
ikke kalenderdager eller backward embargo. Ukjent kausalt prefiks er NaN;
etter første komplette rad er nye ikke-endelige verdier korrupsjon.

Registry-/squeezeparametre tilpasses bare på deklarert kronologisk TRAIN,
med lane-/clock-/split-/pair-/fitprovenans, eksakte grenser og minimumsstøtte.
VAL/TEST/serve refitter aldri og godtar ikke bare/default/kryss-klokkeparametre.
Hent de faktiske bindingene fra konsumerende manifest, ikke en gammel mappesti.

Input-normalisering tilpasses én gang på hele den fysiske TRAIN-populasjonen
før sampling. Gjeldende median/IQR/asinh- og kategoridomener eies av kontrakten.
Et observert {0,1}-utvalg definerer ikke en kategorisk/binary-type.
Sizings separate TRAIN-ECDF er ikke input-normalisering eller retningsautoritet.
Sweep-AVWAP bruker kausal prisoppdateringsaktivitet, ikke ekte utført volum.

## Split, lineage og mål

TRAIN/VAL/TEST er kronologiske, disjunkte og hash-bundne. Resept-/modellvalg
kan ikke lese TEST. Utviklings-VAL omtales ikke som urørt.
Datasettets build-run-id og modellens output-run-id er separate roller.
Splitmanifestet binder kilder/pair, kode, felter, rader, klokke, mål, lifecycle
og hashes. Manglende eller split-brain lineage feiler før konstruksjon.

Fremtidsutfall er supervision, aldri modellinputs. Signert MFE/path-quality
kan ikke klippes eller absoluttverdi-omskrives; MAE er ikke-negativ.
Kausale diagnostiske utfall og native fitted-Q er ulike autoriteter.
entry_causal_m1_target_policy_v1 eier diagnostikkens eksakte veggklokke/
Entry-quote og komplette path; et gap kan ikke flyttes til en senere quote.
Splitpurging følger hvert utfallsdomene, ikke en gjettet felles horisont.

## Entry/Exit-tilstand og økonomi

Entry/Exit bruker én bundle og delte encoders. Fitted-Q bruker kausale
tilstander og bundne target-networks, ikke pathwise hindsight-optimal exit.
HOLD bootstrapper fra neste gyldige tilstand; EXIT_NOW bruker nåværende
gjennomførbare quote. Økonomiske kostnader, finansiering, kapitalhurdle,
terminale hendelser og cutoff/masks kommer fra gjeldende bundne eiere.

M1-tilstandsindekser er kompakte pekere til immutable markedsrader.
Episode-/viewvalidatoren rekonstruerer indekser, clocks og splitcontainment.
Lukkede markeder er kvalifiserte observerte gaps, ikke syntetiske bars.
En sekvens-/minne-/beregningsgrense er aldri maksimal handelsvarighet.
Første M1-tilstand må være eksakt bundet til Entry; additive in-trade-felt
erstatter ikke den komplette markedsflaten.

Full-input-envelopes binder signal-/kontekst-/MTF-tensorer, normalisering,
entry-token, path/side/quotes/trade-id og bundle/datasett/pairidentitet.
entry_decision_token_v1 eier blocknavn og dimensjoner.
Lik vekthash er ikke funksjons- eller train/serve-paritet.

## Publisering, bevis og retention

Immutable outputs publiseres atomisk no-replace etter bytes, SHA, strict-load
og fsync. Ufullstendige outputs etter stopp/krasj er ikke completionbevis.
Input-, syntaks- og kontrakt-PASS er ikke læring, generalisering eller profitt.

Tunge jobber går gjennom capped-runneren og etablerte vakter. Profilverdier
leses fra den faktiske eieren og kildebundne recipe, ikke dette dokumentet.
Ingen TEST, broker eller handel åpnes av en data-/kildekontroll.

DATA/RUNS slettes bare gjennom evidence_retention_v1/cleanup_gx1_evidence_v1:
plan → godkjenning → utføring, aktiv-prosess-/rekkeviddebevis og hash-bundet logg.
Metadata-, lineage-, event- og witnessreferanser beskytter transitive avhengigheter.
Ukjent/manglende metadata, hashdrift, opaque mapper eller forseglet TEST blokkerer.
En fjernet ancestors eksakte manifestbinding kan bare erstattes av eierverifisert
DELETE_COMPLETE-attestasjon; eksisterende tamper og øvrige child-/overlapbevis
feiler fortsatt lukket. Ingen håndbygd unntaksrute. .env/.venv/.git bevares.
