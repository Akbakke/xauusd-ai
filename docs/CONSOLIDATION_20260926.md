# Konsolidering til én kodebase — 26.09.2026

Operatørvedtak 26.09: *«Vi skal ALDRI jobbe med hvert vårt, så en må vel arkiveres …
vi tar det beste fra dette og fortsetter.»* Eneste kodebase er `work/gx1-current`
(`/home/andre2/src/GX1_CURRENT`).

## Hva som skjedde (ikke godkjent av operatøren)

- Felles basis `cf246b37` (08.09). Her kom 282 commits 08.–26.09 (lifecycle-v2 native
  Entry+Exit, BID/ASK-kostpolicy, én-posisjons-replay).
- `audit/v9-premiere-20260905` (`/home/andre2/src/GX1_ENGINE`) sto urørt 08.–19.09. Fra
  19.09 kl. 22:27 til 24.09 kl. 20:02 kom 76 commits der, på den foreldede basen:
  dyprevisjon og reparasjonsbølger, V10/V11/V12-rebuild, V12-kandidattrening (stoppet på
  operatørordre ved epoch 2), Entry-eksperimenter 23.09, walk-forward-forskning 23.–24.09
  og Codex' målerettelser 24.09.
- Årsak: rot-loaderen `/home/andre2/CLAUDE.md` importerte konstitusjonen i GX1_ENGINE, mens
  denne kodebasens egne dokumenter erklærte seg som eneste kilde. Økter startet fra
  `/home/andre2` havnet derfor i feil worktree.

## Slik er det slått sammen

- Grenen er tagget `archive/gx1-engine-audit-v9-20260926` og slått inn med merge-strategien
  `ours` pluss eksplisitt import per fil. En vanlig merge ville latt uønskede endringer gli
  inn uten konfliktmarkører (bl.a. ny påkrevd parameter i Exit-stien, lifecycle-envelope v12,
  vaktkonstanter og cloud-rør) og brukket native TRAIN/VAL.
- Grunnlag: fem lesende analyser, lagret under
  `/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/DIRECT_TARGET_IDENTITY_CLAUDE_20260926/consolidation_analysis/`
  (A featurepipeline, A2 datakjede, B trener/modell/Exit, C drift/vakter/styring,
  D forskningsinstrument).

### Tatt inn i konsoliderings-commiten (M1)

- Forskningsinstrumentet: `research_entry_direction_walkforward_v1`,
  `research_entry_pattern_primitives_v1`, `research_entry_pattern_setup_edge_v1`, tre
  testfiler (57 tester består her) og tre ruter i `scripts/entry_next_edge_control.sh`.
- 22 forskningsdokumenter (retningsdiagnosen, 23.–24.09-kontrollene, funnregistrene
  `PIPELINE_FEATURE_FIDELITY_REVIEW_20260921` og `PROJECT_DEEP_REVIEW_20260919`).
- Styring: `GX1_RULES.md` (regel 1–25 gjenopprettet, fjernet her 14.09 i a63353a4),
  `AGENTS.md`, `CLAUDE.md` med import av begge, oppdatert handover, arbeidsmål og indeks.
- Nye målinger: [DIRECTION_TIMESCALE_20260926.md](DIRECTION_TIMESCALE_20260926.md).
- `passord.md` er fjernet (operatørens overføringstest).
- Guard-testen (`tests/test_guard_hooks_versioned.py`) ignorerer nå personlige preferanser
  (`model`, `modelSettings`, `theme`, `agentPushNotifEnabled`); hooks, tillatelser, `env` og
  dangerous-mode-flagget sjekkes fortsatt eksakt. Et modellbytte blokkerte før alle commits.
- Git: merge-commit 809ba049 (kun historikk) + denne commiten; arkiv-commit 98867b17 i
  arkivgrenen; `core.hooksPath` er relativ (`.claude/git-hooks`), så hver worktree bruker sin
  egen versjonerte hook.

### Planlagt i egne commits (M2–M4)

- **M2 featureflate** (signal v34 → v36: 238 → 241 = 24 + 150 + 67; per-TF 176 → 190):
  826504f3, 280e56fd (kun featurekode), f9e40213 (kun featuredeler), 5ec2e537, 627894ac,
  e8a639ca (+ fjern foreldreløse `_v1_atr14_bps`), b1b034bf (+ ny squeeze-manifestversjon),
  233a602d tilpasset (C-1 grenseeier, C-3 TEST-logg med hjelper definert før bruk, C-4
  registry-vindu, C-7 MAD ≥ 2 avvik med versjonsbump av normaliseringen). CURRENTs
  `MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4` må oppdateres (10 pensjonerte navn ut, nye
  binærfelt inn etter verifikasjon), og normaliseringen må få kategori-domenet for
  `mtf_smc_swing_state`. Deretter full rebuild (squeeze → kjede → audits → lifecycle-v2-lag).
- **M3 trenerfeil** på den kjørte stien (fra 6a2be2a7): gate-entropi-sjekken kan aldri slå
  inn (clamp før multiplikasjon), active-head-diagnostikk (én sesjonsklokke, FLAT ikke unntatt,
  eksakt null-test, manglende masker feiler lukket, stille clamps fjernet), gamma-metadata.
  EMA-warmup vurderes separat med identitetsbump.
- **M4 forskningsinstrument for uker**: `--decision-clock`, statistikk på ikke-overlappende
  perioder, kausal konstant- og alltid-LONG-referanse, vern mot sirkulær-null-krasj, `model_kind`
  i metadata. HAC-standardfeilen er undervurdert ~√(h/17) ved overlappende etiketter (~2,4× ved
  h96, ~4× ved h288); gamle strict_pass på lange horisonter er derfor ikke gyldige.

### Bevisst ikke tatt inn (finnes på arkiv-taggen)

- Exit/lifecycle: 5b29e1b4 og C-2-delene av f9e40213 — her mates allerede ærlig veggklokke-
  alder (`elapsed_wall_clock_seconds_log1p`), og endringen brekker lifecycle-v2.
- Trener/modell: 48d570ad, cd602015, 72dde80a, 4711c518, 2870c187, 5d31ad3b, d648859f
  (bare hypotese), og de mislykkede forskningskontrollene 88f0cf46, eddae3c1, 2eee9ffa,
  022a5d8a, 7abae358, 7edb5579, 80fed3e4.
- Drift: cloud-pakken (e4f0902e, 02babae8), vaktkommitene 5be2d7e2 og 1c747b71, grenens
  handover/launch-state og foreldede statusbannere i eldre dokumenter.
- Grenens handover-dokumenter (CURRENT_HANDOFF_20260921 m.fl.).
- Artefakter: V10/V11/V12-datasettene og V12-kandidatsesjonen ligger urørt under GX1_DATA som
  evidens; de passer ikke denne kodebasens flate. `PROJECT_STATE_xau_direction_launch.json` her
  beskytter ikke grenens V12-artefakter i retention-grafen — ingen opprydding før en
  arkivautoritet dekker begge linjene (regel 9).

## Åpne operatørvedtak

1. Guard-referansen (`.claude/settings.reference.json` vs live `model`) blokkerer alle commits.
2. GPU-kjernestopp på native rute: i dag 85 °C (+ nedtrekk til 200 W ved 80 °C); vedtaket
   20.–21.09 om 70 °C gjaldt grenens trenerrute.
3. Lengre XAU_USD-historikk (bjørnemarkeder) for ukesretning — krever navngitt kilde og manifest.

## Ikke undersøkt

- Om V29-registry-fitten har samme delvis-lukket-HTF-bar-problem som b1b034bf rettet for squeeze.
- Om rutere her får underflow (klassen d648859f fant på grenens vekter).
- Exit M1-lanen ende-til-ende på den nye flaten.
- Åpne funn F-12, F-13, F-17 og F-25 (DST) fra funnregisteret.
