# CURRENT_HANDOVER — GX1, 07.11.2026

Eneste arbeidssted er `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`.
Les [GX1_RULES.md](GX1_RULES.md) og kjør `bash scripts/gx1_handover.sh --check` ved overtakelse. [NEXT_RUN_POLICY.json](NEXT_RUN_POLICY.json) er eneste maskinlesbare arbeidsstatus; faktisk prosess og uforanderlige terminalkvitteringer overstyrer prosa. Ferdig historikk finnes i Git og de opprinnelige kvitteringene.

## Gjeldende mål og status

Før Entry skal modellen lære løpende OPP/NED/FLATT på M5/M15/H1/H4/D1, med ny prognose på lukkede M5-barer. Entry-plassering, Exit og handel kommer senere. [Gjenværende læringsmetode](docs/NATIVE_LEARNING.md) og [målkandidatene](docs/MARKET_STATE_ENDPOINT_CANDIDATE_20261106.md) beskriver kravene. **Markedsmålet er fremdeles foreslått, ikke valgt/låst.** `training_enabled=false`; ingen ny fit, TEST-tilpasning, broker, ordre eller handel er autorisert.

Aktuell policystatus: `MARKET_STATE_GRAY_PHYSICAL_COMPLETE_TARGET_OPEN`. [Gråsoneplanen](configs/research/MARKET_STATE_GRAY_PHYSICAL_20261107_001.json) og alle tidligere engangsplaner er fullført og konsumert. Ingen ny kjøring eller fit er innrømmet; målvalget er fortsatt åpent.

## Målt og bevist konsistent

- Full TRAIN-kilde: 591 682 M1-rader; 413 av 508 hull er fullt dokumenterte stengninger, 95 er ukjente. [Kildekvittering](/home/andre2/GX1_RUNS/FULL_TRAIN_XAU_CLOSURE_AUTHORITY_20261107_001/RESULT.json) SHA-256 `b58875df44811da3dafde11edbb89347ec49c74d830249f2ed0903a53b502d1c`.
- To terskler, 0,5/1,0 ATR: 118 706 M5-prognosetidspunkter per TF; eksakt M5-tilstand på 47 852, delvis AKTIV på 7 659. Av 63 195 ukjente M5-celler er 62 687 den bevisste gråsonen. [Radvise masker](/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001/ROW_MASKS.npz) SHA-256 `ad0e7d56c58b463bdd3d61fe6b2fcf2b206afa56e899dfb36a850e63c5ec3020`.
- Én terskel, 1,0 ATR: eksakt M5 på 110 539/118 706; 62 687 gamle gråsoneceller blir FLATT uten at kjente etiketter endres. M5-medianen for 1,0 ATR er 4,164 siterte spreads; 0,5 ATR er 2,082. De 62 687 gråsonecellene som blir FLATT har selv median lokal range 3,144 spreads; 82,71 % er minst 2 spreads. I lagret 0,5 ATR-overlapp ble 22 657/34 117 M5-gråsoneceller eiermerket NED/OPP, og 94,6–99,9 % av koblede M15/H1/H4/D1-gråsoner ble tilsvarende merket; dette er terskelavhengig, ikke valgt sannhet. Dekningen skyldes **endret FLATT-definisjon**, ikke bedre modell. [Radvise masker/forhold](/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001/ROW_MASKS_AND_RATIOS.npz) SHA-256 `35257429efa4ec553293e1b10bec0993c3c494f04e7f5be72d55877112e590e5`.
- Uavhengig D1-trendlinjeorakel mot mai–desember 2024 TRAIN: 174 rader × 31 felt = 5 394 sammenligninger, 0 avvik. Alle 19 livssyklusfelt hadde hendelsesstøtte; sjeldneste forekom 4/6 ganger. [RESULT](/home/andre2/GX1_RUNS/D1_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/RESULT.json) SHA-256 `aac68ee7dd349f1e56d3d85da280292706c7e6f414556d9c5842a29e24405b8e`. Dette kvalifiserer bare dette D1-vinduet.
- M5-lokalside ved 1,0 ATR: kald replay over fem observerte M1-barer ga samme side på alle 39 670 eksakte NED/OPP-endepunkter; 0 ukjent/motsatt. Fullhistorikkens siste sideovergang var eldre enn vinduet i 13 298 rader og 11 882 hadde FLATT etter overgangen, men den gamle sidehistorikken trengtes ikke for disse lokale sidene. [RESULT](/home/andre2/GX1_RUNS/M5_STATE_SIDE_LOCALITY_20261107_001/RESULT.json) SHA-256 `bbda7175c50ad0b0517c04f0cc07aa77e20692762cdc0fa6c78671dcf2539df3`. Gjelder ikke FLATT/AKTIV/UKJENT eller andre TF.
- Kildekobling i modellkoden fører fem TF, GC, åtte spesialister og 61 D1/makrofelt inn i representasjonen. Dagens forskningshode spør fortsatt 7 200 posisjoner per M5-rad, bruker sekvenstap og gammel singleton-fasit; scoreren er gammel 0,5 ATR/desember 2024. [Samsvarsnotat](docs/MARKET_STATE_FIVE_ENDPOINT_MODEL_ALIGNMENT_20261107.md). Dette beviser kobling, ikke korrekt formel eller prognoseverdi.
- Indikatorbevis: siste fullførte oktoberfit brukte fullprofil med alle 253 cachefelt per TF og 1 866 optimizersteg i første vindu; kompaktprofilen bruker 139 og parkerer 114. Tidligere numerisk indeks kvalifiserer 158 (=139 aktive+19 parkerte), mens 95 parkerte mangler slik full kontroll. På M5/M15/H1/H4 mangler 77 av dem selv en uavhengig formelstikkprøve, og 12 trendlinjefelt har bare betinget kontroll gitt produsentens linjevalg; D1-orakelet reduserer sin tilsvarende uprøvde mengde til 58 i sitt avgrensede vindu. Tilgjengelig input er ikke bevist læringsnytte. [Feature-revisjon](docs/FEATURE_AUDIT.md).
- Fysisk gråsonekontroll: én capped TRAIN-audit (285,59 s, exit 0) gjenbygde lokal range/ATR med eksakt bitparitet på 537 909 komplette fem-TF-celler. De 273 200 cellene med 0,5–1,0 ATR hadde mellomliggende netto og retningsandel: M5-median `|netto|` 0,296 ATR, `|netto|/range` 0,404 og 1,179 siterte spreads; D1 0,265/0,353/22,745. Alle 300 måned×TF×gruppe-antall besto uavhengig kvitteringskontroll. Dette velger ikke mål eller viser økonomi. [RESULT](/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/RESULT.json) SHA-256 `0b8392ed9713113c64a251edd8e1daa70521c0738a2995e85b17ef9de639581e`; [TERMINAL](/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/TERMINAL.json) SHA-256 `8f83777d03c90794dbe5c534e1647d41ccb9137e3dcd0c9af1f2168105e871fb`.
- Lesende TRAIN-referanse på eksakt kildekoblet nå-/framtidstilstand: ved 1,0 ATR slår alltid FLATT enkel tilstandspersistens på alle fem TF (M5 65,63 mot 53,82 % på 104 009 sammenlignbare rader). Den høye dekningen gir altså en dominerende FLATT-klasse, ikke bevist prognoseverdi. [Sammenligning](docs/MARKET_STATE_ENDPOINT_CANDIDATE_20261106.md).
- De fem TRAIN-foldene er kilde-/klokkemessig mulige med eksakt M1-endepunkt og streng `issue+lead < grense`. Uten modningsvakten ville 265–276 D1-mål per skårperiode krysset høyre grense. [Foldtabell](docs/MARKET_STATE_ENDPOINT_CANDIDATE_20261106.md). Dette er ikke etikettbevis eller fit-admisjon.

## Hindring og neste arbeid

Målsemantikken er den første porten. En enkelt 1,0 ATR-grense gir høy dekning ved å kalle en fysisk mellomliggende gråsone FLATT. M5-sideankeret er kontrollert på eksakte retninger, men FLATT/AKTIV/UKJENT og øvrige TF er fortsatt åpne. Ingen side eller ukjent kilde fylles med gjettede etiketter.

1. Velg betydningen av 0,5–1,0 ATR-mellomsonen på grunnlag av den fysiske kontrollen: foreslått AKTIV med ukjent side, fortsatt helt UKJENT eller FLATT ved 1,0 ATR. Målet låses først etter operatørvalg; ingen terskel velges fra klassebalanse eller modellscore. Bevar fem TF, kausalitet og egne masker for aktivitet og side.
2. Bind fem endepunktsutganger, tillatt-mengde-tap, aktivitet/side/treklasse-skår og kronologiske TRAIN-referanser til **samme** mål. Først deretter kvalifiseres en eksakt, begrenset modellfit. `training_enabled=false` inntil egen admisjon.
3. Fortsett uavhengig kontroll av resterende indikatorformler på M5/M15/H1/H4 og de øvrige familiene. D1-vinduets PASS er ingen generell dom om alle felt eller prognoseverdi.

## Bevarte navngitte artefakter

Disse 15 stiene finnes ellers ikke i gjeldende dokumenter og beholdes bare for retention-eierens navngitte sti-vern. De er ingen kjøreautoritet. Historikken rundt dem ligger i Git.

/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_PREP_20261105_003/TEST_TERMINAL.json
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_FIT_002/RUNTIME/prefix-fit/window-0001/RESULT.json
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_PRIVATE_001/CAMPAIGN/CAMPAIGN_PLAN.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_INPUT_LINEAGE_PARITY_REVIEW.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_LOSS_COMPOSITION_REVIEW.json
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_BATCH_GEOMETRY_ROOT_REVIEW_20261028_002.json
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_NATIVE_B16_SUPPORTED_QUALIFICATION_20261028_003/ACTUAL_ROOT_REVIEW_001.json
/home/andre2/GX1_RUNS/NATIVE_M5_CURRENT_COST_TERMS_20261019_059/BROKER_EVIDENCE.json
/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_003/CAPTURE/INPUT_TIMING.json
