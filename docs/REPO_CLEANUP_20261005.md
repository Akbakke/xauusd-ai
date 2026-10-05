# Repo-opprydding - 05.10.2026

Brukeren ba om et renere repo og sletting av de fleste historiske snapshots.

Omfanget er bare tracked handover_snapshot i Git-repoet. Ingen fil under
/home/andre2/GX1_DATA eller /home/andre2/GX1_RUNS er slettet. Ingen aktiv
native prosess ble observert.

- grunnlag: 90ca4ac4e3a83174d35500c921e82c48c2bbe657
- vurdert: 96 filer
- slettet: 78 filer, 3520440 bytes
- beholdt: 18 filer, 171102 bytes
- beholdningsregel: en fil beholdes hvis kode, scripts, tester, configs,
  agentregler eller gjeldende policy/status fortsatt refererer den
- recovery: alle slettede filer kan hentes fra Git-grunnlaget over

Fullt hashbundet inventar står i REPO_CLEANUP_20261005.json.
Historiske Markdown-rapporter er beholdt i denne bølgen når gjeldende policy
fortsatt peker inn i evidenskjeden. De tre dupliserte top-level statusfilene
er erstattet med korte gjeldende roller.
