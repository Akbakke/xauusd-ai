# STEP 1.7 – Policy disk vs run (hash og konklusjon)

**Dato:** 2026-02-20

---

## DEL A — Policy på disk vs policy i run (hash)

**Canonical policy-fil (LAST_GO / run-artefakter):**  
`/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`

**POLICY_DISK_HASH:**
- **sha256_now:** `410a56bb672f32ee65b87c4bc3259aa5ab632bfaaa3d99a1d510ea0995caec3b`
- **mtime:** `2026-02-20 17:55:21.624388769 +0100` (epoch 1771606521)
- **size_now:** 6182 bytes

**POLICY_RUN_HASH:**  
Fra `run_header.json` (replay/chunk_0/run_header.json):
- **sha256_in_run:** **found** – `86be3a3cf2c714dea2a25b79378244258b8fb27d04646b5d913ee423ee30298b`
- **size_in_run:** 6181 bytes (artifacts.policy.size_bytes)
- **embedded_policy:** **NO** – run_header og REPLAY_SSoT_HEADER inneholder kun `policy_path`, `artifacts.policy.path`, `artifacts.policy.sha256`, `artifacts.policy.size_bytes` og `exit_config` (streng). Ingen inline YAML eller full policy-dump.

**Sammenligning:**  
`sha256_now` ≠ `sha256_in_run`. Filen på disk nå er ikke identisk med filen som ble brukt under run. Størrelse: 6182 nå vs 6181 i run (1 byte forskjell).

---

## DEL B — Alle kopier av policy-filen

**Søk:** `find /home/andre2 -name GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`

**POLICY_DUPLICATES:**
- **path:** `/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`
- **sha256:** `410a56bb672f32ee65b87c4bc3259aa5ab632bfaaa3d99a1d510ea0995caec3b`
- **mtime:** 2026-02-20 17:55:21

Kun **én** forekomst funnet. Ingen duplikater i repo/disk under /home/andre2.

---

## DEL C — Symlinks / overlays / mounts

**Canonical policy-path:**  
`/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`

**POLICY_FILESYSTEM_STATUS:**
- **is_symlink:** **NO** – `ls -l`: regular file `-rw-r--r--`; `stat`: "regular file", Links: 1
- **inode:** 95285
- **mount_type:** Ingen treff for GX1_ENGINE eller /home/andre2 i `mount`-output; filen ligger på vanlig filsystem (ikke overlay/bind mount funnet for denne stien).

---

## DEL D — Git-status ved kjøretid

**I run-dir:**  
- `run_header.json`: `"git_commit": "5d9b86ca15757b3d0f65d817c0a1c9fb8f5a5f4c"`
- `REPLAY_SSoT_HEADER.json`: `"git_commit": "5d9b86ca15757b3d0f65d817c0a1c9fb8f5a5f4c"`
- RUN_IDENTITY.json finnes ikke i run-dir. BUILD_INFO.json ikke funnet.

**Nåværende git i GX1_ENGINE:**  
- `git rev-parse HEAD`: `5d9b86ca15757b3d0f65d817c0a1c9fb8f5a5f4c`

**GIT_STATE:**
- **commit_in_run:** `5d9b86ca15757b3d0f65d817c0a1c9fb8f5a5f4c`
- **commit_now:** `5d9b86ca15757b3d0f65d817c0a1c9fb8f5a5f4c`
- **matches:** **YES**

Run ble kjørt med samme git-commit som nå. Endring i policy-innhold skyldes derfor ikke annen commit; filen er blitt endret i arbeidskopi etter run (eller etter commit som brukte den versjonen).

---

## DEL E — Policy snapshot-mekanismer

**Søk:** policy_snapshot, write_policy, dump_yaml, policy_yaml, skriving av policy til run-dir.

**Funn:**
- Ingen kopi av policy YAML-fil i run-dir (ingen .yaml under E2E_SANITY_20260219_203609).
- `run_header.json` skrives av `generate_run_header` (gx1/prod/run_header.py) med `policy_path`, `artifacts.policy` (path, sha256, size_bytes); ingen serialisering av full policy-innhold.
- `_write_policy_path_capsule` (truth_banlist) skriver kun metadata ved path-mismatch; ingen policy-snapshot.
- Ingen TRUTH/replay-sti som skriver full policy YAML/JSON inn i run-dir.

**POLICY_SNAPSHOT_MECHANISM:**
- **found:** **NO**
- **details:** Run-artefakter refererer til policy via path + hash + size; ingen embedded policy-snapshot. Runner leser alltid fra `policy_path` på disk (load_yaml_config). Artefakter reflekterer altså innholdet av filen **ved lesetid**, ikke en frosset kopi i run-dir.

---

## DEL F — Sluttkonklusjon (kun fakta)

**STEP 1.7 CONCLUSION:**

- **policy_file_used_at_runtime:** `/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml` (samme path som i run_header og REPLAY_SSoT_HEADER).

- **policy_file_on_disk_now_differs:** **YES**  
  Bevis: Run hash `86be3a3cf2c714dea2a25b79378244258b8fb27d04646b5d913ee423ee30298b` (size 6181) vs nå `410a56bb672f32ee65b87c4bc3259aa5ab632bfaaa3d99a1d510ea0995caec3b` (size 6182). Filen er endret etter run.

- **evidence_of_multiple_policy_versions:** **NO**  
  Kun én fysisk fil med det navnet; ingen symlink; samme git-commit. Forskjellen er innholdsendring i samme fil (disk drift).

- **most_likely_explanation:** **disk drift** – Policy-filen på disk ble endret etter at run (E2E_SANITY_20260219_203609) ble kjørt. Run brukte versjon med hash 86be... og exit_config MASTER_EXIT_V1_A; nåværende fil har hash 410a... og exit_config EXIT_TRANSFORMER_ONLY_V0. Ingen tegn på at runner leste annen path, annen git-versjon, overlay, eller at artefakter refererer til snapshot i stedet for live fil.
