# PREBUILT KARTLEGGING - SJEKKLISTE

**Dato:** 2025-01-13  
**Mål:** Finn hvor "gated replay" faktisk lever i dagens repo

## SJEKKLISTE

### 1. Finn alle referanser til "replay_eval_gated_parallel"
- [ ] rg -n "replay_eval_gated_parallel" .
- [ ] Dokumenter alle treff med fil + linje

### 2. Finn alle replay-entrypoints som kjøres fra scripts/
- [ ] rg -n "python .*replay" scripts gx1
- [ ] Analyser scripts/go_nogo_prebuilt.sh - identifiser nøyaktig python-kall
- [ ] Analyser scripts/run_fullyear_prebuilt.sh - identifiser nøyaktig python-kall

### 3. Finn alle kandidater til gated replay runner
- [ ] rg -n "(gated|GATED|gated_fusion|fusion|xgb|xgboost).*replay" gx1
- [ ] rg -n "ReplayMode|PREBUILT|BASELINE" gx1
- [ ] rg -n "(worker|spawn|multiprocessing|apply_async|chunk|footer|aggregat|perf_json)" gx1

### 4. Finn hvor worker-spawn skjer
- [ ] rg -n "(apply_async|Pool\\(|Process\\(|multiprocessing)" gx1

### 5. Finn hvor chunk footer / perf JSON / invariants evalueres
- [ ] rg -n "(chunk footer|footer|perf_json|FEATURE_BUILD_TIMEOUT|basic_v1_call_count|feature_time_mean_ms|prebuilt_bypass_count)" gx1

### 6. Sjekk om filen eksisterer under annet navn
- [ ] find gx1 -maxdepth 4 -type f -name "*gated*replay*" -o -name "*replay*parallel*" -o -name "*replay_eval*"

### 7. Hvis filen fortsatt ikke finnes: bruk git-historikk
- [ ] git log --name-only --follow -- gx1/scripts/replay_eval_gated_parallel.py
- [ ] git log --name-only --follow -- "**/*gated*parallel*"
- [ ] git grep -n "replay_eval_gated_parallel" $(git rev-list --all --max-count=100)
