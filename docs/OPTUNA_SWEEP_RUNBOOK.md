# OPTUNA SWEEP RUNBOOK (SSoT)

## 1) Lås python som har optuna (SSoT)

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
./scripts/py_ssot.sh
```

Dette skriver:
- `reports/replay_eval/OPTUNA_SWEEP/PYTHON_EXECUTABLE.txt`

## 2) Start 300 trials i bakgrunnen (resumable)

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
./scripts/run_optuna_sweep.sh --study sniper_prebuilt_screening_v1 --trials 300 --background
```

Dette skriver:
- logg til `reports/replay_eval/OPTUNA_SWEEP/<study>/screening/run_*.log`
- PID til `reports/replay_eval/OPTUNA_SWEEP/<study>/screening/runner.pid`

## 3) Følg loggen (monitor)

```bash
tail -f reports/replay_eval/OPTUNA_SWEEP/sniper_prebuilt_screening_v1/screening/run_*.log
```

