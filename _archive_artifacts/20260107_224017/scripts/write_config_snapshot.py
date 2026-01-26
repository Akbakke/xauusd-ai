import json
import yaml
from pathlib import Path
import sys

policy_path = sys.argv[1]
output_dir = Path(sys.argv[2])

# Load policy config
with open(policy_path) as f:
    policy = yaml.safe_load(f)

# Extract key config for verification
config_snapshot = {
    "policy_path": str(policy_path),
    "entry_config": policy.get("entry_config", ""),
    "entry_v9_policy_sniper": {
        "enabled": policy.get("entry_v9_policy_sniper", {}).get("enabled", False),
        "threshold": policy.get("entry_v9_policy_sniper", {}).get("threshold", None),
        "shadow_only": policy.get("entry_v9_policy_sniper", {}).get("shadow_only", False),
    },
    "entry_v9_policy_farm_v2b": {
        "enabled": policy.get("entry_v9_policy_farm_v2b", {}).get("enabled", False),
    },
    "gates": policy.get("gates", {}),
    "killswitch": policy.get("killswitch", {}),
    "dry_run": False,
    "replay_mode": True,
    "fast_replay": True,
}

# Write config snapshot
config_path = output_dir / "config_snapshot.json"
output_dir.mkdir(parents=True, exist_ok=True)
with open(config_path, "w") as f:
    json.dump(config_snapshot, f, indent=2, default=str)

print(f"OK: Config snapshot written: {config_path}")

