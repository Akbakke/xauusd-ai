#!/usr/bin/env bash
set -euo pipefail
export PATH='/usr/sbin:/usr/bin:/sbin:/bin'

readonly CONFIG_DIR='/etc/gx1-cloud-deadline'
readonly DELETE_COMMAND="$CONFIG_DIR/provider-delete"
readonly FIRE_SCRIPT='/usr/local/libexec/gx1/gx1_cloud_deadline_fire.sh'
readonly SERVICE_UNIT='/etc/systemd/system/gx1-cloud-deadline.service'
readonly TIMER_UNIT='/etc/systemd/system/gx1-cloud-deadline.timer'
readonly PROOF_PATH="$CONFIG_DIR/provider-proof.json"

die() {
  printf 'FATAL: GX1 cloud deadline installer: %s\n' "$*" >&2
  exit 78
}

usage() {
  cat >&2 <<'EOF'
Usage: install_gx1_cloud_deadline_guard.sh \
  --provider NAME --instance-id ID --deadline-utc YYYY-MM-DDTHH:MM:SSZ \
  --provider-delete-command /absolute/root-controlled/executable \
  --provider-delete-command-sha256 HEX
EOF
  exit 64
}

[[ ${EUID:-$(/usr/bin/id -u)} -eq 0 ]] || die 'must run as root'
provider=''
instance_id=''
deadline_utc=''
source_command=''
source_command_sha256=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) [[ $# -ge 2 ]] || usage; provider="$2"; shift 2 ;;
    --instance-id) [[ $# -ge 2 ]] || usage; instance_id="$2"; shift 2 ;;
    --deadline-utc) [[ $# -ge 2 ]] || usage; deadline_utc="$2"; shift 2 ;;
    --provider-delete-command) [[ $# -ge 2 ]] || usage; source_command="$2"; shift 2 ;;
    --provider-delete-command-sha256) [[ $# -ge 2 ]] || usage; source_command_sha256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$provider" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || die 'provider identity is invalid'
[[ "$instance_id" =~ ^[A-Za-z0-9._:-]{1,128}$ ]] || die 'instance identity is invalid'
[[ "$deadline_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] \
  || die 'deadline must be exact UTC seconds'
[[ "$source_command" == /* && -f "$source_command" && ! -L "$source_command" && -x "$source_command" ]] \
  || die 'provider delete command must be an absolute regular executable'
[[ "$source_command_sha256" =~ ^[0-9a-f]{64}$ ]] || die 'provider delete command SHA-256 is invalid'

for helper in /usr/bin/awk /usr/bin/id /usr/bin/install /usr/bin/python3 /usr/bin/sha256sum /usr/bin/systemctl /usr/bin/timeout /bin/chmod /bin/chown; do
  [[ -x "$helper" ]] || die "required helper is unavailable: $helper"
done
observed_sha256=$(/usr/bin/sha256sum "$source_command" | /usr/bin/awk '{print $1}')
[[ "$observed_sha256" == "$source_command_sha256" ]] || die 'provider delete command SHA-256 mismatch'

deadline_epoch=$(/usr/bin/python3 - "$deadline_utc" <<'PY'
from datetime import datetime, timezone
import sys

deadline = datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
delta = int((deadline - now).total_seconds())
if delta <= 0 or delta > 48 * 60 * 60:
    raise SystemExit(1)
print(int(deadline.timestamp()))
PY
) || die 'deadline must be in the future and no more than 48 hours away'

/usr/bin/install -d -o root -g root -m 0755 /usr/local/libexec/gx1
/usr/bin/install -d -o root -g root -m 0700 "$CONFIG_DIR"
/usr/bin/install -o root -g root -m 0700 "$source_command" "$DELETE_COMMAND"

cat >"$FIRE_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
export PATH='/usr/sbin:/usr/bin:/sbin:/bin'
# Power off ONLY after the provider deletion succeeds. A forced poweroff
# after a FAILED delete does not stop provider billing and destroys the only
# agent able to retry; the correct failure posture is to stay up, record the
# failure durably, and keep retrying until the delete verifiably succeeds.
# Each attempt is bounded by the same 120s kill timeout; the timeout itself
# paces the retry loop, so no separate sleep constant is introduced.
attempt=0
while true; do
  attempt=$((attempt + 1))
  status=0
  /usr/bin/timeout --signal=KILL 120s /etc/gx1-cloud-deadline/provider-delete || status=$?
  if [ "$status" -eq 0 ]; then
    /usr/bin/logger -t gx1-cloud-deadline "provider delete succeeded on attempt ${attempt}; powering off"
    /usr/bin/systemctl poweroff --force --no-wall || true
    exit 0
  fi
  /usr/bin/logger -t gx1-cloud-deadline "provider delete FAILED (attempt ${attempt}, status ${status}); host stays up to retry"
  printf '%s attempt=%d status=%d\n' "$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$status" \
    >>/etc/gx1-cloud-deadline/delete-failures.log || true
done
EOF
/bin/chown root:root "$FIRE_SCRIPT"
/bin/chmod 0700 "$FIRE_SCRIPT"

cat >"$SERVICE_UNIT" <<'EOF'
[Unit]
Description=GX1 hard cloud deprovision deadline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
Group=root
ExecStart=/usr/local/libexec/gx1/gx1_cloud_deadline_fire.sh
# The fire script retries the provider delete until it verifiably succeeds
# and only then powers off; it must never be killed for taking long.
# (Type=oneshot forbids Restart=, so the retry loop lives in the script.)
TimeoutStartSec=infinity
UMask=0077
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
RestrictSUIDSGID=yes
LockPersonality=yes
TasksMax=32
MemoryMax=256M
EOF

cat >"$TIMER_UNIT" <<EOF
[Unit]
Description=GX1 48-hour hard cloud deadline timer

[Timer]
OnCalendar=$deadline_utc
AccuracySec=1s
Persistent=yes
Unit=gx1-cloud-deadline.service

[Install]
WantedBy=timers.target
EOF
/bin/chown root:root "$SERVICE_UNIT" "$TIMER_UNIT"
/bin/chmod 0644 "$SERVICE_UNIT" "$TIMER_UNIT"

/usr/bin/python3 - "$PROOF_PATH" "$provider" "$instance_id" "$deadline_utc" "$DELETE_COMMAND" "$source_command_sha256" "$deadline_epoch" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "gx1_cloud_provider_termination_proof_v1",
    "provider": sys.argv[2],
    "instance_id": sys.argv[3],
    "deadline_utc": sys.argv[4],
    "deadline_epoch": int(sys.argv[7]),
    "delete_command_path": sys.argv[5],
    "delete_command_sha256": sys.argv[6],
    "provider_managed_delete": True,
    "hard_cost_cap_nok": 2500.0,
    "local_timer_unit": "gx1-cloud-deadline.timer",
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
/bin/chown root:root "$PROOF_PATH"
/bin/chmod 0644 "$PROOF_PATH"

/usr/bin/systemctl daemon-reload
/usr/bin/systemctl enable --now gx1-cloud-deadline.timer >/dev/null
/usr/bin/systemctl is-active --quiet gx1-cloud-deadline.timer \
  || die 'deadline timer did not become active'
proof_sha256=$(/usr/bin/sha256sum "$PROOF_PATH" | /usr/bin/awk '{print $1}')
printf 'provider_proof_path=%s\n' "$PROOF_PATH"
printf 'provider_proof_sha256=%s\n' "$proof_sha256"
printf 'provider_deadline_utc=%s\n' "$deadline_utc"
printf 'local_timer_unit=gx1-cloud-deadline.timer\n'
