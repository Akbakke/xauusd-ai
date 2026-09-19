#!/usr/bin/env bash
set -euo pipefail
export PATH='/usr/sbin:/usr/bin:/sbin:/bin'

readonly SERVICE_NAME='gx1-host-telemetry.service'
readonly INSTALL_DIR='/usr/local/libexec/gx1'
readonly INSTALLED_SERVICE="$INSTALL_DIR/gx1_linux_host_telemetry_service.py"
readonly CONFIG_DIR='/etc/gx1-host-telemetry'
readonly PRIVATE_KEY="$CONFIG_DIR/private-key.pem"
readonly CERTIFICATE="$CONFIG_DIR/public-cert.pem"
readonly ENVIRONMENT_FILE="$CONFIG_DIR/service.env"
readonly UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"

die() {
  printf 'FATAL: GX1 Linux host telemetry installer: %s\n' "$*" >&2
  exit 78
}

usage() {
  cat >&2 <<'EOF'
Usage: install_gx1_linux_host_telemetry.sh --gpu-uuid GPU-... [--port 38128] [--nvidia-smi /usr/bin/nvidia-smi]
EOF
  exit 64
}

[[ ${EUID:-$(/usr/bin/id -u)} -eq 0 ]] || die 'must run as root'

gpu_uuid=''
port='38128'
nvidia_smi='/usr/bin/nvidia-smi'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-uuid)
      [[ $# -ge 2 ]] || usage
      gpu_uuid="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || usage
      port="$2"
      shift 2
      ;;
    --nvidia-smi)
      [[ $# -ge 2 ]] || usage
      nvidia_smi="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$gpu_uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] \
  || die 'GPU UUID is malformed'
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) \
  || die 'port must be an integer from 1 through 65535'
[[ "$nvidia_smi" =~ ^/[A-Za-z0-9._/+:-]+$ && -x "$nvidia_smi" ]] \
  || die 'nvidia-smi must be an existing absolute executable with a safe path'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE_SERVICE="$SCRIPT_DIR/gx1_linux_host_telemetry_service.py"
[[ -f "$SOURCE_SERVICE" && ! -L "$SOURCE_SERVICE" ]] || die 'service source copy is unavailable'

for helper in /usr/bin/awk /usr/bin/id /usr/bin/install /usr/bin/mktemp /usr/bin/openssl /usr/bin/python3 /usr/bin/sha256sum /usr/bin/systemctl /bin/chown /bin/chmod /bin/rm; do
  [[ -x "$helper" ]] || die "required helper is unavailable: $helper"
done

probe_output=$(LC_ALL=C "$nvidia_smi" \
  --query-gpu=uuid \
  --format=csv,noheader,nounits \
  --id="$gpu_uuid") || die 'nvidia-smi GPU identity probe failed'
[[ "$probe_output" == "$gpu_uuid" ]] || die 'nvidia-smi did not return exactly the requested GPU UUID'

/usr/bin/install -d -o root -g root -m 0755 "$INSTALL_DIR"
/usr/bin/install -o root -g root -m 0755 "$SOURCE_SERVICE" "$INSTALLED_SERVICE"
/usr/bin/install -d -o root -g root -m 0700 "$CONFIG_DIR"

if [[ -e "$PRIVATE_KEY" || -e "$CERTIFICATE" ]]; then
  [[ -f "$PRIVATE_KEY" && ! -L "$PRIVATE_KEY" && -f "$CERTIFICATE" && ! -L "$CERTIFICATE" ]] \
    || die 'refusing a partial, linked, or non-regular signing identity'
else
  key_tmp=$(/usr/bin/mktemp "$CONFIG_DIR/.private-key.XXXXXXXX")
  cert_tmp=$(/usr/bin/mktemp "$CONFIG_DIR/.public-cert.XXXXXXXX")
  cleanup_identity() {
    /bin/rm -f -- "$key_tmp" "$cert_tmp"
  }
  trap cleanup_identity EXIT
  /usr/bin/openssl req \
    -x509 -newkey rsa:3072 -sha256 -days 3650 -nodes \
    -subj '/CN=GX1-linux-host-telemetry' \
    -addext 'basicConstraints=critical,CA:FALSE' \
    -addext 'keyUsage=critical,digitalSignature' \
    -keyout "$key_tmp" -out "$cert_tmp" >/dev/null 2>&1 \
    || die 'could not create the RSA3072 signing identity'
  /usr/bin/install -o root -g root -m 0600 "$key_tmp" "$PRIVATE_KEY"
  /usr/bin/install -o root -g root -m 0644 "$cert_tmp" "$CERTIFICATE"
  cleanup_identity
  trap - EXIT
fi

/bin/chown root:root "$PRIVATE_KEY" "$CERTIFICATE"
/bin/chmod 0600 "$PRIVATE_KEY"
/bin/chmod 0644 "$CERTIFICATE"
/usr/bin/openssl pkey -in "$PRIVATE_KEY" -check -noout >/dev/null 2>&1 \
  || die 'private key validation failed'
private_key_bits=$(/usr/bin/openssl pkey -in "$PRIVATE_KEY" -text -noout 2>/dev/null | /usr/bin/awk '/Private-Key:/{value=$2; gsub(/[^0-9]/, "", value); print value; exit}')
[[ "$private_key_bits" == '3072' ]] || die 'private key is not RSA3072'
/usr/bin/openssl x509 -in "$CERTIFICATE" -noout -checkend 86400 >/dev/null 2>&1 \
  || die 'certificate validation failed or expires within 24 hours'
private_public_sha=$(/usr/bin/openssl pkey -in "$PRIVATE_KEY" -pubout 2>/dev/null | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')
certificate_public_sha=$(/usr/bin/openssl x509 -in "$CERTIFICATE" -pubkey -noout 2>/dev/null | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')
[[ "$private_public_sha" == "$certificate_public_sha" ]] || die 'certificate does not match the private key'

environment_tmp=$(/usr/bin/mktemp "$CONFIG_DIR/.service-env.XXXXXXXX")
unit_tmp=$(/usr/bin/mktemp /etc/systemd/system/.gx1-host-telemetry.XXXXXXXX)
cleanup_files() {
  /bin/rm -f -- "$environment_tmp" "$unit_tmp"
}
trap cleanup_files EXIT

cat >"$environment_tmp" <<EOF
GX1_HOST_TELEMETRY_GPU_UUID=$gpu_uuid
GX1_HOST_TELEMETRY_PORT=$port
GX1_HOST_TELEMETRY_NVIDIA_SMI=$nvidia_smi
EOF
/usr/bin/install -o root -g root -m 0600 "$environment_tmp" "$ENVIRONMENT_FILE"

cat >"$unit_tmp" <<'EOF'
[Unit]
Description=GX1 signed Linux GPU host telemetry
After=local-fs.target

[Service]
Type=simple
User=root
Group=root
EnvironmentFile=/etc/gx1-host-telemetry/service.env
ExecStart=/usr/bin/python3 /usr/local/libexec/gx1/gx1_linux_host_telemetry_service.py --listen-host 127.0.0.1 --port ${GX1_HOST_TELEMETRY_PORT} --gpu-uuid ${GX1_HOST_TELEMETRY_GPU_UUID} --private-key /etc/gx1-host-telemetry/private-key.pem --nvidia-smi ${GX1_HOST_TELEMETRY_NVIDIA_SMI} --openssl /usr/bin/openssl --sensor-timeout-seconds 2 --request-timeout-seconds 2
Restart=on-failure
RestartSec=2s
TimeoutStartSec=15s
TimeoutStopSec=10s
UMask=0077
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectClock=yes
ProtectControlGroups=yes
ProtectKernelLogs=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
ProtectHostname=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
RestrictNamespaces=yes
RestrictAddressFamilies=AF_INET AF_UNIX
IPAddressDeny=any
IPAddressAllow=localhost
CapabilityBoundingSet=
AmbientCapabilities=
SystemCallArchitectures=native
TasksMax=32
MemoryMax=256M

[Install]
WantedBy=multi-user.target
EOF
/usr/bin/install -o root -g root -m 0644 "$unit_tmp" "$UNIT_PATH"
cleanup_files
trap - EXIT

/usr/bin/systemctl daemon-reload
/usr/bin/systemctl enable --now "$SERVICE_NAME" >/dev/null
/usr/bin/systemctl is-active --quiet "$SERVICE_NAME" \
  || die 'systemd service did not become active'

certificate_sha256=$(/usr/bin/sha256sum "$CERTIFICATE" | /usr/bin/awk '{print $1}')
printf 'service=%s\n' "$SERVICE_NAME"
printf 'telemetry_url=http://127.0.0.1:%s/gx1/v1/telemetry/\n' "$port"
printf 'gpu_uuid=%s\n' "$gpu_uuid"
printf 'certificate_path=%s\n' "$CERTIFICATE"
printf 'certificate_sha256=%s\n' "$certificate_sha256"
