#!/usr/bin/env bash
# Query and verify one nonce-bound host GPU telemetry response.
#
# This is deliberately a narrow transport primitive used only by the canonical
# trainer guard.  It accepts no ambient configuration: the capped runner owns
# all four arguments, binds the certificate hash to source, and fails closed on
# every transport, schema, identity, freshness, or signature error.
set -euo pipefail

readonly REQUEST_SCHEMA='gx1_host_gpu_telemetry_request_v1'
readonly RESPONSE_SCHEMA='gx1_host_gpu_telemetry_v1'

die() {
  printf 'FATAL: host telemetry bridge: %s\n' "$*" >&2
  exit 75
}

[[ $# -eq 5 ]] || die 'requires URL, certificate path, certificate SHA-256, GPU UUID, and timeout seconds'
bridge_url="$1"
certificate_path="$2"
certificate_sha256="$3"
expected_gpu_uuid="$4"
timeout_seconds="$5"

# The bridge has two deliberately narrow transports: Windows loopback, or the
# RFC1918 WSL gateway address selected and firewall-restricted by the elevated
# Windows installer.  The canonical runner source-binds the exact URL; this
# helper never accepts a public or wildcard listener.
[[ "$bridge_url" =~ ^http://(127\.0\.0\.1|172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3}):[1-9][0-9]{0,4}/gx1/v1/telemetry/$ ]] \
  || die 'bridge URL is not an approved loopback or private WSL telemetry endpoint'
[[ "$certificate_path" == /* && -f "$certificate_path" && ! -L "$certificate_path" ]] \
  || die 'certificate path is not an existing absolute regular file'
[[ "$certificate_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || die 'certificate SHA-256 must be lowercase hexadecimal'
[[ "$expected_gpu_uuid" =~ ^GPU-[0-9a-fA-F-]{36}$ ]] \
  || die 'GPU UUID is malformed'
[[ "$timeout_seconds" =~ ^[1-5]$ ]] \
  || die 'timeout seconds must be an integer from 1 through 5'

for helper in /usr/bin/curl /usr/bin/jq /usr/bin/od /usr/bin/tr /usr/bin/mktemp /bin/rm /usr/bin/sha256sum /usr/bin/openssl /usr/bin/base64 /usr/bin/awk; do
  [[ -x "$helper" ]] || die "required helper is unavailable: $helper"
done

actual_certificate_sha256=$(/usr/bin/sha256sum "$certificate_path" | /usr/bin/awk '{print $1}')
[[ "$actual_certificate_sha256" == "$certificate_sha256" ]] \
  || die 'certificate SHA-256 does not match the source-bound value'

scratch_dir=$(/usr/bin/mktemp -d /tmp/gx1-host-telemetry.XXXXXXXX) \
  || die 'could not create private bridge scratch directory'
cleanup() {
  /bin/rm -rf -- "$scratch_dir" 2>/dev/null || true
}
trap cleanup EXIT

nonce=$(/usr/bin/od -An -N32 -tx1 /dev/urandom | /usr/bin/tr -d ' \n') \
  || die 'could not generate request nonce'
[[ "$nonce" =~ ^[0-9a-f]{64}$ ]] || die 'generated request nonce is invalid'
request_json=$(/usr/bin/jq -cn --arg nonce "$nonce" \
  '{schema_version:"gx1_host_gpu_telemetry_request_v1",request_nonce:$nonce}') \
  || die 'could not encode request'

response_json=$(
  /usr/bin/curl \
    --fail --silent --show-error \
    --connect-timeout "$timeout_seconds" \
    --max-time "$timeout_seconds" \
    --request POST \
    --header 'Content-Type: application/json' \
    --data "$request_json" \
    "$bridge_url"
) || die 'bridge request failed or timed out'

# No optional fields are tolerated.  The fields are signed in the exact order
# below; the nonce makes a cached response invalid even if every reading looks
# plausible.
if ! printf '%s' "$response_json" | /usr/bin/jq -e \
  --arg schema "$RESPONSE_SCHEMA" \
  --arg nonce "$nonce" \
  --arg uuid "$expected_gpu_uuid" '
    def finite_nonnegative:
      type == "number" and isfinite and . >= 0;
    def finite_positive:
      finite_nonnegative and . > 0;
    type == "object"
    and (keys | sort) == [
      "core_temp_c",
      "gpu_uuid",
      "memory_temp_c",
      "memory_used_mib",
      "observed_monotonic_ms",
      "power_draw_w",
      "power_limit_w",
      "request_nonce",
      "schema_version",
      "signature"
    ]
    and .schema_version == $schema
    and .request_nonce == $nonce
    and .gpu_uuid == $uuid
    and (.core_temp_c | finite_nonnegative)
    and (.memory_temp_c | finite_nonnegative)
    and (.power_draw_w | finite_positive)
    and (.power_limit_w | finite_positive)
    and (.memory_used_mib | finite_nonnegative and floor == .)
    and (.observed_monotonic_ms | finite_nonnegative and floor == .)
    and (.signature | type == "string" and test("^[A-Za-z0-9+/]+={0,2}$"))
  ' >/dev/null; then
  die 'bridge response failed schema, nonce, UUID, or finite-value validation'
fi

IFS=$'\t' read -r core_temp memory_temp power_draw power_limit memory_used observed_monotonic signature_b64 < <(
  printf '%s' "$response_json" | /usr/bin/jq -r \
    '[.core_temp_c, .memory_temp_c, .power_draw_w, .power_limit_w, .memory_used_mib, .observed_monotonic_ms, .signature] | @tsv'
)

canonical_float() {
  local raw="$1"
  LC_ALL=C /usr/bin/awk -v value="$raw" 'BEGIN { printf "%.6f", value }'
}

core_for_signature=$(canonical_float "$core_temp")
memory_for_signature=$(canonical_float "$memory_temp")
draw_for_signature=$(canonical_float "$power_draw")
limit_for_signature=$(canonical_float "$power_limit")

printf '%s\n' \
  "$RESPONSE_SCHEMA" \
  "$nonce" \
  "$expected_gpu_uuid" \
  "$core_for_signature" \
  "$memory_for_signature" \
  "$draw_for_signature" \
  "$limit_for_signature" \
  "$memory_used" \
  "$observed_monotonic" >"$scratch_dir/payload.txt"

printf '%s' "$signature_b64" | /usr/bin/base64 --decode >"$scratch_dir/signature.bin" 2>/dev/null \
  || die 'signature is not valid base64'
/usr/bin/openssl x509 -in "$certificate_path" -pubkey -noout >"$scratch_dir/public.pem" 2>/dev/null \
  || die 'certificate does not contain a usable public key'
/usr/bin/openssl dgst -sha256 \
  -verify "$scratch_dir/public.pem" \
  -signature "$scratch_dir/signature.bin" \
  "$scratch_dir/payload.txt" >/dev/null 2>&1 \
  || die 'response signature verification failed'

# The guard parses this single machine-readable row and applies the existing
# core/VRAM/power/VRAM-residency stops.  Do not add a UUID or signature here:
# those have already been checked and logs must not expose the signature.
printf '%s,%s,%s,%s,%s\n' \
  "$core_temp" "$memory_temp" "$power_draw" "$power_limit" "$memory_used"
