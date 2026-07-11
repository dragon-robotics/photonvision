#!/usr/bin/env bash
set -euo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/orin-nvme-team-image.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

line_number() {
    local pattern="$1"
    grep -n -m 1 -E "${pattern}" "${script}" | cut -d: -f1 || true
}

bash -n "${script}"

grep -q '^MemoryHigh=5G$' "${script}" || fail "missing MemoryHigh=5G"
grep -q '^MemoryMax=6G$' "${script}" || fail "missing MemoryMax=6G"
grep -q '^Restart=always$' "${script}" || fail "missing Restart=always"
grep -q '^RestartSec=2$' "${script}" || fail "missing RestartSec=2"
grep -q '10\.23\.75\.15/8' "${script}" || fail "missing Team 2375 static address"
grep -q '10\.23\.75\.4' "${script}" || fail "missing Team 2375 gateway"
grep -q 'photonvision-orin-nano-ml-tag' "${script}" || fail "missing Orin hostname"
grep -q 'UVC_BANDWIDTH_FIX="no"' "${script}" || fail "UVC bandwidth fix default changed"
grep -q 'DEFER_USB_DEVICE=""' "${script}" || fail "deferred USB device default changed"

stage_line="$(line_number 'lib971apriltag\.so\.new')"
install_line="$(line_number 'install -m 0755.*lib971apriltag\.so\.new')"
move_line="$(line_number 'mv .*lib971apriltag\.so\.new.*lib971apriltag\.so')"
stop_line="$(line_number 'systemctl stop photonvision\.service')"
reload_line="$(line_number 'systemctl daemon-reload')"
start_line="$(line_number 'systemctl start photonvision\.service')"

[[ -n "${stage_line}" && -n "${install_line}" && -n "${move_line}" ]] || fail "native library is not staged and renamed atomically"
[[ -n "${stop_line}" && -n "${reload_line}" && -n "${start_line}" ]] || fail "missing active-service replacement lifecycle"
[[ "${stop_line}" -lt "${install_line}" ]] || fail "service must stop before staging native library"
[[ "${install_line}" -lt "${move_line}" ]] || fail "native library must be renamed after staging"
[[ "${move_line}" -lt "${reload_line}" && "${reload_line}" -lt "${start_line}" ]] || fail "service reload/start order is incorrect"

grep -q 'HEALTH_TIMEOUT_SECONDS=45' "${script}" || fail "health poll is not bounded to 45 seconds"
grep -q 'curl.*127\.0\.0\.1:5800' "${script}" || fail "missing localhost health check"
grep -q 'http_code.*200' "${script}" || fail "health check does not require HTTP 200"
grep -q 'health check.*failed' "${script}" || fail "health failure does not fail provisioning"

echo "PASS: Orin CUDA service deployment contract"
