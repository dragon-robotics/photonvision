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

next_fi_line() {
    local start_line="$1"
    awk -v start_line="${start_line}" \
        'NR > start_line && /^[[:space:]]*fi$/ { print NR; exit }' "${script}"
}

run_health_check() {
    local http_code="$1"
    local curl_exit="$2"
    local health_timeout="$3"
    local outer_timeout="$4"

    STUB_HTTP_CODE="${http_code}" STUB_CURL_EXIT="${curl_exit}" \
        timeout "${outer_timeout}s" bash -c '
            source "$1"
            HEALTH_TIMEOUT_SECONDS="$2"
            wait_for_photonvision
        ' bash "${script}" "${health_timeout}"
}

expect_health_failure() {
    local description="$1"
    local http_code="$2"
    local curl_exit="$3"
    local output
    local status

    set +e
    output="$(run_health_check "${http_code}" "${curl_exit}" 1 4 2>&1)"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "${description} unexpectedly passed"
    [[ "${status}" -ne 124 ]] || fail "${description} escaped the production hard timeout"
    [[ "${output}" == *"PhotonVision health check failed after 1 seconds."* ]] \
        || fail "${description} did not report the health failure"
}

bash -n "${script}"

grep -q '^set -euo pipefail$' "${script}" || fail "provisioning script must keep strict mode"
grep -q '^HEALTH_TIMEOUT_SECONDS=45$' "${script}" || fail "production health timeout must be 45 seconds"
grep -q '^MemoryHigh=5G$' "${script}" || fail "missing MemoryHigh=5G"
grep -q '^MemoryMax=6G$' "${script}" || fail "missing MemoryMax=6G"
grep -q '^Restart=always$' "${script}" || fail "missing Restart=always"
grep -q '^RestartSec=2$' "${script}" || fail "missing RestartSec=2"
grep -q '10\.23\.75\.15/8' "${script}" || fail "missing Team 2375 static address"
grep -q '10\.23\.75\.4' "${script}" || fail "missing Team 2375 gateway"
grep -q 'photonvision-orin-nano-ml-tag' "${script}" || fail "missing Orin hostname"
grep -q 'RESET_PHOTON_CONFIG="no"' "${script}" || fail "PhotonVision reset default changed"
grep -q 'UVC_BANDWIDTH_FIX="no"' "${script}" || fail "UVC bandwidth fix default changed"
grep -q 'DEFER_USB_DEVICE=""' "${script}" || fail "deferred USB device default changed"

active_if_line="$(line_number '^[[:space:]]+if systemctl is-active --quiet photonvision\.service; then$')"
active_assignment_line="$(line_number '^[[:space:]]+PHOTONVISION_WAS_ACTIVE="yes"$')"
stop_line="$(line_number '^[[:space:]]+systemctl stop photonvision\.service$')"
active_fi_line="$(next_fi_line "${active_if_line}")"
stop_count="$(grep -c -E 'systemctl stop photonvision\.service' "${script}" || true)"

[[ -n "${active_if_line}" && -n "${active_assignment_line}" && -n "${stop_line}" && -n "${active_fi_line}" ]] \
    || fail "missing conditional active-service detection"
[[ "${stop_count}" -eq 1 ]] || fail "service stop must occur only in the active-service branch"
[[ "${active_if_line}" -lt "${active_assignment_line}" \
    && "${active_assignment_line}" -lt "${stop_line}" \
    && "${stop_line}" -lt "${active_fi_line}" ]] \
    || fail "active-service assignment and stop must stay inside systemctl is-active branch"

stage_line="$(line_number 'lib971apriltag\.so\.new')"
install_line="$(line_number 'install -m 0755.*lib971apriltag\.so\.new')"
move_line="$(line_number 'mv .*lib971apriltag\.so\.new.*lib971apriltag\.so')"
reload_line="$(line_number '^systemctl daemon-reload$')"

restart_if_line="$(line_number '^if \[\[ "\$\{PHOTONVISION_WAS_ACTIVE\}" == "yes" \]\]; then$')"
start_line="$(line_number '^[[:space:]]+systemctl start photonvision\.service$')"
health_call_line="$(line_number '^[[:space:]]+wait_for_photonvision$')"
restart_fi_line="$(next_fi_line "${restart_if_line}")"
success_line="$(line_number '^echo "Team .* Orin Nano baseline installed\."$')"

[[ -n "${stage_line}" && -n "${install_line}" && -n "${move_line}" ]] \
    || fail "native library is not staged and renamed atomically"
[[ "${stop_line}" -lt "${install_line}" ]] || fail "service must stop before staging native library"
[[ "${install_line}" -lt "${move_line}" ]] || fail "native library must be renamed after staging"
[[ -n "${reload_line}" && -n "${restart_if_line}" && -n "${start_line}" \
    && -n "${health_call_line}" && -n "${restart_fi_line}" && -n "${success_line}" ]] \
    || fail "missing guarded restart and health lifecycle"
[[ "${move_line}" -lt "${reload_line}" && "${reload_line}" -lt "${restart_if_line}" ]] \
    || fail "service reload order is incorrect"
[[ "${restart_if_line}" -lt "${start_line}" \
    && "${start_line}" -lt "${health_call_line}" \
    && "${health_call_line}" -lt "${restart_fi_line}" ]] \
    || fail "restart and health check must stay inside the previously-active branch"
[[ "${health_call_line}" -lt "${success_line}" ]] \
    || fail "success output must follow the required health check"

grep -q 'timeout --signal=KILL.*HEALTH_TIMEOUT_SECONDS.*poll_photonvision_health' "${script}" \
    || fail "health polling loop lacks a hard coreutils timeout"
grep -q 'curl.*127\.0\.0\.1:5800' "${script}" || fail "missing localhost health check"
grep -q 'http_code.*200' "${script}" || fail "health check does not require HTTP 200"
grep -q 'health check.*failed' "${script}" || fail "health failure does not fail provisioning"

stub_dir="$(mktemp -d)"
trap 'rm -rf "${stub_dir}"' EXIT
cat >"${stub_dir}/curl" <<'EOF'
#!/usr/bin/env bash
printf '%s' "${STUB_HTTP_CODE:-000}"
exit "${STUB_CURL_EXIT:-0}"
EOF
chmod 755 "${stub_dir}/curl"
export PATH="${stub_dir}:${PATH}"

run_health_check 200 0 1 4 >/dev/null \
    || fail "curl success with exact HTTP 200 must pass"
expect_health_failure "curl failure reporting HTTP 200" 200 7
expect_health_failure "HTTP redirect" 302 0
expect_health_failure "non-200 HTTP response" 503 0

start_ms="$(date +%s%3N)"
set +e
timed_output="$(run_health_check 503 0 1 4 2>&1)"
timed_status=$?
set -e
elapsed_ms=$(( $(date +%s%3N) - start_ms ))

[[ "${timed_status}" -ne 0 && "${timed_status}" -ne 124 ]] \
    || fail "unhealthy polling escaped the production hard timeout"
[[ "${timed_output}" == *"PhotonVision health check failed after 1 seconds."* ]] \
    || fail "timed health failure was not reported"
[[ "${elapsed_ms}" -le 2500 ]] \
    || fail "one-second health timeout took ${elapsed_ms} ms"

echo "PASS: Orin CUDA service deployment contract (${elapsed_ms} ms hard-bound check)"
