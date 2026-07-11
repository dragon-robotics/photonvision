#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="${root}/scripts/monitor-orin-photonvision.sh"
fixture_dir="$(mktemp -d)"
trap 'rm -rf "${fixture_dir}"' EXIT

header='monotonic_seconds,pid,rss_kib,available_kib,restart_count,active_state,http_status'

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

fixture() {
    local name="$1"
    cat >"${fixture_dir}/${name}.csv"
}

expect_pass() {
    local description="$1"
    local fixture_name="$2"
    local max_rss_growth_mib="$3"
    local min_available_mib="$4"
    local output="${fixture_dir}/${fixture_name}.out.csv"
    local result
    local status

    set +e
    result="$(bash "${script}" \
        --samples-file "${fixture_dir}/${fixture_name}.csv" \
        --output "${output}" \
        --max-rss-growth-mib "${max_rss_growth_mib}" \
        --min-available-mib "${min_available_mib}" 2>&1)"
    status=$?
    set -e

    [[ "${status}" -eq 0 ]] || fail "${description} expected pass, got: ${result}"
    cmp -- "${fixture_dir}/${fixture_name}.csv" "${output}" \
        || fail "${description} did not emit normalized evaluated rows"
}

expect_failure() {
    local description="$1"
    local fixture_name="$2"
    local max_rss_growth_mib="$3"
    local min_available_mib="$4"
    local expected_message="$5"
    local result
    local status

    set +e
    result="$(bash "${script}" \
        --samples-file "${fixture_dir}/${fixture_name}.csv" \
        --max-rss-growth-mib "${max_rss_growth_mib}" \
        --min-available-mib "${min_available_mib}" 2>&1)"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "${description} unexpectedly passed"
    [[ "${result}" == *"${expected_message}"* ]] \
        || fail "${description} did not report ${expected_message}: ${result}"
}

fixture rss-64-pass <<EOF
${header}
0,321,1048576,2097152,7,active,200
60,321,2097152,2097152,7,active,200
61,321,2162688,2097152,7,active,200
EOF

fixture rss-256-fail <<EOF
${header}
0,321,1048576,2097152,7,active,200
60,321,2097152,2097152,7,active,200
61,321,2359296,2097152,7,active,200
EOF

fixture low-available <<EOF
${header}
0,321,1048576,1468006,7,active,200
EOF

fixture restart-increase <<EOF
${header}
0,321,1048576,2097152,7,active,200
5,321,1048576,2097152,8,active,200
EOF

fixture stable <<EOF
${header}
0,321,1048576,2097152,7,active,200
60,321,1179648,2097152,7,active,200
120,321,1245184,2097152,7,active,200
EOF

fixture inactive <<EOF
${header}
0,321,1048576,2097152,7,inactive,200
EOF

fixture http-recovery <<EOF
${header}
0,321,1048576,2097152,7,active,503
5,321,1048576,2097152,7,active,200
EOF

fixture http-two-failures <<EOF
${header}
0,321,1048576,2097152,7,active,503
5,321,1048576,2097152,7,active,000
EOF

fixture malformed-header <<'EOF'
time,pid,rss_kib,available_kib,restart_count,active_state,http_status
0,321,1048576,2097152,7,active,200
EOF

fixture malformed-row <<EOF
${header}
0,321,1048576,2097152,7,active
EOF

fixture malformed-numeric <<EOF
${header}
0,321,not-a-number,2097152,7,active,200
EOF

fixture warmup-only <<EOF
${header}
0,321,1048576,2097152,7,active,200
59,321,1310720,2097152,7,active,200
EOF

fixture warmup-after <<EOF
${header}
0,321,1048576,2097152,7,active,200
59,321,1310720,2097152,7,active,200
60,321,1048576,2097152,7,active,200
61,321,1310720,2097152,7,active,200
EOF

expect_pass "+64 MiB RSS growth" rss-64-pass 128 1536
expect_failure "+256 MiB RSS growth" rss-256-fail 128 1536 "RSS growth"
expect_failure "1.4 GiB available memory" low-available 128 1536 "Available memory"
expect_failure "restart-count increase" restart-increase 128 1536 "Restart count"
expect_pass "stable samples with 2 GiB available" stable 128 1536
expect_failure "inactive service" inactive 128 1536 "not active"
expect_pass "one HTTP failure followed by recovery" http-recovery 128 1536
expect_failure "two consecutive HTTP failures" http-two-failures 128 1536 "consecutive HTTP"
expect_failure "malformed header" malformed-header 128 1536 "header"
expect_failure "malformed row" malformed-row 128 1536 "row"
expect_failure "malformed numeric data" malformed-numeric 128 1536 "numeric"
expect_pass "RSS threshold before 60-second warmup" warmup-only 128 1536
expect_failure "RSS threshold after 60-second warmup" warmup-after 128 1536 "RSS growth"

echo "PASS: Orin PhotonVision monitor threshold contract"
