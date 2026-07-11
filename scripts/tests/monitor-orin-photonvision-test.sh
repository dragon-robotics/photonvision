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

single_row_fixture() {
    printf '%s\n%s\n' "${header}" "$2" >"${fixture_dir}/$1.csv"
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

expect_source_output_alias_failure() {
    local description="$1"
    local samples="$2"
    local output="$3"
    local backup="${samples}.backup"
    local result
    local status

    cp -- "${samples}" "${backup}"
    set +e
    result="$(bash "${script}" \
        --samples-file "${samples}" \
        --output "${output}" 2>&1)"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "${description} unexpectedly passed"
    [[ "${result}" == *"same file"* ]] \
        || fail "${description} did not report the file alias: ${result}"
    cmp -- "${backup}" "${samples}" \
        || fail "${description} truncated or changed the samples file"
}

expect_invocation_failure() {
    local description="$1"
    local expected_message="$2"
    shift 2
    local result
    local status

    set +e
    result="$(bash "${script}" "$@" 2>&1)"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "${description} unexpectedly passed"
    [[ "${result}" == *"${expected_message}"* ]] \
        || fail "${description} did not report ${expected_message}: ${result}"
}

expect_live_rss_failure() {
    local description="$1"
    local mode="$2"
    local expected_message="$3"
    local status_contents="${4-__missing__}"
    local awk_command="${5-awk}"
    local proc_root="${fixture_dir}/proc-${mode}"
    local status_path="${proc_root}/4242/status"
    local result
    local status

    mkdir -p "${proc_root}/4242"
    if [[ "${status_contents}" != '__missing__' ]]; then
        printf '%s' "${status_contents}" >"${status_path}"
    fi

    set +e
    result="$(
        MONITOR_ORIN_PHOTONVISION_SOURCE_ONLY=1 \
        MONITOR_PROC_ROOT="${proc_root}" \
        MONITOR_AWK_COMMAND="${awk_command}" \
        MONITOR_SCRIPT="${script}" \
        bash -c '
            source "${MONITOR_SCRIPT}"
            rss_kib=""
            read_live_rss_kib 4242 rss_kib
        ' bash 2>&1
    )"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "${description} unexpectedly passed"
    [[ "${result}" == *"${expected_message}"* ]] \
        || fail "${description} did not report ${expected_message}: ${result}"
}

expect_setter_settings_failure() {
    local settings_json="$1"
    local result
    local status

    set +e
    result="$(cd "${root}/photon-client" && \
        node scripts/set-photonvision-pipeline.mjs unused.invalid:1 camera "${settings_json}" 2>&1)"
    status=$?
    set -e

    [[ "${status}" -ne 0 ]] || fail "setter accepted invalid settings: ${settings_json}"
    [[ "${result}" == *"Settings JSON"* ]] \
        || fail "setter reported the wrong validation error for ${settings_json}: ${result}"
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

single_row_fixture trailing-empty-field '0,321,1048576,2097152,7,active,200,'
single_row_fixture extra-populated-field '0,321,1048576,2097152,7,active,200,extra'
single_row_fixture middle-empty-field '0,321,,2097152,7,active,200'
single_row_fixture quoted-field '0,"321",1048576,2097152,7,active,200'

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

single_row_fixture same-path-source '0,321,1048576,2097152,7,active,200'
single_row_fixture hardlink-source '0,321,1048576,2097152,7,active,200'
single_row_fixture bounds-valid '0,4194304,1073741824,1073741824,2147483647,active,200'
single_row_fixture pid-overflow '0,4194305,1048576,2097152,7,active,200'
single_row_fixture rss-overflow '0,321,1073741825,2097152,7,active,200'
single_row_fixture available-overflow '0,321,1048576,1073741825,7,active,200'
single_row_fixture restart-overflow '0,321,1048576,2097152,2147483648,active,200'

ln "${fixture_dir}/hardlink-source.csv" "${fixture_dir}/hardlink-output.csv"
[[ "${fixture_dir}/hardlink-source.csv" -ef "${fixture_dir}/hardlink-output.csv" ]] \
    || fail "test environment did not create a hardlink"

expect_source_output_alias_failure \
    "identical samples and output path" \
    "${fixture_dir}/same-path-source.csv" \
    "${fixture_dir}/same-path-source.csv"
expect_source_output_alias_failure \
    "hardlinked samples and output paths" \
    "${fixture_dir}/hardlink-source.csv" \
    "${fixture_dir}/hardlink-output.csv"

expect_invocation_failure \
    "reviewer minimum-available overflow probe" "exceeds maximum" \
    --samples-file "${fixture_dir}/stable.csv" \
    --min-available-mib 18014398509481984
expect_invocation_failure \
    "reviewer maximum-RSS overflow probe" "exceeds maximum" \
    --samples-file "${fixture_dir}/stable.csv" \
    --max-rss-growth-mib 9223372036854775808
expect_invocation_failure \
    "duration above seven days" "exceeds maximum" \
    --samples-file "${fixture_dir}/stable.csv" \
    --duration-seconds 604801
expect_invocation_failure \
    "interval above one day" "exceeds maximum" \
    --samples-file "${fixture_dir}/stable.csv" \
    --interval-seconds 86401
expect_invocation_failure \
    "PID row overflow" "exceeds maximum" \
    --samples-file "${fixture_dir}/pid-overflow.csv"
expect_invocation_failure \
    "RSS row overflow" "exceeds maximum" \
    --samples-file "${fixture_dir}/rss-overflow.csv"
expect_invocation_failure \
    "available-memory row overflow" "exceeds maximum" \
    --samples-file "${fixture_dir}/available-overflow.csv"
expect_invocation_failure \
    "restart-count row overflow" "exceeds maximum" \
    --samples-file "${fixture_dir}/restart-overflow.csv"
bash "${script}" \
    --samples-file "${fixture_dir}/bounds-valid.csv" \
    --duration-seconds 604800 --interval-seconds 86400 \
    --max-rss-growth-mib 1048576 --min-available-mib 1048576 >/dev/null \
    || fail "numeric boundary values expected pass"
expect_live_rss_failure "disappeared PID status" missing "PID status disappeared"
expect_live_rss_failure "absent VmRSS" absent "VmRSS is absent" $'Name:\tphotonvision\n'
expect_live_rss_failure "zero VmRSS" zero "VmRSS is zero" $'VmRSS:\t0 kB\n'
expect_live_rss_failure "nonnumeric VmRSS" nonnumeric "VmRSS is not numeric" $'VmRSS:\tnot-a-number kB\n'
expect_live_rss_failure \
    "VmRSS awk read failure" awk-failure "Could not read VmRSS" $'VmRSS:\t1048576 kB\n' false

expect_pass "+64 MiB RSS growth" rss-64-pass 128 1536
expect_failure "+256 MiB RSS growth" rss-256-fail 128 1536 "RSS growth"
expect_failure "1.4 GiB available memory" low-available 128 1536 "Available memory"
expect_failure "restart-count increase" restart-increase 128 1536 "Restart count"
expect_pass "stable samples with 2 GiB available" stable 128 1536
expect_failure "inactive service" inactive 128 1536 "not active"
expect_pass "one HTTP failure followed by recovery" http-recovery 128 1536
expect_failure "two consecutive HTTP failures" http-two-failures 128 1536 "consecutive HTTP"
expect_failure "malformed header" malformed-header 128 1536 "header"
expect_failure "trailing empty eighth CSV field" trailing-empty-field 128 1536 "CSV fields"
expect_failure "extra populated CSV field" extra-populated-field 128 1536 "CSV fields"
expect_failure "missing CSV field" malformed-row 128 1536 "CSV fields"
expect_failure "middle empty CSV field" middle-empty-field 128 1536 "CSV fields"
expect_failure "quoted CSV field" quoted-field 128 1536 "CSV fields"
expect_failure "malformed numeric data" malformed-numeric 128 1536 "numeric"
expect_pass "RSS threshold before 60-second warmup" warmup-only 128 1536
expect_failure "RSS threshold after 60-second warmup" warmup-after 128 1536 "RSS growth"
for invalid_settings in 'null' '[]' '42' '"text"' 'true' '{}'; do
    expect_setter_settings_failure "${invalid_settings}"
done

echo "PASS: Orin PhotonVision monitor threshold contract"
