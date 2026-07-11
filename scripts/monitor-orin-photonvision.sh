#!/usr/bin/env bash
set -euo pipefail

readonly HEADER='monotonic_seconds,pid,rss_kib,available_kib,restart_count,active_state,http_status'
readonly ROW_PATTERN='^[^,"]+(,[^,"]+){6}$'
readonly SERVICE='photonvision.service'
readonly WARMUP_SECONDS=60
readonly MAX_MONOTONIC_SECONDS='2147483647'
readonly MAX_DURATION_SECONDS='604800'
readonly MAX_INTERVAL_SECONDS='86400'
readonly MAX_THRESHOLD_MIB='1048576'
readonly MAX_MEMORY_KIB='1073741824'
readonly MAX_PID='4194304'
readonly MAX_RESTART_COUNT='2147483647'
readonly MAX_SAMPLE_ROWS='10000000'

duration_seconds=300
interval_seconds=5
output_file=''
max_rss_growth_mib=128
min_available_mib=1536
samples_file=''

start_monotonic=''
previous_monotonic=''
baseline_rss_kib=''
baseline_restart_count=''
consecutive_http_failures=0

usage() {
    cat <<'EOF'
Usage: monitor-orin-photonvision.sh [options]

Monitor PhotonVision service health and memory use on an Orin Nano.

Options:
  --duration-seconds <n>       Monitoring duration in seconds (default: 300)
  --interval-seconds <n>       Sample interval in seconds (default: 5)
  --output <csv>               Write normalized CSV rows to this path
  --max-rss-growth-mib <n>     Maximum RSS growth after warmup (default: 128)
  --min-available-mib <n>      Minimum MemAvailable threshold (default: 1536)
  --samples-file <csv>         Evaluate fixture rows without host probes or sleeps
  --help                       Show this help text
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

digit_string_leq() {
    local value="$1"
    local maximum="$2"

    if [[ ${#value} -lt ${#maximum} ]]; then
        return 0
    fi
    if [[ ${#value} -gt ${#maximum} ]]; then
        return 1
    fi
    [[ "${value}" == "${maximum}" || "${value}" < "${maximum}" ]]
}

validate_bounded_nonnegative_integer() {
    local name="$1"
    local raw_value="$2"
    local maximum="$3"
    local destination_name="$4"
    local normalized
    local -n destination="${destination_name}"

    [[ "${raw_value}" =~ ^[0-9]+$ ]] \
        || fail "Invalid numeric value for ${name}: ${raw_value}"
    [[ "${raw_value}" =~ ^0*([1-9][0-9]*|0)$ ]]
    normalized="${BASH_REMATCH[1]}"
    digit_string_leq "${normalized}" "${maximum}" \
        || fail "${name} exceeds maximum ${maximum}: ${raw_value}"
    destination="${normalized}"
}

validate_bounded_positive_integer() {
    local name="$1"
    local destination_name="$4"
    local -n destination="${destination_name}"

    validate_bounded_nonnegative_integer "$@"
    [[ "${destination}" != '0' ]] || fail "${name} must be greater than zero"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration-seconds)
            [[ $# -ge 2 ]] || fail "--duration-seconds requires a value"
            duration_seconds="$2"
            shift 2
            ;;
        --interval-seconds)
            [[ $# -ge 2 ]] || fail "--interval-seconds requires a value"
            interval_seconds="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 && -n "$2" ]] || fail "--output requires a path"
            output_file="$2"
            shift 2
            ;;
        --max-rss-growth-mib)
            [[ $# -ge 2 ]] || fail "--max-rss-growth-mib requires a value"
            max_rss_growth_mib="$2"
            shift 2
            ;;
        --min-available-mib)
            [[ $# -ge 2 ]] || fail "--min-available-mib requires a value"
            min_available_mib="$2"
            shift 2
            ;;
        --samples-file)
            [[ $# -ge 2 && -n "$2" ]] || fail "--samples-file requires a path"
            samples_file="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

validate_bounded_positive_integer \
    '--duration-seconds' "${duration_seconds}" "${MAX_DURATION_SECONDS}" duration_seconds
validate_bounded_positive_integer \
    '--interval-seconds' "${interval_seconds}" "${MAX_INTERVAL_SECONDS}" interval_seconds
validate_bounded_nonnegative_integer \
    '--max-rss-growth-mib' "${max_rss_growth_mib}" "${MAX_THRESHOLD_MIB}" max_rss_growth_mib
validate_bounded_nonnegative_integer \
    '--min-available-mib' "${min_available_mib}" "${MAX_THRESHOLD_MIB}" min_available_mib

max_rss_growth_kib=$((10#${max_rss_growth_mib} * 1024))
min_available_kib=$((10#${min_available_mib} * 1024))

initialize_output() {
    if [[ -n "${output_file}" ]]; then
        : >"${output_file}" || fail "Cannot write output file: ${output_file}"
        printf '%s\n' "${HEADER}" >>"${output_file}"
    else
        printf '%s\n' "${HEADER}"
    fi
}

validate_source_output_paths() {
    [[ -n "${samples_file}" ]] || return 0
    [[ -r "${samples_file}" ]] || fail "Cannot read samples file: ${samples_file}"

    if [[ -n "${output_file}" && -e "${output_file}" && "${samples_file}" -ef "${output_file}" ]]; then
        fail "Samples file and output resolve to the same file: ${samples_file}"
    fi
}

emit_row() {
    local row="$1"
    if [[ -n "${output_file}" ]]; then
        printf '%s\n' "${row}" >>"${output_file}"
    else
        printf '%s\n' "${row}"
    fi
}

validate_monotonic_seconds() {
    local name="$1"
    local raw_value="$2"
    local destination_name="$3"
    local integer_part fraction_part normalized
    local -n destination="${destination_name}"

    [[ "${raw_value}" =~ ^([0-9]+)([.]([0-9]{1,6}))?$ ]] \
        || fail "Invalid ${name}: expected seconds with at most 6 fractional digits"
    integer_part="${BASH_REMATCH[1]}"
    fraction_part="${BASH_REMATCH[3]-}"
    validate_bounded_nonnegative_integer \
        "${name}" "${integer_part}" "${MAX_MONOTONIC_SECONDS}" integer_part
    normalized="${integer_part}"
    if [[ -n "${fraction_part}" ]]; then
        normalized="${normalized}.${fraction_part}"
    fi
    destination="${normalized}"
}

elapsed_from_start() {
    awk -v now="$1" -v start="${start_monotonic}" 'BEGIN { printf "%.6f", now - start }'
}

is_at_least_warmup() {
    awk -v elapsed="$1" -v warmup="${WARMUP_SECONDS}" 'BEGIN { exit !(elapsed >= warmup) }'
}

process_row() {
    local line_number="$1"
    local row="$2"
    local monotonic_seconds pid rss_kib available_kib restart_count active_state http_status
    local elapsed_seconds rss_growth_kib

    [[ "${row}" =~ ${ROW_PATTERN} ]] \
        || fail "Malformed row ${line_number}: expected exactly seven unquoted, nonempty CSV fields"
    IFS=',' read -r monotonic_seconds pid rss_kib available_kib restart_count active_state http_status <<<"${row}"

    validate_monotonic_seconds \
        "row ${line_number} monotonic_seconds" "${monotonic_seconds}" monotonic_seconds
    validate_bounded_nonnegative_integer \
        "row ${line_number} pid" "${pid}" "${MAX_PID}" pid
    validate_bounded_nonnegative_integer \
        "row ${line_number} rss_kib" "${rss_kib}" "${MAX_MEMORY_KIB}" rss_kib
    validate_bounded_nonnegative_integer \
        "row ${line_number} available_kib" "${available_kib}" "${MAX_MEMORY_KIB}" available_kib
    validate_bounded_nonnegative_integer \
        "row ${line_number} restart_count" "${restart_count}" "${MAX_RESTART_COUNT}" restart_count
    [[ "${http_status}" =~ ^[0-9]{3}$ ]] || fail "Malformed numeric data in row ${line_number}: http_status"

    if [[ -z "${start_monotonic}" ]]; then
        start_monotonic="${monotonic_seconds}"
    elif ! awk -v previous="${previous_monotonic}" -v now="${monotonic_seconds}" 'BEGIN { exit !(now >= previous) }'; then
        fail "Malformed row ${line_number}: monotonic_seconds moved backwards"
    fi
    previous_monotonic="${monotonic_seconds}"

    emit_row "${monotonic_seconds},${pid},${rss_kib},${available_kib},${restart_count},${active_state},${http_status}"

    [[ "${active_state}" == 'active' ]] || fail "PhotonVision service is not active (row ${line_number}: ${active_state})"
    [[ "${pid}" != '0' ]] || fail "PhotonVision PID is invalid or missing (row ${line_number})"
    digit_string_leq "${min_available_kib}" "${available_kib}" \
        || fail "Available memory below threshold in row ${line_number}: ${available_kib} KiB < ${min_available_kib} KiB"

    if [[ -z "${baseline_restart_count}" ]]; then
        baseline_restart_count="${restart_count}"
    elif [[ "${restart_count}" != "${baseline_restart_count}" ]]; then
        fail "Restart count changed in row ${line_number}: ${baseline_restart_count} -> ${restart_count}"
    fi

    if [[ "${http_status}" == '200' ]]; then
        consecutive_http_failures=0
    else
        if [[ "${consecutive_http_failures}" == '1' ]]; then
            fail "Two consecutive HTTP failures ending in row ${line_number} (status ${http_status})"
        fi
        [[ "${consecutive_http_failures}" == '0' ]] \
            || fail "Internal HTTP failure counter is invalid: ${consecutive_http_failures}"
        consecutive_http_failures=1
    fi

    elapsed_seconds="$(elapsed_from_start "${monotonic_seconds}")"
    if is_at_least_warmup "${elapsed_seconds}"; then
        if [[ -z "${baseline_rss_kib}" ]]; then
            baseline_rss_kib="${rss_kib}"
        else
            rss_growth_kib=$((10#${rss_kib} - 10#${baseline_rss_kib}))
            (( rss_growth_kib <= max_rss_growth_kib )) \
                || fail "RSS growth exceeded threshold in row ${line_number}: ${rss_growth_kib} KiB > ${max_rss_growth_kib} KiB"
        fi
    fi
}

evaluate_samples_file() {
    local line
    local line_number=0
    local header=''

    [[ -r "${samples_file}" ]] || fail "Cannot read samples file: ${samples_file}"
    exec 3<"${samples_file}" || fail "Cannot open samples file: ${samples_file}"
    IFS= read -r header <&3 || fail "Samples file is empty: ${samples_file}"
    header="${header%$'\r'}"
    [[ "${header}" == "${HEADER}" ]] || fail "Malformed samples header; expected: ${HEADER}"

    while IFS= read -r line || [[ -n "${line}" ]]; do
        validate_bounded_nonnegative_integer \
            'sample row count' "${line_number}" "${MAX_SAMPLE_ROWS}" line_number
        [[ "${line_number}" != "${MAX_SAMPLE_ROWS}" ]] \
            || fail "Samples file exceeds maximum ${MAX_SAMPLE_ROWS} rows"
        line_number=$((10#${line_number} + 1))
        line="${line%$'\r'}"
        [[ -n "${line}" ]] || fail "Malformed row ${line_number}: blank rows are not allowed"
        process_row "${line_number}" "${line}"
    done <&3
    exec 3<&-

    [[ "${line_number}" != '0' ]] || fail "Samples file contains no data rows: ${samples_file}"
}

read_monotonic_seconds() {
    local monotonic_seconds ignored
    IFS=' ' read -r monotonic_seconds ignored </proc/uptime || return 1
    printf '%s' "${monotonic_seconds}"
}

read_live_rss_kib() {
    local pid="$1"
    local destination_name="$2"
    local status_file raw_rss_kib
    local -n destination="${destination_name}"

    [[ "${pid}" =~ ^[0-9]+$ ]] || fail "PhotonVision PID is invalid or missing: ${pid:-missing}"
    validate_bounded_positive_integer 'live PhotonVision PID' "${pid}" "${MAX_PID}" pid
    status_file="/proc/${pid}/status"
    [[ -f "${status_file}" && -r "${status_file}" ]] \
        || fail "PhotonVision PID status disappeared: ${status_file}"
    if ! raw_rss_kib="$(awk '/^VmRSS:/ { print $2; exit }' "${status_file}" 2>/dev/null)"; then
        fail "Could not read VmRSS from PhotonVision PID status: ${status_file}"
    fi
    [[ -n "${raw_rss_kib}" ]] || fail "VmRSS is absent from PhotonVision PID status: ${status_file}"
    [[ "${raw_rss_kib}" =~ ^[0-9]+$ ]] || fail "VmRSS is not numeric for PhotonVision PID ${pid}: ${raw_rss_kib}"
    validate_bounded_nonnegative_integer \
        'live VmRSS KiB' "${raw_rss_kib}" "${MAX_MEMORY_KIB}" raw_rss_kib
    [[ "${raw_rss_kib}" != '0' ]] || fail "VmRSS is zero for PhotonVision PID ${pid}"
    destination="${raw_rss_kib}"
}

collect_live_sample() {
    local monotonic_seconds active_state pid rss_kib available_kib restart_count http_status curl_status

    monotonic_seconds="$(read_monotonic_seconds)" || fail 'Could not read /proc/uptime'
    active_state="$(systemctl is-active "${SERVICE}" 2>/dev/null || true)"
    [[ "${active_state}" == 'active' ]] \
        || fail "PhotonVision service is not active: ${active_state:-unknown}"
    pid="$(systemctl show --property=MainPID --value "${SERVICE}" 2>/dev/null || true)"
    restart_count="$(systemctl show --property=NRestarts --value "${SERVICE}" 2>/dev/null || true)"
    if ! available_kib="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null)"; then
        fail 'Could not read MemAvailable from /proc/meminfo'
    fi

    read_live_rss_kib "${pid}" rss_kib

    set +e
    http_status="$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 1 --max-time 2 http://127.0.0.1:5800/)"
    curl_status=$?
    set -e
    if [[ "${curl_status}" != '0' || ! "${http_status}" =~ ^[0-9]{3}$ ]]; then
        http_status='000'
    fi

    process_row 1 "${monotonic_seconds},${pid:-0},${rss_kib:-0},${available_kib:-0},${restart_count:-0},${active_state:-unknown},${http_status}"
}

run_live_monitor() {
    local now elapsed_seconds sleep_seconds

    collect_live_sample
    while true; do
        now="$(read_monotonic_seconds)" || fail 'Could not read /proc/uptime'
        validate_monotonic_seconds 'live monotonic_seconds' "${now}" now
        elapsed_seconds="$(elapsed_from_start "${now}")"
        if awk -v elapsed="${elapsed_seconds}" -v duration="${duration_seconds}" 'BEGIN { exit !(elapsed >= duration) }'; then
            break
        fi

        sleep_seconds="$(awk -v elapsed="${elapsed_seconds}" -v duration="${duration_seconds}" -v interval="${interval_seconds}" \
            'BEGIN { remaining = duration - elapsed; print (remaining < interval ? remaining : interval) }')"
        sleep "${sleep_seconds}"
        collect_live_sample
    done
}

validate_source_output_paths
initialize_output
if [[ -n "${samples_file}" ]]; then
    evaluate_samples_file
else
    run_live_monitor
fi
