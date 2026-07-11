#!/usr/bin/env bash
set -euo pipefail

readonly HEADER='monotonic_seconds,pid,rss_kib,available_kib,restart_count,active_state,http_status'
readonly SERVICE='photonvision.service'
readonly WARMUP_SECONDS=60

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

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer"
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    require_nonnegative_integer "${name}" "${value}"
    (( 10#${value} > 0 )) || fail "${name} must be greater than zero"
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

require_positive_integer '--duration-seconds' "${duration_seconds}"
require_positive_integer '--interval-seconds' "${interval_seconds}"
require_nonnegative_integer '--max-rss-growth-mib' "${max_rss_growth_mib}"
require_nonnegative_integer '--min-available-mib' "${min_available_mib}"

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

emit_row() {
    local row="$1"
    if [[ -n "${output_file}" ]]; then
        printf '%s\n' "${row}" >>"${output_file}"
    else
        printf '%s\n' "${row}"
    fi
}

is_decimal() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
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
    local fields=()
    local monotonic_seconds pid rss_kib available_kib restart_count active_state http_status
    local elapsed_seconds rss_growth_kib

    IFS=',' read -r -a fields <<<"${row}"
    [[ ${#fields[@]} -eq 7 ]] || fail "Malformed row ${line_number}: expected 7 CSV columns"

    monotonic_seconds="${fields[0]}"
    pid="${fields[1]}"
    rss_kib="${fields[2]}"
    available_kib="${fields[3]}"
    restart_count="${fields[4]}"
    active_state="${fields[5]}"
    http_status="${fields[6]}"

    is_decimal "${monotonic_seconds}" || fail "Malformed numeric data in row ${line_number}: monotonic_seconds"
    is_nonnegative_integer "${pid}" || fail "Malformed numeric data in row ${line_number}: pid"
    is_nonnegative_integer "${rss_kib}" || fail "Malformed numeric data in row ${line_number}: rss_kib"
    is_nonnegative_integer "${available_kib}" || fail "Malformed numeric data in row ${line_number}: available_kib"
    is_nonnegative_integer "${restart_count}" || fail "Malformed numeric data in row ${line_number}: restart_count"
    [[ "${http_status}" =~ ^[0-9]{3}$ ]] || fail "Malformed numeric data in row ${line_number}: http_status"

    if [[ -z "${start_monotonic}" ]]; then
        start_monotonic="${monotonic_seconds}"
    elif ! awk -v previous="${previous_monotonic}" -v now="${monotonic_seconds}" 'BEGIN { exit !(now >= previous) }'; then
        fail "Malformed row ${line_number}: monotonic_seconds moved backwards"
    fi
    previous_monotonic="${monotonic_seconds}"

    emit_row "${monotonic_seconds},${pid},${rss_kib},${available_kib},${restart_count},${active_state},${http_status}"

    [[ "${active_state}" == 'active' ]] || fail "PhotonVision service is not active (row ${line_number}: ${active_state})"
    (( 10#${pid} > 0 )) || fail "PhotonVision PID is invalid or missing (row ${line_number})"
    (( 10#${available_kib} >= min_available_kib )) \
        || fail "Available memory below threshold in row ${line_number}: ${available_kib} KiB < ${min_available_kib} KiB"

    if [[ -z "${baseline_restart_count}" ]]; then
        baseline_restart_count="${restart_count}"
    elif [[ "${restart_count}" != "${baseline_restart_count}" ]]; then
        fail "Restart count changed in row ${line_number}: ${baseline_restart_count} -> ${restart_count}"
    fi

    if [[ "${http_status}" == '200' ]]; then
        consecutive_http_failures=0
    else
        consecutive_http_failures=$((consecutive_http_failures + 1))
        (( consecutive_http_failures < 2 )) \
            || fail "Two consecutive HTTP failures ending in row ${line_number} (status ${http_status})"
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
        line_number=$((line_number + 1))
        line="${line%$'\r'}"
        [[ -n "${line}" ]] || fail "Malformed row ${line_number}: blank rows are not allowed"
        process_row "${line_number}" "${line}"
    done <&3
    exec 3<&-

    (( line_number > 0 )) || fail "Samples file contains no data rows: ${samples_file}"
}

read_monotonic_seconds() {
    awk '{ printf "%.6f", $1 }' /proc/uptime
}

collect_live_sample() {
    local monotonic_seconds active_state pid rss_kib available_kib restart_count http_status curl_status

    monotonic_seconds="$(read_monotonic_seconds)" || fail 'Could not read /proc/uptime'
    active_state="$(systemctl is-active "${SERVICE}" 2>/dev/null || true)"
    pid="$(systemctl show --property=MainPID --value "${SERVICE}" 2>/dev/null || true)"
    restart_count="$(systemctl show --property=NRestarts --value "${SERVICE}" 2>/dev/null || true)"
    available_kib="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"
    rss_kib=0

    if [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/status" ]]; then
        rss_kib="$(awk '/^VmRSS:/ { print $2; exit }' "/proc/${pid}/status")"
    fi

    set +e
    http_status="$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 1 --max-time 2 http://127.0.0.1:5800/)"
    curl_status=$?
    set -e
    if [[ ${curl_status} -ne 0 || ! "${http_status}" =~ ^[0-9]{3}$ ]]; then
        http_status='000'
    fi

    process_row 1 "${monotonic_seconds},${pid:-0},${rss_kib:-0},${available_kib:-0},${restart_count:-0},${active_state:-unknown},${http_status}"
}

run_live_monitor() {
    local now elapsed_seconds sleep_seconds

    collect_live_sample
    while true; do
        now="$(read_monotonic_seconds)" || fail 'Could not read /proc/uptime'
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

initialize_output
if [[ -n "${samples_file}" ]]; then
    evaluate_samples_file
else
    run_live_monitor
fi
