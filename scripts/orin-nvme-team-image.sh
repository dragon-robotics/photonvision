#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEAM_NUMBER="2375"
STATIC_ADDRESS="10.23.75.15/8"
GATEWAY="10.23.75.4"
HOSTNAME_VALUE="photonvision-orin-nano-ml-tag"
PV_DIR="/opt/photonvision"
JAR_PATH=""
INSTALL_PACKAGES="yes"
HEADLESS="yes"
ENABLE_PERFORMANCE="yes"
RESET_PHOTON_CONFIG="no"
CUDA_APRILTAG_LIBRARY=""
UVC_BANDWIDTH_FIX="no"
DEFER_USB_DEVICE=""
HEALTH_TIMEOUT_SECONDS=45
PHOTONVISION_WAS_ACTIVE="no"

usage() {
    cat <<'EOF'
Usage: sudo bash ./scripts/orin-nvme-team-image.sh --jar <photonvision-linuxarm64.jar> [options]

Turns an NVIDIA Jetson Orin Nano NVMe-root JetPack install into the Team 2375
headless PhotonVision baseline.

Options:
  --jar <path>              PhotonVision linuxarm64 JAR to install.
  --no-apt                  Do not apt-install Java, CUDA build dependencies, NetworkManager, or SSH.
  --keep-gui                Do not switch the system target to headless mode.
  --no-performance-service  Do not install the nvpmodel/jetson_clocks service.
  --cuda-apriltag-library <path>
                            Use an existing JetPack-compatible lib971apriltag.so instead of building it.
  --uvc-bandwidth-fix       Set uvcvideo quirks=128 for two USB 3 cameras.
  --defer-usb-device <id>   Rebind one USB device after PhotonVision starts.
  --reset-photon-config     Remove PhotonVision config DB/settings on this image.
  -h, --help                Show this help.
EOF
}

poll_photonvision_health() {
    local http_code

    while true; do
        if http_code="$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 1 --max-time 1 http://127.0.0.1:5800/)" \
            && [[ "${http_code}" == "200" ]]; then
            return 0
        fi
        sleep 1
    done
}

wait_for_photonvision() {
    if timeout --signal=KILL "${HEALTH_TIMEOUT_SECONDS}s" bash -c 'source "$1"; poll_photonvision_health' bash "${BASH_SOURCE[0]}"; then
        return 0
    fi
    echo "PhotonVision health check failed after ${HEALTH_TIMEOUT_SECONDS} seconds." >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jar)
            JAR_PATH="${2:-}"
            shift 2
            ;;
        --no-apt)
            INSTALL_PACKAGES="no"
            shift
            ;;
        --keep-gui)
            HEADLESS="no"
            shift
            ;;
        --no-performance-service)
            ENABLE_PERFORMANCE="no"
            shift
            ;;
        --cuda-apriltag-library)
            CUDA_APRILTAG_LIBRARY="${2:-}"
            shift 2
            ;;
        --uvc-bandwidth-fix)
            UVC_BANDWIDTH_FIX="yes"
            shift
            ;;
        --defer-usb-device)
            DEFER_USB_DEVICE="${2:-}"
            shift 2
            ;;
        --reset-photon-config)
            RESET_PHOTON_CONFIG="yes"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script with sudo." >&2
    exit 1
fi

if [[ -n "${JAR_PATH}" && ! -f "${JAR_PATH}" ]]; then
    echo "JAR does not exist: ${JAR_PATH}" >&2
    exit 1
fi

if [[ -n "${CUDA_APRILTAG_LIBRARY}" && ! -f "${CUDA_APRILTAG_LIBRARY}" ]]; then
    echo "CUDA AprilTag library does not exist: ${CUDA_APRILTAG_LIBRARY}" >&2
    exit 1
fi

if [[ -n "${DEFER_USB_DEVICE}" && ! "${DEFER_USB_DEVICE}" =~ ^[0-9]+-[0-9]+(\.[0-9]+)*$ ]]; then
    echo "Invalid USB device ID: ${DEFER_USB_DEVICE}" >&2
    exit 1
fi

if [[ -z "${JAR_PATH}" && ! -f "${PV_DIR}/photonvision.jar" ]]; then
    echo "No --jar provided and ${PV_DIR}/photonvision.jar does not exist." >&2
    exit 1
fi

if [[ -r /proc/device-tree/model ]] && ! tr -d '\0' </proc/device-tree/model | grep -qi "NVIDIA Jetson"; then
    echo "Warning: /proc/device-tree/model does not look like a Jetson." >&2
fi

export DEBIAN_FRONTEND=noninteractive

if [[ "${INSTALL_PACKAGES}" == "yes" ]]; then
    apt-get update
    apt-get install -y \
        build-essential \
        cmake \
        git \
        network-manager \
        ninja-build \
        nvidia-cuda-dev \
        openjdk-17-jdk-headless \
        openssh-server
fi

if [[ -z "${CUDA_APRILTAG_LIBRARY}" ]]; then
    if [[ ! -f "${ROOT}/scripts/build-orin-cuda-apriltag.sh" ]]; then
        echo "Cannot build CUDA AprilTag library: PhotonVision source tree is incomplete." >&2
        exit 1
    fi
    echo "Building CUDA AprilTag library for this JetPack installation."
    bash "${ROOT}/scripts/build-orin-cuda-apriltag.sh"
    CUDA_APRILTAG_LIBRARY="${ROOT}/build/orin-apriltag/lib971apriltag.so"
fi

if [[ ! -f "${CUDA_APRILTAG_LIBRARY}" ]]; then
    echo "CUDA AprilTag build did not produce ${CUDA_APRILTAG_LIBRARY}." >&2
    exit 1
fi

if [[ "${UVC_BANDWIDTH_FIX}" == "yes" ]]; then
    cat >/etc/modprobe.d/photonvision-uvcvideo.conf <<'EOF'
options uvcvideo quirks=128
EOF
else
    rm -f /etc/modprobe.d/photonvision-uvcvideo.conf
fi

mkdir -p "${PV_DIR}" "${PV_DIR}/tmp" "${PV_DIR}/logs" "${PV_DIR}/models" "${PV_DIR}/lib"
chmod 755 "${PV_DIR}"

if [[ -n "${JAR_PATH}" ]]; then
    install -m 0644 "${JAR_PATH}" "${PV_DIR}/photonvision.jar"
fi

if [[ -n "${CUDA_APRILTAG_LIBRARY}" ]]; then
    if systemctl is-active --quiet photonvision.service; then
        PHOTONVISION_WAS_ACTIVE="yes"
        systemctl stop photonvision.service
    fi
    install -m 0755 "${CUDA_APRILTAG_LIBRARY}" "${PV_DIR}/lib/lib971apriltag.so.new"
    mv "${PV_DIR}/lib/lib971apriltag.so.new" "${PV_DIR}/lib/lib971apriltag.so"
fi

hostnamectl set-hostname "${HOSTNAME_VALUE}"

mkdir -p /etc/NetworkManager/system-connections
cat >/etc/NetworkManager/system-connections/static-team2375-orin.nmconnection <<EOF
[connection]
id=static-team2375-orin
uuid=23752375-2375-4375-8375-000000000015
type=ethernet
autoconnect=true
autoconnect-priority=100

[ethernet]

[ipv4]
method=manual
address1=${STATIC_ADDRESS},${GATEWAY}
may-fail=false

[ipv6]
method=disabled
EOF
chmod 600 /etc/NetworkManager/system-connections/static-team2375-orin.nmconnection
systemctl enable NetworkManager.service

if [[ "${RESET_PHOTON_CONFIG}" == "yes" ]]; then
    rm -f "${PV_DIR}"/photonvision_config/photon.sqlite*
    rm -f "${PV_DIR}/photonvision_config/networkSettings.json"
fi

cat >/etc/systemd/system/photonvision.service <<EOF
[Unit]
Description=PhotonVision
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${PV_DIR}
Environment=JAVA_TOOL_OPTIONS=-Djava.io.tmpdir=${PV_DIR}/tmp
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${PV_DIR}/lib
MemoryHigh=5G
MemoryMax=6G
ExecStartPre=/bin/sleep 15
EOF

if [[ -n "${DEFER_USB_DEVICE}" ]]; then
    cat >>/etc/systemd/system/photonvision.service <<EOF
ExecStartPre=/bin/sh -c 'if [ -e /sys/bus/usb/drivers/usb/${DEFER_USB_DEVICE} ]; then echo ${DEFER_USB_DEVICE} > /sys/bus/usb/drivers/usb/unbind || true; fi; exit 0'
ExecStartPost=/bin/sh -c 'sleep 8; if [ ! -e /sys/bus/usb/drivers/usb/${DEFER_USB_DEVICE} ]; then echo ${DEFER_USB_DEVICE} > /sys/bus/usb/drivers/usb/bind || true; fi; exit 0'
EOF
fi

cat >>/etc/systemd/system/photonvision.service <<EOF
ExecStart=/usr/bin/java -jar ${PV_DIR}/photonvision.jar
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

if [[ "${ENABLE_PERFORMANCE}" == "yes" ]]; then
    cat >/etc/systemd/system/photonvision-orin-performance.service <<'EOF'
[Unit]
Description=Jetson performance settings for PhotonVision
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -lc 'if command -v nvpmodel >/dev/null; then super=/etc/nvpmodel/nvpmodel_p3767_0003_super.conf; if [ -f "$super" ]; then ln -sfn "$super" /etc/nvpmodel.conf; nvpmodel -m 2; else nvpmodel -m 0; fi; fi; command -v jetson_clocks >/dev/null && jetson_clocks || true'

[Install]
WantedBy=multi-user.target
EOF
    systemctl enable photonvision-orin-performance.service
fi

if [[ "${HEADLESS}" == "yes" ]]; then
    systemctl set-default multi-user.target
    systemctl disable --now gdm3.service lightdm.service sddm.service display-manager.service 2>/dev/null || true
fi

systemctl enable ssh.service 2>/dev/null || systemctl enable sshd.service 2>/dev/null || true
systemctl daemon-reload
systemctl enable photonvision.service

if [[ "${PHOTONVISION_WAS_ACTIVE}" == "yes" ]]; then
    systemctl start photonvision.service
    wait_for_photonvision
fi

echo "Team ${TEAM_NUMBER} Orin Nano baseline installed."
echo "Hostname: ${HOSTNAME_VALUE}"
echo "Static Ethernet: ${STATIC_ADDRESS}, gateway ${GATEWAY}"
echo "PhotonVision: ${PV_DIR}/photonvision.jar"
if [[ -n "${CUDA_APRILTAG_LIBRARY}" ]]; then
    echo "CUDA AprilTag: ${PV_DIR}/lib/lib971apriltag.so"
fi
echo "UVC bandwidth fix: ${UVC_BANDWIDTH_FIX}"
if [[ -n "${DEFER_USB_DEVICE}" ]]; then
    echo "Deferred USB device: ${DEFER_USB_DEVICE}"
fi
echo "Reboot the Jetson before capturing or using the team image."
