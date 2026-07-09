###
# Alternative ARM Runner installer to setup PhotonVision JAR
# for ARM based builds such as Raspberry Pi, Orange Pi, etc.
# This assumes that the image provided to arm-runner-action contains
# the servicefile needed to auto-launch PhotonVision.
###
NEW_JAR=$(realpath $(find . -name photonvision\*-linuxarm64.jar))
echo "Using jar: " $(basename $NEW_JAR)

DEST_PV_LOCATION=/opt/photonvision
sudo mkdir -p $DEST_PV_LOCATION
sudo cp $NEW_JAR ${DEST_PV_LOCATION}/photonvision.jar

# Bring up the team static address during OS networking startup. PhotonVision will still
# reconcile its own managed connection later, but this avoids a boot race where the web UI
# is not reachable at the static IP until PhotonVision sees an active wired connection.
sudo mkdir -p /etc/NetworkManager/system-connections
sudo tee /etc/NetworkManager/system-connections/static-team2375-orin.nmconnection >/dev/null <<'EOF'
[connection]
id=static-team2375-orin
uuid=23752375-2375-4375-8375-000000000015
type=ethernet
autoconnect=true
autoconnect-priority=100

[ethernet]

[ipv4]
method=manual
address1=10.23.75.15/8,10.23.75.4
may-fail=false

[ipv6]
method=disabled
EOF
sudo chmod 600 /etc/NetworkManager/system-connections/static-team2375-orin.nmconnection

# The base image can carry an old PhotonVision settings database. Remove stale
# network settings so first boot uses this branch's team-specific NetworkConfig defaults.
sudo rm -f ${DEST_PV_LOCATION}/photonvision_config/photon.sqlite*
sudo rm -f ${DEST_PV_LOCATION}/photonvision_config/networkSettings.json
