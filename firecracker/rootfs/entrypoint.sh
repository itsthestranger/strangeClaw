#!/bin/sh
# Fire guest bootstrap:
# - validates vsock device
# - fetches MMDS V2 token + metadata
# - applies guest network config
# - writes full agent config for runtime
# - starts agent on vsock port 5000

set -eu

MMDS_IP="${MMDS_IP:-169.254.169.254}"
MMDS_IFACE="${MMDS_IFACE:-eth0}"
MMDS_TIMEOUT_SECONDS="${MMDS_TIMEOUT_SECONDS:-30}"
MMDS_RETRY_INTERVAL_SECONDS="${MMDS_RETRY_INTERVAL_SECONDS:-0.5}"
MMDS_TOKEN_TTL_SECONDS="${MMDS_TOKEN_TTL_SECONDS:-300}"

RUN_DIR="/run/strangeclaw"
AGENT_CONFIG_PATH="${RUN_DIR}/config.json"
METADATA_TMP_PATH="${RUN_DIR}/mmds.json.tmp"
METADATA_PATH="${RUN_DIR}/mmds.json"
NET_IP_PATH="${RUN_DIR}/network.ip"
NET_PREFIX_PATH="${RUN_DIR}/network.prefix"
NET_GATEWAY_PATH="${RUN_DIR}/network.gateway"
RESOLV_PATH="${RUN_DIR}/resolv.conf"

log() {
  printf '[entrypoint] %s\n' "$*"
}

fatal() {
  printf '[entrypoint] FATAL: %s\n' "$*" >&2
  exit 1
}

retry_until() {
  timeout_seconds="$1"
  interval_seconds="$2"
  shift 2

  start_seconds="$(date +%s)"
  while true; do
    if "$@"; then
      return 0
    fi

    now_seconds="$(date +%s)"
    elapsed_seconds=$((now_seconds - start_seconds))
    if [ "${elapsed_seconds}" -ge "${timeout_seconds}" ]; then
      return 1
    fi
    sleep "${interval_seconds}"
  done
}

if [ ! -e /dev/vsock ]; then
  fatal "/dev/vsock not found — guest kernel must have CONFIG_VIRTIO_VSOCKETS=y (built-in)"
fi

mkdir -p "${RUN_DIR}" /output
chmod 700 "${RUN_DIR}"

log "Bringing up ${MMDS_IFACE} for MMDS access"
ip addr add 169.254.1.1/16 dev "${MMDS_IFACE}" 2>/dev/null || true
ip link set "${MMDS_IFACE}" up
ip route add "${MMDS_IP}" dev "${MMDS_IFACE}" 2>/dev/null || \
  ip route replace "${MMDS_IP}" dev "${MMDS_IFACE}"

MMDS_TOKEN=""
fetch_mmds_token() {
  MMDS_TOKEN="$(curl -fsS -X PUT "http://${MMDS_IP}/latest/api/token" \
    -H "X-metadata-token-ttl-seconds: ${MMDS_TOKEN_TTL_SECONDS}" 2>/dev/null || true)"
  [ -n "${MMDS_TOKEN}" ]
}

log "Acquiring MMDS V2 session token"
if ! retry_until "${MMDS_TIMEOUT_SECONDS}" "${MMDS_RETRY_INTERVAL_SECONDS}" fetch_mmds_token; then
  fatal "Timed out acquiring MMDS V2 token after ${MMDS_TIMEOUT_SECONDS}s"
fi

fetch_and_parse_mmds() {
  if ! curl -fsS \
    -H "X-metadata-token: ${MMDS_TOKEN}" \
    -H "Accept: application/json" \
    "http://${MMDS_IP}/" > "${METADATA_TMP_PATH}" 2>/dev/null; then
    return 1
  fi

  python3 - "${METADATA_TMP_PATH}" "${AGENT_CONFIG_PATH}" "${NET_IP_PATH}" \
    "${NET_PREFIX_PATH}" "${NET_GATEWAY_PATH}" "${RESOLV_PATH}" <<'PY'
import ipaddress
import json
import os
import stat
import sys

metadata_path, config_path, ip_path, prefix_path, gateway_path, resolv_path = sys.argv[1:7]

try:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    sys.exit(1)

if not isinstance(payload, dict):
    sys.exit(1)

network = payload.get("network")
agent_config = payload.get("config")
if not isinstance(network, dict) or not isinstance(agent_config, dict):
    sys.exit(1)

ip_value = network.get("ip")
gateway_value = network.get("gateway")
netmask_value = network.get("netmask")
dns_value = network.get("dns")

if not isinstance(ip_value, str) or not isinstance(gateway_value, str) or not isinstance(netmask_value, str):
    sys.exit(1)
if not isinstance(dns_value, list) or not dns_value:
    sys.exit(1)
if not all(isinstance(item, str) for item in dns_value):
    sys.exit(1)

try:
    ipaddress.IPv4Address(ip_value)
    ipaddress.IPv4Address(gateway_value)
    prefix_len = ipaddress.IPv4Network(f"0.0.0.0/{netmask_value}").prefixlen
    for dns_entry in dns_value:
        ipaddress.IPv4Address(dns_entry)
except Exception:
    sys.exit(1)

os.makedirs(os.path.dirname(config_path), mode=0o700, exist_ok=True)

# No credential values are written here. All secrets are held by the host broker.
fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(agent_config, handle, ensure_ascii=True, separators=(",", ":"))
    handle.write("\n")

for path, value in (
    (ip_path, ip_value),
    (prefix_path, str(prefix_len)),
    (gateway_path, gateway_value),
):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{value}\n")

with open(resolv_path, "w", encoding="utf-8") as handle:
    for dns_entry in dns_value:
        handle.write(f"nameserver {dns_entry}\n")

for path in (ip_path, prefix_path, gateway_path, resolv_path):
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
PY
}

log "Fetching MMDS metadata (network + agent config)"
if ! retry_until "${MMDS_TIMEOUT_SECONDS}" "${MMDS_RETRY_INTERVAL_SECONDS}" fetch_and_parse_mmds; then
  fatal "Timed out fetching/parsing MMDS metadata after ${MMDS_TIMEOUT_SECONDS}s"
fi

mv -f "${METADATA_TMP_PATH}" "${METADATA_PATH}"

REAL_IP="$(cat "${NET_IP_PATH}")"
REAL_PREFIX="$(cat "${NET_PREFIX_PATH}")"
REAL_GATEWAY="$(cat "${NET_GATEWAY_PATH}")"

log "Applying guest network configuration"
ip addr flush dev "${MMDS_IFACE}"
ip addr add "${REAL_IP}/${REAL_PREFIX}" dev "${MMDS_IFACE}"
ip link set "${MMDS_IFACE}" up
ip route replace default via "${REAL_GATEWAY}" dev "${MMDS_IFACE}"
cat "${RESOLV_PATH}" > /etc/resolv.conf

export PATH="/opt/strangeclaw/.venv/bin:${PATH}"
# Unbuffered stdio so guest agent logs/tracebacks reach the serial console
# (captured host-side) even if the process is killed before a buffer flush.
export PYTHONUNBUFFERED=1
cd /opt/strangeclaw
log "Starting agent runtime"
exec /opt/strangeclaw/.venv/bin/python -u -m agent.agent --vsock-port 5000
