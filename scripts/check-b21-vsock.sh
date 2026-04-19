#!/usr/bin/env bash
# Verify B2.1 guest vsock transport path end-to-end.
#
# Checks:
# 1) Firecracker boots with rootfs/kernel
# 2) Host CONNECT 5000 handshake over vsock UDS returns OK
# 3) Guest sends {"type":"agent_ready"}
# 4) Host can send {"type":"stop"} and close cleanly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FC_BIN="${FC_BIN:-/usr/local/bin/firecracker}"
KERNEL="${KERNEL:-${ROOT}/firecracker/kernel/vmlinux}"
ROOTFS="${ROOTFS:-${ROOT}/firecracker/rootfs/agent.ext4}"
BOOT_TIMEOUT_SECONDS="${BOOT_TIMEOUT_SECONDS:-30}"

WORKDIR="$(mktemp -d)"
API_SOCK="${WORKDIR}/firecracker.socket"
VSOCK_UDS="${WORKDIR}/fire.vsock"
ROOTFS_TEST="${WORKDIR}/rootfs-test.ext4"
SERIAL_LOG="${WORKDIR}/serial.log"
TAP_DEV="tapb21$$"

cleanup() {
  if [[ -n "${FC_PID:-}" ]] && kill -0 "${FC_PID}" >/dev/null 2>&1; then
    kill "${FC_PID}" >/dev/null 2>&1 || true
    wait "${FC_PID}" >/dev/null 2>&1 || true
  fi
  sudo ip link del "${TAP_DEV}" >/dev/null 2>&1 || true
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

log() {
  printf '[check-b21-vsock] %s\n' "$*"
}

fail() {
  printf '[check-b21-vsock] ERROR: %s\n' "$*" >&2
  exit 1
}

put_fc() {
  local path="$1"
  local payload="$2"
  local out="${WORKDIR}/curl.out"
  local code
  code="$(curl -sS --unix-socket "${API_SOCK}" \
    -X PUT \
    -H 'Content-Type: application/json' \
    -d "${payload}" \
    -o "${out}" \
    -w '%{http_code}' \
    "http://localhost${path}")"
  if [[ "${code}" != 2* ]]; then
    fail "Firecracker API PUT ${path} failed (HTTP ${code}): $(cat "${out}")"
  fi
}

[[ -x "${FC_BIN}" ]] || fail "Firecracker binary not executable: ${FC_BIN}"
[[ -f "${KERNEL}" ]] || fail "Kernel image not found: ${KERNEL}"
[[ -f "${ROOTFS}" ]] || fail "Rootfs image not found: ${ROOTFS}"

cp --reflink=auto "${ROOTFS}" "${ROOTFS_TEST}"

log "Creating TAP device ${TAP_DEV}"
sudo ip tuntap add dev "${TAP_DEV}" mode tap
sudo ip addr add 172.16.0.1/30 dev "${TAP_DEV}"
sudo ip link set "${TAP_DEV}" up

log "Starting Firecracker"
"${FC_BIN}" --api-sock "${API_SOCK}" >"${SERIAL_LOG}" 2>&1 &
FC_PID=$!

for _ in $(seq 1 100); do
  [[ -S "${API_SOCK}" ]] && break
  sleep 0.1
done
[[ -S "${API_SOCK}" ]] || fail "API socket did not appear: ${API_SOCK}"

log "Configuring VM via API socket"
put_fc /boot-source "{\"kernel_image_path\":\"${KERNEL}\",\"boot_args\":\"console=ttyS0 reboot=k panic=1 pci=off init=/sbin/init\"}"
put_fc /drives/rootfs "{\"drive_id\":\"rootfs\",\"path_on_host\":\"${ROOTFS_TEST}\",\"is_root_device\":true,\"is_read_only\":false}"
put_fc /machine-config '{"vcpu_count":1,"mem_size_mib":512}'
put_fc /network-interfaces/eth0 "{\"iface_id\":\"eth0\",\"guest_mac\":\"06:00:AC:10:00:02\",\"host_dev_name\":\"${TAP_DEV}\"}"
put_fc /vsock "{\"guest_cid\":52,\"uds_path\":\"${VSOCK_UDS}\"}"
put_fc /mmds/config '{"network_interfaces":["eth0"],"version":"V2"}'
put_fc /mmds '{"network":{"ip":"172.16.0.2","gateway":"172.16.0.1","netmask":"255.255.255.252","dns":["8.8.8.8"]},"llm":{"model":"test/model","api_key":"sk-test","max_tokens":128,"temperature":0.1}}'
put_fc /actions '{"action_type":"InstanceStart"}'

for _ in $(seq 1 200); do
  [[ -S "${VSOCK_UDS}" ]] && break
  sleep 0.1
done
[[ -S "${VSOCK_UDS}" ]] || fail "Vsock UDS did not appear: ${VSOCK_UDS}"

log "Waiting for vsock CONNECT to succeed"
if ! python3 - "${VSOCK_UDS}" "${BOOT_TIMEOUT_SECONDS}" <<'PY'
import json
import socket
import sys
import time

uds = sys.argv[1]
timeout_seconds = int(sys.argv[2])
deadline = time.monotonic() + timeout_seconds
attempt = 0
last_error = None

while time.monotonic() < deadline:
    attempt += 1
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(2.0)
        s.connect(uds)
        s.sendall(b"CONNECT 5000\n")

        ack = b""
        while b"\n" not in ack:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError("EOF waiting for CONNECT ack")
            ack += chunk

        line, remainder = ack.split(b"\n", 1)
        if not line.startswith(b"OK "):
            raise RuntimeError(f"Unexpected CONNECT ack: {line!r}")

        buf = remainder
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError("EOF waiting for agent_ready")
            buf += chunk

        first, _rest = buf.split(b"\n", 1)
        event = json.loads(first.decode("utf-8"))
        if event.get("type") != "agent_ready":
            raise RuntimeError(f"Expected agent_ready, got: {event!r}")

        s.sendall(json.dumps({"type": "stop"}).encode("utf-8") + b"\n")
        s.close()
        print("agent_ready received and stop sent")
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(0.5)
    finally:
        try:
            s.close()
        except Exception:
            pass

raise SystemExit(f"Timed out waiting for agent_ready over vsock. Last error: {last_error}")
PY
then
  log "Serial tail for diagnosis:"
  tail -n 120 "${SERIAL_LOG}" || true
  fail "Timed out waiting for agent_ready over vsock."
fi

log "Success: CONNECT handshake and agent_ready verified."
printf '\n=== entrypoint logs ===\n'
grep -E '\[entrypoint\]' "${SERIAL_LOG}" || true
