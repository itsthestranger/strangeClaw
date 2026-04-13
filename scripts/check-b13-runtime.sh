#!/usr/bin/env bash
# Verify B1.3 runtime path: Firecracker boot + MMDS V2 fetch + llm.json write.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FC_BIN="${FC_BIN:-/usr/local/bin/firecracker}"
KERNEL="${KERNEL:-${ROOT}/firecracker/kernel/vmlinux}"
ROOTFS="${ROOTFS:-${ROOT}/firecracker/rootfs/agent.ext4}"

WORKDIR="$(mktemp -d)"
API_SOCK="${WORKDIR}/firecracker.socket"
VSOCK_UDS="${WORKDIR}/fire.vsock"
ROOTFS_TEST="${WORKDIR}/rootfs-test.ext4"
SERIAL_LOG="${WORKDIR}/serial.log"
TAP_DEV="tapb13$$"

cleanup() {
  if [[ -n "${FC_PID:-}" ]] && kill -0 "${FC_PID}" >/dev/null 2>&1; then
    kill "${FC_PID}" >/dev/null 2>&1 || true
    wait "${FC_PID}" >/dev/null 2>&1 || true
  fi
  sudo ip link del "${TAP_DEV}" >/dev/null 2>&1 || true
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

fail() {
  printf '[check-b13-runtime] ERROR: %s\n' "$*" >&2
  exit 1
}

put_fc() {
  local path="$1"
  local payload="$2"
  curl -fsS --unix-socket "${API_SOCK}" \
    -X PUT \
    -H 'Content-Type: application/json' \
    -d "${payload}" \
    "http://localhost${path}" >/dev/null
}

[[ -x "${FC_BIN}" ]] || fail "Firecracker binary not executable: ${FC_BIN}"
[[ -f "${KERNEL}" ]] || fail "Kernel image not found: ${KERNEL}"
[[ -f "${ROOTFS}" ]] || fail "Rootfs image not found: ${ROOTFS}"

cp --reflink=auto "${ROOTFS}" "${ROOTFS_TEST}"

sudo ip tuntap add dev "${TAP_DEV}" mode tap
sudo ip addr add 172.16.0.1/30 dev "${TAP_DEV}"
sudo ip link set "${TAP_DEV}" up

"${FC_BIN}" --api-sock "${API_SOCK}" >"${SERIAL_LOG}" 2>&1 &
FC_PID=$!

for _ in $(seq 1 100); do
  [[ -S "${API_SOCK}" ]] && break
  sleep 0.1
done
[[ -S "${API_SOCK}" ]] || fail "API socket did not appear: ${API_SOCK}"

put_fc /boot-source "{\"kernel_image_path\":\"${KERNEL}\",\"boot_args\":\"console=ttyS0 reboot=k panic=1 pci=off init=/sbin/init\"}"
put_fc /drives/rootfs "{\"drive_id\":\"rootfs\",\"path_on_host\":\"${ROOTFS_TEST}\",\"is_root_device\":true,\"is_read_only\":false}"
put_fc /machine-config '{"vcpu_count":1,"mem_size_mib":512}'
put_fc /network-interfaces/eth0 "{\"iface_id\":\"eth0\",\"guest_mac\":\"06:00:AC:10:00:02\",\"host_dev_name\":\"${TAP_DEV}\"}"
put_fc /vsock "{\"guest_cid\":52,\"uds_path\":\"${VSOCK_UDS}\"}"
put_fc /mmds/config '{"network_interfaces":["eth0"],"version":"V2"}'
put_fc /mmds '{"network":{"ip":"172.16.0.2","gateway":"172.16.0.1","netmask":"255.255.255.252","dns":["8.8.8.8"]},"llm":{"model":"test/model","api_key":"sk-test","max_tokens":128,"temperature":0.1}}'
put_fc /actions '{"action_type":"InstanceStart"}'

sleep 8

printf '=== entrypoint logs ===\n'
grep -E '\[entrypoint\]' "${SERIAL_LOG}" || true

printf '\n=== llm.json in guest rootfs ===\n'
debugfs -R "cat /run/strangeclaw/llm.json" "${ROOTFS_TEST}" 2>/dev/null || true

printf '\n=== raw serial tail ===\n'
tail -n 80 "${SERIAL_LOG}" || true
