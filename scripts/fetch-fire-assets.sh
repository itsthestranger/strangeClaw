#!/usr/bin/env bash
# Download Firecracker CI kernel/rootfs artifacts and verify vsock readiness.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION_FILE="${PROJECT_ROOT}/firecracker/VERSION"
KERNEL_DIR="${PROJECT_ROOT}/firecracker/kernel"
ROOTFS_DIR="${PROJECT_ROOT}/firecracker/rootfs"

FIRECRACKER_BIN="/usr/local/bin/firecracker"
DOWNLOAD_ONLY=0

TMP_DIR=""
FC_PID=""

usage() {
  cat <<'USAGE'
Usage: scripts/fetch-fire-assets.sh [--download-only] [--firecracker-binary <path>]

Downloads Firecracker CI kernel + Ubuntu rootfs artifacts for the release line
pinned in firecracker/VERSION, generates firecracker/rootfs/agent.ext4, verifies
vsock kernel config, and runs smoke/vsock runtime checks.

Options:
  --download-only            Skip smoke and vsock runtime checks.
  --firecracker-binary PATH Firecracker binary path. Default: /usr/local/bin/firecracker
  -h, --help                Show this help.
USAGE
}

log() {
  printf '[fetch-fire-assets] %s\n' "$*"
}

warn() {
  printf '[fetch-fire-assets] WARN: %s\n' "$*" >&2
}

fail() {
  printf '[fetch-fire-assets] ERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ -n "${FC_PID}" ]] && kill -0 "${FC_PID}" >/dev/null 2>&1; then
    kill "${FC_PID}" >/dev/null 2>&1 || true
    wait "${FC_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}" || true
  fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --firecracker-binary)
      [[ $# -ge 2 ]] || fail "Missing value for --firecracker-binary"
      FIRECRACKER_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

require_commands() {
  local required=(curl sed grep awk sort mktemp basename dirname cp rm mv unsquashfs truncate mkfs.ext4 e2fsck debugfs python3)
  local missing=0
  local cmd

  for cmd in "${required[@]}"; do
    if ! command_exists "${cmd}"; then
      warn "Missing command: ${cmd}"
      missing=1
    fi
  done

  if [[ "${DOWNLOAD_ONLY}" -eq 0 ]] && [[ ! -x "${FIRECRACKER_BIN}" ]]; then
    warn "Firecracker binary not found/executable: ${FIRECRACKER_BIN}"
    missing=1
  fi

  if [[ "${missing}" -ne 0 ]]; then
    fail "Install missing prerequisites and re-run."
  fi

  if ! printf 'x\n' | grep -qP 'x'; then
    fail "This script requires grep with PCRE support (-P)."
  fi
}

read_pinned_version() {
  [[ -f "${VERSION_FILE}" ]] || fail "Missing version file: ${VERSION_FILE}"
  local version
  version="$(tr -d '[:space:]' < "${VERSION_FILE}")"
  [[ -n "${version}" ]] || fail "firecracker/VERSION is empty"
  printf '%s' "${version}"
}

detect_arch() {
  local arch
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|aarch64)
      printf '%s' "${arch}"
      ;;
    *)
      fail "Unsupported architecture: ${arch}. Expected x86_64 or aarch64."
      ;;
  esac
}

fetch_ci_listing() {
  local ci_version="$1"
  local arch="$2"
  local prefix_suffix="${3:-}"
  local url
  # Firecracker docs use HTTP for the XML listing endpoint.
  url="http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/${ci_version}/${arch}/${prefix_suffix}&list-type=2"

  curl -fsSL --retry 3 "${url}"
}

extract_latest_key_pcre() {
  local listing="$1"
  local ci_version="$2"
  local arch="$3"
  local pcre_pattern="$4"

  printf '%s\n' "${listing}" \
    | grep -oP "(?<=<Key>)(firecracker-ci/${ci_version}/${arch}/${pcre_pattern})(?=</Key>)" \
    | sort -V | tail -n 1
}

download_artifact() {
  local key="$1"
  local target="$2"
  local url="https://s3.amazonaws.com/spec.ccfc.min/${key}"

  mkdir -p "$(dirname "${target}")"
  curl -fsSL --retry 3 "${url}" -o "${target}" || fail "Failed to download ${url}"
}

build_ext4_from_squashfs() {
  local squashfs_path="$1"
  local target_ext4="$2"
  local extract_dir="${TMP_DIR}/rootfs-extract"

  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"

  log "Extracting rootfs squashfs"
  unsquashfs -f -d "${extract_dir}" "${squashfs_path}" >/dev/null || fail "unsquashfs failed"

  local bytes
  bytes="$(du -sb "${extract_dir}" | awk '{print $1}')"
  [[ -n "${bytes}" ]] || fail "Could not measure extracted rootfs size"

  local mib
  # +20% and +256 MiB headroom, minimum 1024 MiB
  mib=$(( (bytes + (bytes / 5) + 268435456 + 1048575) / 1048576 ))
  if (( mib < 1024 )); then
    mib=1024
  fi

  local tmp_ext4
  tmp_ext4="${target_ext4}.tmp"
  rm -f "${tmp_ext4}"

  log "Building ext4 rootfs (${mib} MiB)"
  truncate -s "${mib}M" "${tmp_ext4}" || fail "truncate failed for ext4 image"
  mkfs.ext4 -q -F -d "${extract_dir}" "${tmp_ext4}" || fail "mkfs.ext4 failed"
  e2fsck -fn "${tmp_ext4}" >/dev/null || fail "e2fsck validation failed"

  mv -f "${tmp_ext4}" "${target_ext4}" || fail "Failed to install ${target_ext4}"
}

verify_kernel_vsock_config() {
  local config_path="${KERNEL_DIR}/kernel.config"

  if [[ ! -f "${config_path}" ]]; then
    warn "kernel.config not found; runtime /dev/vsock test will be used as authoritative check."
    return
  fi

  if grep -q '^CONFIG_VIRTIO_VSOCKETS=y$' "${config_path}"; then
    log "Kernel config check passed: CONFIG_VIRTIO_VSOCKETS=y"
    return
  fi

  fail "Kernel config missing CONFIG_VIRTIO_VSOCKETS=y. Use scripts/build-fire-kernel.sh fallback."
}

wait_for_path() {
  local path="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if [[ -e "${path}" ]]; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

fc_put() {
  local api_sock="$1"
  local path="$2"
  local payload="$3"
  local out_file="${TMP_DIR}/fc_put.out"

  local code
  code="$(curl -sS --unix-socket "${api_sock}" \
    -X PUT \
    -H 'Accept: application/json' \
    -H 'Content-Type: application/json' \
    -d "${payload}" \
    -o "${out_file}" \
    -w '%{http_code}' \
    "http://localhost${path}")" || fail "Firecracker API call failed: PUT ${path}"

  case "${code}" in
    2*)
      return 0
      ;;
    *)
      fail "Firecracker API PUT ${path} failed (HTTP ${code}): $(cat "${out_file}" 2>/dev/null || true)"
      ;;
  esac
}

inject_test_payload() {
  local ext4_path="$1"
  local init_path="${TMP_DIR}/strangeclaw-smoke-init.sh"
  local listener_c="${TMP_DIR}/vsock-listener.c"
  local listener_bin="${TMP_DIR}/strangeclaw-vsock-listener"

  cat > "${listener_c}" <<'C_EOF'
#include <errno.h>
#include <linux/vm_sockets.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
    int fd;
    struct sockaddr_vm addr;
    struct timeval tv;

    if (access("/dev/vsock", F_OK) == 0) {
        puts("STRANGECLAW_DEV_VSOCK_PRESENT");
    } else {
        puts("STRANGECLAW_DEV_VSOCK_MISSING");
        return 10;
    }
    fflush(stdout);

    fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) {
        puts("STRANGECLAW_VSOCK_SOCKET_FAILED");
        return 11;
    }

    memset(&addr, 0, sizeof(addr));
    addr.svm_family = AF_VSOCK;
    addr.svm_cid = VMADDR_CID_ANY;
    addr.svm_port = 5000;

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        puts("STRANGECLAW_VSOCK_BIND_FAILED");
        close(fd);
        return 12;
    }

    if (listen(fd, 1) < 0) {
        puts("STRANGECLAW_VSOCK_LISTEN_FAILED");
        close(fd);
        return 13;
    }

    tv.tv_sec = 20;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    puts("STRANGECLAW_VSOCK_LISTENING");
    fflush(stdout);

    int client = accept(fd, NULL, NULL);
    if (client < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            puts("STRANGECLAW_VSOCK_ACCEPT_TIMEOUT");
        } else {
            puts("STRANGECLAW_VSOCK_ACCEPT_FAILED");
        }
        close(fd);
        return 14;
    }

    puts("STRANGECLAW_VSOCK_ACCEPTED");
    fflush(stdout);
    (void)write(client, "guest-ack\n", 10);
    close(client);
    close(fd);
    return 0;
}
C_EOF

  if ! cc -O2 -static -s -o "${listener_bin}" "${listener_c}" >/dev/null 2>&1; then
    cc -O2 -o "${listener_bin}" "${listener_c}" >/dev/null 2>&1 || fail "Failed to compile vsock listener helper"
  fi

  cat > "${init_path}" <<'INIT_EOF'
#!/bin/sh
set -eu

mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true

exec >/dev/ttyS0 2>&1

echo "STRANGECLAW_SMOKE_BOOT_OK"
/strangeclaw-vsock-listener || true
sync
reboot -f || poweroff -f || halt -f || exit 0
INIT_EOF

  chmod 0755 "${listener_bin}" "${init_path}"

  debugfs -w -R "write ${listener_bin} /strangeclaw-vsock-listener" "${ext4_path}" >/dev/null || \
    fail "Failed to write /strangeclaw-vsock-listener into rootfs"
  debugfs -w -R "write ${init_path} /strangeclaw-smoke-init.sh" "${ext4_path}" >/dev/null || \
    fail "Failed to write /strangeclaw-smoke-init.sh into rootfs"

  debugfs -w -R "set_inode_field /strangeclaw-vsock-listener mode 0100755" "${ext4_path}" >/dev/null || \
    fail "Failed to set executable mode on /strangeclaw-vsock-listener"
  debugfs -w -R "set_inode_field /strangeclaw-smoke-init.sh mode 0100755" "${ext4_path}" >/dev/null || \
    fail "Failed to set executable mode on /strangeclaw-smoke-init.sh"
}

wait_for_log_marker() {
  local marker="$1"
  local logfile="$2"
  local timeout="$3"
  local deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if grep -q "${marker}" "${logfile}" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done

  return 1
}

host_connect_test() {
  local uds_path="$1"

  python3 - "$uds_path" <<'PY'
import socket
import sys
import time

uds_path = sys.argv[1]
deadline = time.time() + 20.0

while time.time() < deadline:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      s.settimeout(2.0)
      s.connect(uds_path)
      s.sendall(b"CONNECT 5000\n")
      reply = s.recv(128)
      if not reply.startswith(b"OK "):
          s.close()
          time.sleep(0.5)
          continue
      try:
          ack = s.recv(128)
      except socket.timeout:
          ack = b""
      if ack.startswith(b"guest-ack"):
          print("ok")
          sys.exit(0)
    except OSError:
      pass
    finally:
      s.close()
    time.sleep(0.5)

print("failed")
sys.exit(1)
PY
}

run_smoke_and_vsock_checks() {
  local kernel_path="${KERNEL_DIR}/vmlinux"
  local rootfs_path="${ROOTFS_DIR}/agent.ext4"

  [[ -f "${kernel_path}" ]] || fail "Missing kernel image: ${kernel_path}"
  [[ -f "${rootfs_path}" ]] || fail "Missing rootfs image: ${rootfs_path}"

  local test_rootfs="${TMP_DIR}/agent-vsock-check.ext4"
  cp -f "${rootfs_path}" "${test_rootfs}" || fail "Failed to create test rootfs copy"
  inject_test_payload "${test_rootfs}"

  local api_sock="${TMP_DIR}/firecracker.socket"
  local vsock_uds="${TMP_DIR}/firecracker.vsock"
  local serial_log="${TMP_DIR}/serial.log"

  rm -f "${api_sock}" "${vsock_uds}" "${serial_log}"

  "${FIRECRACKER_BIN}" --api-sock "${api_sock}" >"${serial_log}" 2>&1 &
  FC_PID="$!"

  wait_for_path "${api_sock}" 5 || fail "Firecracker API socket did not appear"

  fc_put "${api_sock}" "/boot-source" "{\"kernel_image_path\":\"${kernel_path}\",\"boot_args\":\"console=ttyS0 reboot=k panic=1 pci=off init=/strangeclaw-smoke-init.sh\"}"
  fc_put "${api_sock}" "/drives/rootfs" "{\"drive_id\":\"rootfs\",\"path_on_host\":\"${test_rootfs}\",\"is_root_device\":true,\"is_read_only\":false}"
  fc_put "${api_sock}" "/machine-config" '{"vcpu_count":1,"mem_size_mib":512}'
  fc_put "${api_sock}" "/vsock" "{\"guest_cid\":1234,\"uds_path\":\"${vsock_uds}\"}"
  fc_put "${api_sock}" "/actions" '{"action_type":"InstanceStart"}'

  wait_for_path "${vsock_uds}" 20 || fail "Vsock UDS path was not created"

  wait_for_log_marker "STRANGECLAW_SMOKE_BOOT_OK" "${serial_log}" 30 || fail "Smoke boot marker missing"
  wait_for_log_marker "STRANGECLAW_DEV_VSOCK_PRESENT" "${serial_log}" 30 || fail "Guest did not report /dev/vsock"
  wait_for_log_marker "STRANGECLAW_VSOCK_LISTENING" "${serial_log}" 30 || fail "Guest vsock listener did not start"

  host_connect_test "${vsock_uds}" || fail "Host CONNECT handshake to guest vsock listener failed"

  wait_for_log_marker "STRANGECLAW_VSOCK_ACCEPTED" "${serial_log}" 15 || fail "Guest did not accept host vsock connection"

  wait "${FC_PID}" >/dev/null 2>&1 || fail "Firecracker exited with error during smoke test"
  FC_PID=""

  log "Smoke test passed."
  log "Vsock test passed (host CONNECT handshake + guest /dev/vsock)."
}

download_assets() {
  local ci_version="$1"
  local arch="$2"
  local listing
  listing="$(fetch_ci_listing "${ci_version}" "${arch}" "vmlinux-")" || \
    fail "Failed to fetch kernel listing: http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/${ci_version}/${arch}/vmlinux-&list-type=2"
  local kernel_key
  kernel_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'vmlinux-[0-9]+\.[0-9]+\.[0-9]{1,3}')"
  if [[ -z "${kernel_key}" ]]; then
    kernel_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'vmlinux-[0-9][0-9.]*')"
  fi

  listing="$(fetch_ci_listing "${ci_version}" "${arch}" "ubuntu-")" || \
    fail "Failed to fetch rootfs listing: http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/${ci_version}/${arch}/ubuntu-&list-type=2"
  local rootfs_key
  # Firecracker docs pattern (preferred):
  rootfs_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'ubuntu-[0-9]+\.[0-9]+\.squashfs')"
  # Fallbacks for naming drift:
  if [[ -z "${rootfs_key}" ]]; then
    rootfs_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'ubuntu-.*\.squashfs')"
  fi
  if [[ -z "${rootfs_key}" ]]; then
    rootfs_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'ubuntu-.*\.ext4')"
  fi

  listing="$(fetch_ci_listing "${ci_version}" "${arch}" "config-" || true)"
  local config_key
  config_key=""
  if [[ -n "${listing}" ]]; then
    config_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'config-[0-9]+\.[0-9]+\.[0-9]{1,3}')"
    if [[ -z "${config_key}" ]]; then
      config_key="$(extract_latest_key_pcre "${listing}" "${ci_version}" "${arch}" 'config-[0-9][0-9.]*')"
    fi
  fi

  [[ -n "${kernel_key}" ]] || fail "No vmlinux-* artifact found for ${ci_version}/${arch}"
  [[ -n "${rootfs_key}" ]] || fail "No ubuntu rootfs artifact (*.squashfs or *.ext4) found for ${ci_version}/${arch}"

  mkdir -p "${KERNEL_DIR}" "${ROOTFS_DIR}"

  local kernel_name
  kernel_name="$(basename "${kernel_key}")"
  local rootfs_name
  rootfs_name="$(basename "${rootfs_key}")"

  log "Downloading kernel artifact ${kernel_name}"
  download_artifact "${kernel_key}" "${KERNEL_DIR}/${kernel_name}"
  cp -f "${KERNEL_DIR}/${kernel_name}" "${KERNEL_DIR}/vmlinux"

  log "Downloading rootfs artifact ${rootfs_name}"
  download_artifact "${rootfs_key}" "${ROOTFS_DIR}/${rootfs_name}.upstream"

  if [[ -n "${config_key}" ]]; then
    local config_name
    config_name="$(basename "${config_key}")"
    log "Downloading kernel config ${config_name}"
    download_artifact "${config_key}" "${KERNEL_DIR}/${config_name}"
    cp -f "${KERNEL_DIR}/${config_name}" "${KERNEL_DIR}/kernel.config"
  else
    warn "No config-* artifact found; kernel config pre-check will be skipped."
    rm -f "${KERNEL_DIR}/kernel.config"
  fi

  if [[ "${rootfs_name}" == *.squashfs ]]; then
    build_ext4_from_squashfs "${ROOTFS_DIR}/${rootfs_name}.upstream" "${ROOTFS_DIR}/agent.ext4"
  elif [[ "${rootfs_name}" == *.ext4 ]]; then
    cp -f "${ROOTFS_DIR}/${rootfs_name}.upstream" "${ROOTFS_DIR}/agent.ext4" || \
      fail "Failed to install ${ROOTFS_DIR}/agent.ext4 from ext4 artifact"
    e2fsck -fn "${ROOTFS_DIR}/agent.ext4" >/dev/null || fail "Downloaded ext4 rootfs failed filesystem check"
  else
    fail "Unsupported rootfs artifact type: ${rootfs_name}"
  fi

  log "Asset update complete:"
  log "  ${KERNEL_DIR}/vmlinux"
  log "  ${ROOTFS_DIR}/agent.ext4"
}

main() {
  require_commands

  TMP_DIR="$(mktemp -d)"

  local version
  version="$(read_pinned_version)"
  local ci_version
  ci_version="${version%.*}"
  local arch
  arch="$(detect_arch)"

  [[ -n "${ci_version}" ]] || fail "Could not derive CI version from ${version}"

  log "Pinned Firecracker version: ${version}"
  log "CI artifact line: ${ci_version}"
  log "Host architecture: ${arch}"

  download_assets "${ci_version}" "${arch}"
  verify_kernel_vsock_config

  if [[ "${DOWNLOAD_ONLY}" -eq 1 ]]; then
    log "Download-only mode enabled; skipped runtime smoke/vsock checks."
    exit 0
  fi

  run_smoke_and_vsock_checks
  log "Done."
}

main "$@"
