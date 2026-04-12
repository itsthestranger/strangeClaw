#!/usr/bin/env bash
# Validate Fire mode host prerequisites.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a CHECK_NAMES=()
declare -a CHECK_STATUS=()
declare -a CHECK_DETAILS=()

FAIL_COUNT=0
WARN_COUNT=0

add_check() {
  local name="$1"
  local status="$2"
  local details="$3"
  CHECK_NAMES+=("${name}")
  CHECK_STATUS+=("${status}")
  CHECK_DETAILS+=("${details}")
  if [[ "${status}" == "FAIL" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
  elif [[ "${status}" == "WARN" ]]; then
    WARN_COUNT=$((WARN_COUNT + 1))
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

read_os_release_value() {
  local key="$1"
  if [[ ! -f /etc/os-release ]]; then
    echo ""
    return
  fi
  awk -F= -v k="${key}" '$1==k {gsub(/^"|"$/, "", $2); print $2}' /etc/os-release 2>/dev/null
}

detect_package_manager() {
  if command_exists apt-get; then
    echo "apt"
    return
  fi
  if command_exists dnf; then
    echo "dnf"
    return
  fi
  if command_exists pacman; then
    echo "pacman"
    return
  fi
  echo ""
}

check_host_os() {
  local distro_id
  local distro_version
  local pretty_name
  local pkg_manager
  distro_id="$(read_os_release_value ID)"
  distro_version="$(read_os_release_value VERSION_ID)"
  pretty_name="$(read_os_release_value PRETTY_NAME)"
  pkg_manager="$(detect_package_manager)"

  if [[ -z "${distro_id}" ]]; then
    add_check "host_os" "WARN" "Could not determine distro from /etc/os-release."
    return
  fi
  if [[ -z "${pkg_manager}" ]]; then
    add_check "host_os" "WARN" "Detected ${pretty_name:-${distro_id}}, but apt/dnf/pacman was not found."
    return
  fi

  if [[ -n "${distro_version}" ]]; then
    add_check "host_os" "PASS" "Detected ${pretty_name:-${distro_id}} (package manager: ${pkg_manager})."
    return
  fi
  add_check "host_os" "PASS" "Detected ${distro_id} (package manager: ${pkg_manager})."
}

check_architecture() {
  local arch
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|aarch64)
      add_check "arch" "PASS" "Supported architecture: ${arch}."
      ;;
    *)
      add_check "arch" "FAIL" "Unsupported architecture: ${arch} (expected x86_64 or aarch64)."
      ;;
  esac
}

check_kvm() {
  if [[ ! -e /dev/kvm ]]; then
    add_check "kvm_device" "FAIL" "/dev/kvm not found."
    add_check "kvm_access" "FAIL" "Cannot check access without /dev/kvm."
  else
    add_check "kvm_device" "PASS" "/dev/kvm is present."
    if [[ -r /dev/kvm && -w /dev/kvm ]]; then
      add_check "kvm_access" "PASS" "Read/write access is available."
    else
      add_check "kvm_access" "FAIL" "Read/write access missing."
    fi
  fi

  if [[ -d /sys/module/kvm ]]; then
    add_check "kvm_module" "PASS" "KVM module is loaded."
  else
    add_check "kvm_module" "FAIL" "KVM module is not loaded."
  fi
}

check_required_commands() {
  if command_exists setfacl; then
    add_check "setfacl" "PASS" "Found setfacl."
  else
    add_check "setfacl" "FAIL" "Missing setfacl (acl package)."
  fi

  if command_exists curl; then
    add_check "curl" "PASS" "Found curl."
  else
    add_check "curl" "FAIL" "Missing curl."
  fi

  if command_exists ip; then
    add_check "iproute2" "PASS" "Found ip (iproute2)."
  else
    add_check "iproute2" "FAIL" "Missing ip (iproute2)."
  fi

  if command_exists iptables; then
    add_check "iptables" "PASS" "Found iptables."
  else
    add_check "iptables" "FAIL" "Missing iptables."
  fi
}

check_ip_forwarding() {
  if [[ ! -f /proc/sys/net/ipv4/ip_forward ]]; then
    add_check "ip_forward" "FAIL" "Cannot read net.ipv4.ip_forward."
    return
  fi
  if [[ "$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null)" == "1" ]]; then
    add_check "ip_forward" "PASS" "IPv4 forwarding is enabled."
  else
    add_check "ip_forward" "FAIL" "IPv4 forwarding is disabled."
  fi
}

check_tun() {
  if [[ -d /sys/module/tun && -c /dev/net/tun ]]; then
    add_check "tun" "PASS" "tun module and /dev/net/tun are present."
  else
    add_check "tun" "FAIL" "tun module or /dev/net/tun is missing."
  fi
}

check_firecracker_binary() {
  local pinned_version
  local version_output
  pinned_version="$(tr -d '[:space:]' < "${PROJECT_ROOT}/firecracker/VERSION" 2>/dev/null || true)"

  if [[ ! -x /usr/local/bin/firecracker ]]; then
    add_check "firecracker_binary" "FAIL" "Missing executable /usr/local/bin/firecracker."
    return
  fi

  add_check "firecracker_binary" "PASS" "Found /usr/local/bin/firecracker."
  version_output="$(/usr/local/bin/firecracker --version 2>&1 | head -n 1 || true)"
  if [[ -z "${version_output}" ]]; then
    add_check "firecracker_version" "FAIL" "Unable to run firecracker --version."
    return
  fi
  if [[ -z "${pinned_version}" ]]; then
    add_check "firecracker_version" "WARN" "Pinned version file is missing."
    return
  fi
  if [[ "${version_output}" == *"${pinned_version}"* ]]; then
    add_check "firecracker_version" "PASS" "Version matches pinned ${pinned_version}."
  else
    add_check "firecracker_version" "FAIL" "Expected ${pinned_version}, got: ${version_output}."
  fi
}

check_container_runtime() {
  if command_exists docker; then
    add_check "container_runtime" "PASS" "Found docker."
    return
  fi
  if command_exists podman; then
    add_check "container_runtime" "PASS" "Found podman."
    return
  fi
  add_check "container_runtime" "WARN" "Docker/Podman not found (needed for rootfs build)."
}

print_report() {
  local name_width=5
  local i
  for i in "${!CHECK_NAMES[@]}"; do
    if (( ${#CHECK_NAMES[$i]} > name_width )); then
      name_width=${#CHECK_NAMES[$i]}
    fi
  done

  printf "%-${name_width}s  %-6s  %s\n" "check" "status" "details"
  printf "%-${name_width}s  %-6s  %s\n" "$(printf '%*s' "${name_width}" '' | tr ' ' '-')" "------" "----------------------------------------"

  for i in "${!CHECK_NAMES[@]}"; do
    printf "%-${name_width}s  %-6s  %s\n" \
      "${CHECK_NAMES[$i]}" \
      "${CHECK_STATUS[$i]}" \
      "${CHECK_DETAILS[$i]}"
  done

  local pass_count
  pass_count=$(( ${#CHECK_NAMES[@]} - FAIL_COUNT - WARN_COUNT ))
  printf "\nSummary: PASS=%d WARN=%d FAIL=%d\n" "${pass_count}" "${WARN_COUNT}" "${FAIL_COUNT}"
}

run_checks() {
  check_host_os
  check_architecture
  check_kvm
  check_required_commands
  check_ip_forwarding
  check_tun
  check_firecracker_binary
  check_container_runtime
}

run_checks
print_report

if (( FAIL_COUNT > 0 )); then
  exit 1
fi
exit 0
