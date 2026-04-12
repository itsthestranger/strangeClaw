#!/usr/bin/env bash
# Install and validate Fire mode prerequisites.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECK_ONLY=0
ENABLE_IP_FORWARDING_NOW=0
PERSIST_IP_FORWARDING=0

declare -a STEP_NAMES=()
declare -a STEP_STATUS=()
declare -a STEP_DETAILS=()

STEP_FAILS=0
STEP_WARNS=0

usage() {
  cat <<'EOF'
Usage: scripts/setup-fire.sh [--check-only] [--enable-ip-forwarding-now] [--persist-ip-forwarding]

Options:
  --check-only               Do not install or change host state; only run checks.
  --enable-ip-forwarding-now Enable net.ipv4.ip_forward=1 for the current runtime.
  --persist-ip-forwarding    Persist net.ipv4.ip_forward=1 in /etc/sysctl.d.
                             Implies --enable-ip-forwarding-now.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --enable-ip-forwarding-now)
      ENABLE_IP_FORWARDING_NOW=1
      shift
      ;;
    --persist-ip-forwarding)
      ENABLE_IP_FORWARDING_NOW=1
      PERSIST_IP_FORWARDING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

add_step() {
  local name="$1"
  local status="$2"
  local details="$3"
  STEP_NAMES+=("${name}")
  STEP_STATUS+=("${status}")
  STEP_DETAILS+=("${details}")
  if [[ "${status}" == "FAIL" ]]; then
    STEP_FAILS=$((STEP_FAILS + 1))
  elif [[ "${status}" == "WARN" ]]; then
    STEP_WARNS=$((STEP_WARNS + 1))
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    if ! command_exists sudo; then
      return 1
    fi
    sudo "$@"
  fi
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

check_supported_arch() {
  local arch
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|aarch64)
      add_step "arch" "PASS" "Supported architecture: ${arch}."
      ;;
    *)
      add_step "arch" "FAIL" "Unsupported architecture: ${arch} (expected x86_64 or aarch64)."
      ;;
  esac
}

setup_os_check() {
  local distro_id
  local distro_version
  local pretty_name
  local pkg_manager
  distro_id="$(read_os_release_value ID)"
  distro_version="$(read_os_release_value VERSION_ID)"
  pretty_name="$(read_os_release_value PRETTY_NAME)"
  pkg_manager="$(detect_package_manager)"

  if [[ -z "${distro_id}" ]]; then
    add_step "os_check" "WARN" "Could not detect distro from /etc/os-release."
    return
  fi
  if [[ -z "${pkg_manager}" ]]; then
    add_step "os_check" "WARN" "Detected ${pretty_name:-${distro_id}}, but apt/dnf/pacman was not found."
    return
  fi
  if [[ -n "${distro_version}" ]]; then
    add_step "os_check" "PASS" "Detected ${pretty_name:-${distro_id}} (package manager: ${pkg_manager})."
    return
  fi
  add_step "os_check" "PASS" "Detected ${distro_id} (package manager: ${pkg_manager})."
}

setup_packages() {
  local pkg_manager
  pkg_manager="$(detect_package_manager)"

  case "${pkg_manager}" in
    apt)
      if run_as_root apt-get update >/dev/null 2>&1 && \
        run_as_root apt-get install -y acl curl iproute2 iptables ca-certificates kmod >/dev/null 2>&1; then
        add_step "packages" "PASS" "Installed via apt: acl curl iproute2 iptables ca-certificates kmod."
      else
        add_step "packages" "FAIL" "Failed to install prerequisites with apt."
      fi
      ;;
    dnf)
      if run_as_root dnf -y install acl curl iproute iptables ca-certificates kmod >/dev/null 2>&1; then
        add_step "packages" "PASS" "Installed via dnf: acl curl iproute iptables ca-certificates kmod."
      else
        add_step "packages" "FAIL" "Failed to install prerequisites with dnf."
      fi
      ;;
    pacman)
      if run_as_root pacman -Sy --noconfirm --needed acl curl iproute2 iptables ca-certificates kmod >/dev/null 2>&1; then
        add_step "packages" "PASS" "Installed via pacman: acl curl iproute2 iptables ca-certificates kmod."
      else
        add_step "packages" "FAIL" "Failed to install prerequisites with pacman."
      fi
      ;;
    *)
      add_step "packages" "WARN" "No supported package manager found (expected apt, dnf, or pacman)."
      ;;
  esac
}

setup_iptables_backend_check() {
  local version_output

  if ! command_exists iptables; then
    add_step "iptables_backend" "FAIL" "iptables command not found after package setup."
    return
  fi

  version_output="$(iptables -V 2>/dev/null || true)"
  if [[ "${version_output}" == *"nf_tables"* ]]; then
    add_step "iptables_backend" "PASS" "iptables is using nf_tables backend."
    return
  fi
  if command_exists iptables-nft; then
    add_step "iptables_backend" "WARN" "iptables default backend may be legacy; iptables-nft is available."
    return
  fi
  add_step "iptables_backend" "WARN" "Could not confirm nft backend for iptables."
}

setup_kvm_access() {
  local target_user
  target_user="${SUDO_USER:-${USER}}"

  if [[ ! -e /dev/kvm ]]; then
    add_step "kvm_access" "FAIL" "/dev/kvm not found."
    return
  fi
  if [[ -r /dev/kvm && -w /dev/kvm ]]; then
    add_step "kvm_access" "PASS" "Read/write access to /dev/kvm already available."
    return
  fi
  if ! command_exists setfacl; then
    add_step "kvm_access" "FAIL" "setfacl not found; cannot grant /dev/kvm access."
    return
  fi
  if run_as_root setfacl -m "u:${target_user}:rw" /dev/kvm >/dev/null 2>&1 && [[ -r /dev/kvm && -w /dev/kvm ]]; then
    add_step "kvm_access" "PASS" "Granted /dev/kvm read/write access to ${target_user}."
  else
    add_step "kvm_access" "FAIL" "Could not grant /dev/kvm read/write access."
  fi
}

firecracker_archive_url() {
  local version="$1"
  local arch
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|aarch64)
      ;;
    *)
      echo ""
      return
      ;;
  esac
  echo "https://github.com/firecracker-microvm/firecracker/releases/download/${version}/firecracker-${version}-${arch}.tgz"
}

setup_firecracker_binary() {
  local pinned_version
  local version_output
  local url
  local tmp_dir
  local arch
  local expected_binary
  local checksum_url
  local checksum_file
  local expected_sha
  local actual_sha

  pinned_version="$(tr -d '[:space:]' < "${PROJECT_ROOT}/firecracker/VERSION" 2>/dev/null || true)"
  if [[ -z "${pinned_version}" ]]; then
    add_step "firecracker" "FAIL" "Missing pinned version file at firecracker/VERSION."
    return
  fi

  if [[ -x /usr/local/bin/firecracker ]]; then
    version_output="$(/usr/local/bin/firecracker --version 2>&1 | head -n 1 || true)"
    if [[ "${version_output}" == *"${pinned_version}"* ]]; then
      add_step "firecracker" "PASS" "Pinned Firecracker ${pinned_version} already installed."
      return
    fi
  fi

  if ! command_exists curl || ! command_exists tar; then
    add_step "firecracker" "FAIL" "curl and tar are required to install Firecracker."
    return
  fi

  url="$(firecracker_archive_url "${pinned_version}")"
  if [[ -z "${url}" ]]; then
    add_step "firecracker" "FAIL" "Unsupported architecture: $(uname -m)."
    return
  fi

  arch="$(uname -m)"
  tmp_dir="$(mktemp -d)"
  if ! curl -fsSL --retry 3 "${url}" -o "${tmp_dir}/firecracker.tgz"; then
    rm -rf "${tmp_dir}"
    add_step "firecracker" "FAIL" "Failed to download Firecracker ${pinned_version}."
    return
  fi

  checksum_url="${url}.sha256.txt"
  checksum_file="${tmp_dir}/firecracker.tgz.sha256.txt"
  if command_exists sha256sum; then
    if curl -fsSL --retry 2 "${checksum_url}" -o "${checksum_file}"; then
      expected_sha="$(grep -E "[A-Fa-f0-9]{64}.*firecracker-${pinned_version}-${arch}\.tgz" "${checksum_file}" | head -n 1 | sed -E 's/^.*([A-Fa-f0-9]{64}).*$/\1/')"
      if [[ -n "${expected_sha}" ]]; then
        actual_sha="$(sha256sum "${tmp_dir}/firecracker.tgz" | awk '{print $1}')"
        if [[ "${actual_sha}" == "${expected_sha}" ]]; then
          add_step "firecracker_checksum" "PASS" "Archive checksum verified."
        else
          rm -rf "${tmp_dir}"
          add_step "firecracker_checksum" "FAIL" "Archive checksum mismatch."
          return
        fi
      else
        add_step "firecracker_checksum" "WARN" "Downloaded checksum file but could not parse expected hash."
      fi
    else
      add_step "firecracker_checksum" "WARN" "Could not download release checksum file; verification skipped."
    fi
  else
    add_step "firecracker_checksum" "WARN" "sha256sum not found; archive checksum not verified."
  fi

  if ! tar -xzf "${tmp_dir}/firecracker.tgz" -C "${tmp_dir}"; then
    rm -rf "${tmp_dir}"
    add_step "firecracker" "FAIL" "Failed to extract Firecracker archive."
    return
  fi

  expected_binary="${tmp_dir}/release-${pinned_version}-${arch}/firecracker-${pinned_version}-${arch}"
  if [[ ! -f "${expected_binary}" ]]; then
    expected_binary="$(find "${tmp_dir}" -type f -name "firecracker-${pinned_version}-${arch}" | head -n 1)"
  fi
  if [[ -z "${expected_binary}" || ! -f "${expected_binary}" ]]; then
    rm -rf "${tmp_dir}"
    add_step "firecracker" "FAIL" "Firecracker binary not found in downloaded archive."
    return
  fi

  if run_as_root install -m 0755 "${expected_binary}" /usr/local/bin/firecracker >/dev/null 2>&1; then
    add_step "firecracker" "PASS" "Installed Firecracker ${pinned_version} to /usr/local/bin/firecracker."
  else
    add_step "firecracker" "FAIL" "Failed to install Firecracker to /usr/local/bin/firecracker."
  fi
  rm -rf "${tmp_dir}"
}

setup_ip_forwarding() {
  local current_value
  if [[ ! -f /proc/sys/net/ipv4/ip_forward ]]; then
    add_step "ip_forward" "FAIL" "Cannot read net.ipv4.ip_forward."
    return
  fi

  current_value="$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null || true)"

  if [[ "${ENABLE_IP_FORWARDING_NOW}" -eq 0 ]]; then
    if [[ "${current_value}" == "1" ]]; then
      add_step "ip_forward" "PASS" "IPv4 forwarding already enabled; left unchanged (runtime-managed policy)."
    else
      add_step "ip_forward" "WARN" "IPv4 forwarding is disabled; left unchanged (runtime-managed policy)."
    fi
    return
  fi

  if ! run_as_root sysctl -w net.ipv4.ip_forward=1 >/dev/null 2>&1; then
    add_step "ip_forward" "FAIL" "Failed to enable IPv4 forwarding."
    return
  fi

  if [[ "${PERSIST_IP_FORWARDING}" -eq 1 ]]; then
    if printf 'net.ipv4.ip_forward = 1\n' | run_as_root tee /etc/sysctl.d/99-strangeclaw-fire.conf >/dev/null; then
      add_step "ip_forward" "PASS" "Enabled IPv4 forwarding and persisted it in /etc/sysctl.d."
    else
      add_step "ip_forward" "FAIL" "Enabled IPv4 forwarding but failed to persist sysctl config."
    fi
    return
  fi

  add_step "ip_forward" "PASS" "Enabled IPv4 forwarding for current runtime (not persisted)."
}

setup_tun_module() {
  if run_as_root modprobe tun >/dev/null 2>&1; then
    add_step "tun_module" "PASS" "Loaded tun module."
  else
    add_step "tun_module" "FAIL" "Failed to load tun module."
  fi
}

setup_container_runtime_check() {
  if command_exists docker; then
    add_step "container_runtime" "PASS" "Found docker."
    return
  fi
  if command_exists podman; then
    add_step "container_runtime" "PASS" "Found podman."
    return
  fi
  add_step "container_runtime" "WARN" "Docker/Podman not found (needed for rootfs build)."
}

print_step_report() {
  local name_width=4
  local i
  for i in "${!STEP_NAMES[@]}"; do
    if (( ${#STEP_NAMES[$i]} > name_width )); then
      name_width=${#STEP_NAMES[$i]}
    fi
  done

  printf "%-${name_width}s  %-6s  %s\n" "step" "status" "details"
  printf "%-${name_width}s  %-6s  %s\n" "$(printf '%*s' "${name_width}" '' | tr ' ' '-')" "------" "----------------------------------------"

  for i in "${!STEP_NAMES[@]}"; do
    printf "%-${name_width}s  %-6s  %s\n" \
      "${STEP_NAMES[$i]}" \
      "${STEP_STATUS[$i]}" \
      "${STEP_DETAILS[$i]}"
  done

  local pass_count
  pass_count=$(( ${#STEP_NAMES[@]} - STEP_FAILS - STEP_WARNS ))
  printf "\nSetup summary: PASS=%d WARN=%d FAIL=%d\n" "${pass_count}" "${STEP_WARNS}" "${STEP_FAILS}"
}

run_setup() {
  check_supported_arch
  setup_os_check
  if [[ "${CHECK_ONLY}" -eq 0 ]]; then
    setup_packages
    setup_iptables_backend_check
    setup_kvm_access
    setup_firecracker_binary
    setup_ip_forwarding
    setup_tun_module
    setup_container_runtime_check
  else
    add_step "setup" "WARN" "Skipped setup changes (--check-only)."
  fi
}

run_setup
print_step_report
echo
echo "Post-setup prerequisite report:"

if bash "${SCRIPT_DIR}/fire-check.sh"; then
  exit 0
fi
exit 1
