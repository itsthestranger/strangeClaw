#!/usr/bin/env bash
# Best-effort cleanup for orphaned strangeClaw Firecracker resources.

set -u
set -o pipefail

DRY_RUN=0
TEMP_ROOT="/tmp"

usage() {
  cat <<'EOF'
Usage: scripts/cleanup-fire.sh [--dry-run] [--temp-root PATH]

Best-effort recovery after abnormal Fire-mode termination. The script only
targets strangeClaw-owned resources:
  - Firecracker processes with --api-sock under a strangeclaw temp directory
  - TAP interfaces named fc + 12 lowercase hex chars
  - iptables rules tied to those TAP interfaces and their guest IPs
  - stale strangeclaw-* runtime directories/files under the temp root

Options:
  --dry-run         Print actions without changing host state.
  --temp-root PATH  Temp root to scan for strangeclaw-* runtime paths.
                    Default: /tmp
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --temp-root)
      [[ $# -ge 2 ]] || {
        echo "Missing value for --temp-root" >&2
        exit 2
      }
      TEMP_ROOT="$2"
      shift 2
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

log() {
  printf '%s\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: $*"
    return 0
  fi
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi
  if command_exists sudo; then
    sudo "$@"
    return
  fi
  warn "Need root privileges for: $*"
  return 1
}

capture_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi
  if command_exists sudo; then
    sudo "$@"
    return
  fi
  "$@"
}

is_strangeclaw_firecracker_cmdline() {
  local cmdline="$1"
  [[ "${cmdline}" =~ --api-sock[[:space:]]+[^[:space:]]*strangeclaw-[^[:space:]]*/firecracker\.socket ]]
}

cleanup_firecracker_processes() {
  log "Checking for orphaned strangeClaw Firecracker processes..."
  if ! command_exists pgrep; then
    warn "pgrep not found; skipping Firecracker process cleanup."
    return
  fi

  local found=0
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    local pid="${line%% *}"
    local cmdline="${line#* }"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    if ! is_strangeclaw_firecracker_cmdline "${cmdline}"; then
      continue
    fi

    found=1
    log "Stopping Firecracker process ${pid}: ${cmdline}"
    run_as_root kill -TERM "${pid}" || true
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      sleep 1
      if kill -0 "${pid}" >/dev/null 2>&1; then
        run_as_root kill -KILL "${pid}" || true
      fi
    else
      log "DRY-RUN: kill -KILL ${pid} if still running"
    fi
  done < <(pgrep -af '[f]irecracker' 2>/dev/null || true)

  if [[ "${found}" -eq 0 ]]; then
    log "No strangeClaw Firecracker processes found."
  fi
}

list_fire_taps() {
  if ! command_exists ip; then
    return
  fi
  ip -o link show 2>/dev/null \
    | awk -F': ' '{print $2}' \
    | cut -d@ -f1 \
    | grep -E '^fc[0-9a-f]{12}$' || true
}

is_fire_tap_name() {
  local value="$1"
  [[ "${value}" =~ ^fc[0-9a-f]{12}$ ]]
}

tap_ipv4() {
  local tap_name="$1"
  ip -4 -o addr show dev "${tap_name}" 2>/dev/null \
    | sed -nE 's/.* inet ([0-9.]+)\/.*/\1/p' \
    | head -n 1
}

guest_ip_from_tap_ip() {
  local tap_ip="$1"
  local a b c d
  IFS=. read -r a b c d <<<"${tap_ip}"
  if [[ -z "${a:-}" || -z "${b:-}" || -z "${c:-}" || -z "${d:-}" ]]; then
    return 1
  fi
  printf '%s.%s.%s.%s\n' "${a}" "${b}" "${c}" "$((d + 1))"
}

is_strangeclaw_guest_ip() {
  local value="$1"
  local ip="${value%/32}"
  local a b c d
  IFS=. read -r a b c d <<<"${ip}"
  if [[ "${a}" != "172" || "${b}" != "16" ]]; then
    return 1
  fi
  if ! [[ "${c}" =~ ^[0-9]+$ && "${d}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if (( c < 0 || c > 255 || d < 0 || d > 255 )); then
    return 1
  fi
  local offset=$((c * 256 + d))
  (( offset >= 2 && offset <= 65534 && offset % 4 == 2 ))
}

iptables_line_has_fire_tap() {
  local line="$1"
  local -a token
  read -r -a token <<<"${line}"
  local i
  for ((i = 0; i < ${#token[@]}; i++)); do
    if [[ "${token[$i]}" == "-i" && $((i + 1)) -lt ${#token[@]} ]]; then
      if is_fire_tap_name "${token[$((i + 1))]}"; then
        return 0
      fi
    fi
  done
  return 1
}

iptables_line_source_is_fire_guest() {
  local line="$1"
  local -a token
  read -r -a token <<<"${line}"
  local i
  for ((i = 0; i < ${#token[@]}; i++)); do
    if [[ "${token[$i]}" == "-s" && $((i + 1)) -lt ${#token[@]} ]]; then
      if is_strangeclaw_guest_ip "${token[$((i + 1))]}"; then
        return 0
      fi
    fi
  done
  return 1
}

delete_iptables_rule_line() {
  local table="$1"
  local line="$2"
  local -a parts
  read -r -a parts <<<"${line}"
  if [[ "${#parts[@]}" -eq 0 || "${parts[0]}" != "-A" ]]; then
    return
  fi
  parts[0]="-D"
  if [[ "${table}" == "nat" ]]; then
    run_as_root iptables -t nat "${parts[@]}" || true
  else
    run_as_root iptables "${parts[@]}" || true
  fi
}

cleanup_iptables_for_tap() {
  local tap_name="$1"
  local guest_ip="${2:-}"

  if ! command_exists iptables-save || ! command_exists iptables; then
    warn "iptables-save or iptables not found; skipping firewall cleanup for ${tap_name}."
    return
  fi

  log "Removing iptables rules for ${tap_name}${guest_ip:+ / ${guest_ip}}..."

  local line
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line}" == "-A INPUT "* && "${line}" == *"-i ${tap_name} "* && "${line}" == *" -j DROP"* ]]; then
      delete_iptables_rule_line "filter" "${line}"
      continue
    fi
    if [[ "${line}" == "-A FORWARD "* && "${line}" == *"-i ${tap_name} "* && "${line}" == *" -j ACCEPT"* ]]; then
      delete_iptables_rule_line "filter" "${line}"
      continue
    fi
  done < <(capture_as_root iptables-save -t filter 2>/dev/null || true)

  if [[ -n "${guest_ip}" ]]; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      if [[ "${line}" == "-A POSTROUTING "* && "${line}" == *"-s ${guest_ip}/32 "* && "${line}" == *" -j MASQUERADE"* ]]; then
        delete_iptables_rule_line "nat" "${line}"
      fi
    done < <(capture_as_root iptables-save -t nat 2>/dev/null || true)
  fi
}

cleanup_orphaned_iptables_rules() {
  if ! command_exists iptables-save || ! command_exists iptables; then
    warn "iptables-save or iptables not found; skipping orphaned firewall rule cleanup."
    return
  fi

  log "Checking for orphaned strangeClaw iptables rules..."

  local line
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line}" == "-A INPUT "* && "${line}" == *" -j DROP"* ]] \
      && iptables_line_has_fire_tap "${line}"; then
      delete_iptables_rule_line "filter" "${line}"
      continue
    fi
    if [[ "${line}" == "-A FORWARD "* && "${line}" == *" -j ACCEPT"* ]] \
      && iptables_line_has_fire_tap "${line}"; then
      delete_iptables_rule_line "filter" "${line}"
      continue
    fi
  done < <(capture_as_root iptables-save -t filter 2>/dev/null || true)

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line}" == "-A POSTROUTING "* && "${line}" == *" -j MASQUERADE"* ]] \
      && iptables_line_source_is_fire_guest "${line}"; then
      delete_iptables_rule_line "nat" "${line}"
    fi
  done < <(capture_as_root iptables-save -t nat 2>/dev/null || true)
}

cleanup_shared_conntrack_rule_if_safe() {
  if ! command_exists ip || ! command_exists iptables-save || ! command_exists iptables; then
    return
  fi

  local -a taps=()
  local tap
  while IFS= read -r tap; do
    [[ -n "${tap}" ]] && taps+=("${tap}")
  done < <(list_fire_taps)

  if [[ "${#taps[@]}" -ne 0 ]]; then
    return
  fi

  local line
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line}" == "-A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT" ]]; then
      log "Removing shared Firecracker conntrack rule; no strangeClaw TAP devices remain."
      delete_iptables_rule_line "filter" "${line}"
      return
    fi
  done < <(capture_as_root iptables-save -t filter 2>/dev/null || true)
}

cleanup_taps_and_firewall() {
  log "Checking for orphaned strangeClaw TAP devices..."
  if ! command_exists ip; then
    warn "ip command not found; skipping TAP cleanup."
    return
  fi

  local -a taps=()
  local tap
  while IFS= read -r tap; do
    [[ -n "${tap}" ]] && taps+=("${tap}")
  done < <(list_fire_taps)

  if [[ "${#taps[@]}" -eq 0 ]]; then
    log "No strangeClaw TAP devices found."
    return
  fi

  for tap in "${taps[@]}"; do
    local tap_ip=""
    local guest_ip=""
    tap_ip="$(tap_ipv4 "${tap}")"
    if [[ -n "${tap_ip}" ]]; then
      guest_ip="$(guest_ip_from_tap_ip "${tap_ip}" || true)"
    fi
    cleanup_iptables_for_tap "${tap}" "${guest_ip}"
    log "Removing TAP device ${tap}..."
    run_as_root ip link del "${tap}" || true
  done
}

is_fire_runtime_path() {
  local path="$1"
  local base
  base="$(basename -- "${path}")"

  if [[ -d "${path}" ]]; then
    [[ -e "${path}/firecracker.socket" \
      || -e "${path}/fire.vsock" \
      || -e "${path}/firecracker.log" \
      || -e "${path}/rootfs.ext4" ]]
    return
  fi

  [[ "${base}" =~ ^strangeclaw-.*\.(socket|vsock|sock|log)$ ]]
}

cleanup_runtime_paths() {
  log "Checking for stale strangeClaw runtime paths under ${TEMP_ROOT}..."
  if [[ ! -d "${TEMP_ROOT}" ]]; then
    warn "Temp root does not exist: ${TEMP_ROOT}"
    return
  fi

  local found=0
  local path
  while IFS= read -r -d '' path; do
    if is_fire_runtime_path "${path}"; then
      found=1
      log "Removing stale runtime path ${path}"
      run_as_root rm -rf -- "${path}" || true
    else
      log "Skipping unrecognized strangeClaw temp path ${path}"
    fi
  done < <(find "${TEMP_ROOT}" -maxdepth 1 -name 'strangeclaw-*' -print0 2>/dev/null)

  if [[ "${found}" -eq 0 ]]; then
    log "No stale strangeClaw runtime paths found."
  fi
}

main() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "strangeClaw Fire cleanup (dry run)"
  else
    log "strangeClaw Fire cleanup"
  fi
  cleanup_firecracker_processes
  cleanup_taps_and_firewall
  cleanup_orphaned_iptables_rules
  cleanup_shared_conntrack_rule_if_safe
  cleanup_runtime_paths
  log "Cleanup complete."
}

main
