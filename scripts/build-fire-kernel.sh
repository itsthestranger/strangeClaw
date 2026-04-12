#!/usr/bin/env bash
# Fallback builder for Firecracker guest kernels when CI artifacts are unsuitable.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION_FILE="${PROJECT_ROOT}/firecracker/VERSION"
OUT_DIR="${PROJECT_ROOT}/firecracker/kernel"

KERNEL_VERSION="6.1"
WORK_DIR=""
KEEP_WORKDIR=0

usage() {
  cat <<'USAGE'
Usage: scripts/build-fire-kernel.sh [--kernel-version <5.10|5.10-no-acpi|6.1>] [--keep-workdir]

Builds Firecracker CI-style kernel artifacts via the official devtool recipe and
installs the resulting kernel + config under firecracker/kernel/.

Options:
  --kernel-version VERSION Kernel version to build (default: 6.1)
  --keep-workdir          Keep temporary checkout/build directory.
  -h, --help              Show this help.
USAGE
}

log() {
  printf '[build-fire-kernel] %s\n' "$*"
}

fail() {
  printf '[build-fire-kernel] ERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 1 ]]; then
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
      log "Keeping workdir: ${WORK_DIR}"
    fi
    return
  fi

  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
    rm -rf "${WORK_DIR}" || true
  fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-version)
      [[ $# -ge 2 ]] || fail "Missing value for --kernel-version"
      KERNEL_VERSION="$2"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
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

case "${KERNEL_VERSION}" in
  5.10|5.10-no-acpi|6.1)
    ;;
  *)
    fail "Unsupported kernel version '${KERNEL_VERSION}'. Use 5.10, 5.10-no-acpi, or 6.1."
    ;;
esac

if ! command_exists git; then
  fail "git is required."
fi
if ! command_exists docker && ! command_exists podman; then
  fail "docker or podman is required for Firecracker devtool builds."
fi
if ! command_exists python3; then
  fail "python3 is required for Firecracker devtool."
fi

PINNED_VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}" 2>/dev/null || true)"
[[ -n "${PINNED_VERSION}" ]] || fail "Missing or empty ${VERSION_FILE}"

WORK_DIR="$(mktemp -d)"
REPO_DIR="${WORK_DIR}/firecracker"

log "Cloning Firecracker ${PINNED_VERSION}"
git clone --depth 1 --branch "${PINNED_VERSION}" https://github.com/firecracker-microvm/firecracker "${REPO_DIR}" >/dev/null 2>&1 || \
  fail "Failed to clone Firecracker ${PINNED_VERSION}"

log "Building kernel artifacts via tools/devtool"
(
  cd "${REPO_DIR}" || exit 1
  ./tools/devtool build_ci_artifacts kernels "${KERNEL_VERSION}"
) || fail "Kernel build failed. Ensure docker/podman is usable for your user."

ARCH="$(uname -m)"
RESOURCE_DIR="${REPO_DIR}/resources/${ARCH}"
[[ -d "${RESOURCE_DIR}" ]] || fail "Expected kernel artifact directory not found: ${RESOURCE_DIR}"

KERNEL_PATH="$(find "${RESOURCE_DIR}" -type f -name "vmlinux-${KERNEL_VERSION}*" | sort -V | tail -n 1)"
if [[ -z "${KERNEL_PATH}" ]]; then
  KERNEL_PATH="$(find "${RESOURCE_DIR}" -type f -name 'vmlinux-*' | sort -V | tail -n 1)"
fi
[[ -n "${KERNEL_PATH}" ]] || fail "No built vmlinux artifact found under ${RESOURCE_DIR}"

CONFIG_PATH="$(find "${RESOURCE_DIR}" -type f -name "config-${KERNEL_VERSION}*" | sort -V | tail -n 1)"
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="$(find "${RESOURCE_DIR}" -type f -name 'config-*' | sort -V | tail -n 1)"
fi
[[ -n "${CONFIG_PATH}" ]] || fail "No built kernel config artifact found under ${RESOURCE_DIR}"

mkdir -p "${OUT_DIR}"
cp -f "${KERNEL_PATH}" "${OUT_DIR}/$(basename "${KERNEL_PATH}")"
cp -f "${KERNEL_PATH}" "${OUT_DIR}/vmlinux"
cp -f "${CONFIG_PATH}" "${OUT_DIR}/$(basename "${CONFIG_PATH}")"
cp -f "${CONFIG_PATH}" "${OUT_DIR}/kernel.config"

if ! grep -q '^CONFIG_VIRTIO_VSOCKETS=y$' "${OUT_DIR}/kernel.config"; then
  fail "Built kernel config does not have CONFIG_VIRTIO_VSOCKETS=y"
fi

log "Installed kernel artifacts to ${OUT_DIR}:"
log "  ${OUT_DIR}/vmlinux"
log "  ${OUT_DIR}/kernel.config"
log "Kernel fallback build completed successfully."
