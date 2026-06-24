#!/usr/bin/env bash
# Build a custom Fire guest rootfs from firecracker/rootfs/Dockerfile.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="${PROJECT_ROOT}/firecracker/rootfs/Dockerfile"
OUTPUT_EXT4="${PROJECT_ROOT}/firecracker/rootfs/agent.ext4"

CONTAINER_RUNTIME=""
IMAGE_TAG=""
KEEP_IMAGE=0

TMP_DIR=""
ROOTFS_DIR=""
EXPORT_TAR=""
CONTAINER_ID=""
CREATED_IMAGE=0

usage() {
  cat <<'USAGE'
Usage: scripts/build-fire-rootfs.sh [--runtime docker|podman] [--image-tag TAG] [--keep-image]

Build pipeline:
  1) Build rootfs container image from firecracker/rootfs/Dockerfile
  2) Create + export container filesystem
  3) Build firecracker/rootfs/agent.ext4 via mkfs.ext4 -d (no sudo)
  4) Verify critical guest files with debugfs

Options:
  --runtime RUNTIME   Container runtime to use: docker or podman (auto-detected by default)
  --image-tag TAG     Container image tag to use/create
  --keep-image        Keep the built image after export
  -h, --help          Show this help
USAGE
}

log() {
  printf '[build-fire-rootfs] %s\n' "$*"
}

fail() {
  printf '[build-fire-rootfs] ERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ -n "${CONTAINER_ID}" ]]; then
    "${CONTAINER_RUNTIME}" rm -f "${CONTAINER_ID}" >/dev/null 2>&1 || true
  fi

  if [[ "${CREATED_IMAGE}" -eq 1 && "${KEEP_IMAGE}" -eq 0 && -n "${IMAGE_TAG}" ]]; then
    "${CONTAINER_RUNTIME}" rmi "${IMAGE_TAG}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}" || true
  fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      [[ $# -ge 2 ]] || fail "Missing value for --runtime"
      CONTAINER_RUNTIME="$2"
      shift 2
      ;;
    --image-tag)
      [[ $# -ge 2 ]] || fail "Missing value for --image-tag"
      IMAGE_TAG="$2"
      shift 2
      ;;
    --keep-image)
      KEEP_IMAGE=1
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

if [[ -z "${CONTAINER_RUNTIME}" ]]; then
  if command_exists docker; then
    CONTAINER_RUNTIME="docker"
  elif command_exists podman; then
    CONTAINER_RUNTIME="podman"
  else
    fail "Neither docker nor podman is available."
  fi
fi

case "${CONTAINER_RUNTIME}" in
  docker|podman)
    ;;
  *)
    fail "Unsupported runtime: ${CONTAINER_RUNTIME}. Use docker or podman."
    ;;
esac

required_cmds=(tar mkfs.ext4 debugfs e2fsck truncate du awk stat)
for cmd in "${required_cmds[@]}"; do
  command_exists "${cmd}" || fail "Missing required command: ${cmd}"
done

[[ -f "${DOCKERFILE_PATH}" ]] || fail "Missing Dockerfile: ${DOCKERFILE_PATH}"

if [[ -z "${IMAGE_TAG}" ]]; then
  IMAGE_TAG="strangeclaw-fire-rootfs:$(date +%Y%m%d%H%M%S)"
fi

TMP_DIR="$(mktemp -d)"
ROOTFS_DIR="${TMP_DIR}/rootfs"
EXPORT_TAR="${TMP_DIR}/rootfs.tar"
mkdir -p "${ROOTFS_DIR}"

log "Building image ${IMAGE_TAG} with ${CONTAINER_RUNTIME}"
"${CONTAINER_RUNTIME}" build \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE_PATH}" \
  "${PROJECT_ROOT}" || fail "Container image build failed."
CREATED_IMAGE=1

log "Creating container for filesystem export"
CONTAINER_ID="$("${CONTAINER_RUNTIME}" create "${IMAGE_TAG}")" || \
  fail "Failed to create temporary container."

log "Exporting container filesystem"
"${CONTAINER_RUNTIME}" export "${CONTAINER_ID}" -o "${EXPORT_TAR}" || \
  fail "Failed to export container filesystem."

log "Extracting rootfs tarball"
tar -xf "${EXPORT_TAR}" -C "${ROOTFS_DIR}" || fail "Failed to extract rootfs tarball."

bytes="$(du -sb "${ROOTFS_DIR}" | awk '{print $1}')"
[[ -n "${bytes}" ]] || fail "Could not calculate extracted rootfs size."

# Allocate +25% growth headroom and +256 MiB fixed slack; floor at 1 GiB.
size_mib=$(( (bytes + (bytes / 4) + 268435456 + 1048575) / 1048576 ))
if (( size_mib < 1024 )); then
  size_mib=1024
fi

mkdir -p "$(dirname "${OUTPUT_EXT4}")"
tmp_ext4="${OUTPUT_EXT4}.tmp"
rm -f "${tmp_ext4}"

log "Building ext4 image (${size_mib} MiB)"
truncate -s "${size_mib}M" "${tmp_ext4}" || fail "truncate failed for ${tmp_ext4}"
mkfs.ext4 -q -F -d "${ROOTFS_DIR}" "${tmp_ext4}" || fail "mkfs.ext4 failed"
e2fsck -fn "${tmp_ext4}" >/dev/null || fail "e2fsck validation failed"

debugfs_check_path() {
  local path="$1"
  if ! debugfs -R "stat ${path}" "${tmp_ext4}" >/dev/null 2>&1; then
    fail "Missing required path in rootfs image: ${path}"
  fi
}

debugfs_stat_text() {
  local path="$1"
  debugfs -R "stat ${path}" "${tmp_ext4}" 2>/dev/null
}

debugfs_expect_type() {
  local path="$1"
  local expected_type="$2"
  local stat_text
  stat_text="$(debugfs_stat_text "${path}")"
  if [[ "${stat_text}" != *"Type: ${expected_type}"* ]]; then
    fail "Unexpected file type for ${path}. Expected '${expected_type}'."
  fi
}

debugfs_dump_file() {
  local src_path="$1"
  local dst_path="$2"
  if ! debugfs -R "dump -p ${src_path} ${dst_path}" "${tmp_ext4}" >/dev/null 2>&1; then
    fail "Failed to dump ${src_path} from rootfs image."
  fi
}

log "Verifying required guest files with debugfs"
debugfs_check_path "/sbin/init"
debugfs_check_path "/sbin/entrypoint.sh"
debugfs_check_path "/sbin/tini"
debugfs_check_path "/opt/strangeclaw/agent/agent.py"
debugfs_check_path "/opt/strangeclaw/agent/subagents.py"
debugfs_expect_type "/sbin/init" "regular"
debugfs_expect_type "/sbin/entrypoint.sh" "regular"
debugfs_expect_type "/opt/strangeclaw/agent/agent.py" "regular"

init_dump="${TMP_DIR}/init.sh"
debugfs_dump_file "/sbin/init" "${init_dump}"
expected_init='#!/bin/sh
exec /sbin/tini -- /sbin/entrypoint.sh'
if [[ "$(cat "${init_dump}")" != "${expected_init}" ]]; then
  fail "Unexpected /sbin/init contents in rootfs image."
fi

mv -f "${tmp_ext4}" "${OUTPUT_EXT4}" || fail "Failed to install ${OUTPUT_EXT4}"

image_size_bytes="$(stat -c '%s' "${OUTPUT_EXT4}")"
log "Rootfs build complete:"
log "  Runtime: ${CONTAINER_RUNTIME}"
log "  Image: ${IMAGE_TAG}"
log "  Output: ${OUTPUT_EXT4} (${image_size_bytes} bytes)"
