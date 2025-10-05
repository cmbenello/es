#!/usr/bin/env bash
# Build vmtouch into the current directory (./vmtouch)
# Requirements: git, make, and a C compiler (gcc or clang)
set -euo pipefail

# -------- config --------
REPO_URL="https://github.com/hoytech/vmtouch.git"
TARGET_DIR="$(pwd)"
TARGET_BIN="$TARGET_DIR/vmtouch"

# -------- helpers --------
have() { command -v "$1" >/dev/null 2>&1; }

cleanup() {
  [[ -n "${BUILD_DIR:-}" && -d "${BUILD_DIR:-}" ]] && rm -rf "$BUILD_DIR" || true
}
trap cleanup EXIT

# -------- check dependencies --------
missing=()
if ! have git; then missing+=("git"); fi
if ! have make; then missing+=("make"); fi
if ! have gcc && ! have clang; then missing+=("gcc or clang"); fi

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Error: Missing required tools: ${missing[*]}" >&2
  echo "Please install them manually and try again." >&2
  exit 1
fi

# Determine compiler
if have clang; then CC=clang
elif have gcc; then CC=gcc
fi

# -------- build --------
BUILD_DIR="$(mktemp -d -t vmtouch-build-XXXXXX)"
echo "Building in: $BUILD_DIR"
git clone --depth=1 "$REPO_URL" "$BUILD_DIR/vmtouch"
cd "$BUILD_DIR/vmtouch"

# Build
echo "Compiling vmtouch..."
make CC="${CC:-cc}"

# Copy the binary to the current directory
cp -f vmtouch "$TARGET_BIN"
chmod +x "$TARGET_BIN"

echo
echo "✅ Built: $TARGET_BIN"
echo
echo "Version check:"
"$TARGET_BIN" -v || true
