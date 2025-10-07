#!/usr/bin/env bash

set -euo pipefail

# DuckDB 1.3.x and 1.4.x side-by-side installer for Linux x86_64
# - Downloads CLIs for both versions into current directory by default (override with --bin-dir)
# - No import or query logic here; see scripts/duckdb_lineitem_run.sh for running
#
# Usage examples:
#   scripts/duckdb_setup.sh
#   scripts/duckdb_setup.sh --version-13 1.3.2 --version-14 1.4.0 --bin-dir .
#
# You can customize defaults via env vars:
#   DUCKDB_V13, DUCKDB_V14, INSTALL_BIN_DIR, WORK_DIR


VER13_DEFAULT="1.3.2"
VER14_DEFAULT="1.4.1"

VER13="${DUCKDB_V13:-$VER13_DEFAULT}"
VER14="${DUCKDB_V14:-$VER14_DEFAULT}"

INSTALL_BIN_DIR="${INSTALL_BIN_DIR:-${PWD}}"
WORK_DIR="${WORK_DIR:-${PWD}/.duckdb_cli_setup}"
FORCE=0

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "[WARN] %s\n" "$*" >&2; }
die() { printf "[ERR ] %s\n" "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Linux x86_64 DuckDB 1.3.x + 1.4.x CLI setup

Flags:
  --version-13 <ver>     DuckDB 1.3.x version (default: ${VER13_DEFAULT})
  --version-14 <ver>     DuckDB 1.4.x version (default: ${VER14_DEFAULT})
  --bin-dir <dir>        Install CLIs to this dir (default: current dir)
  --work-dir <dir>       Temp working dir for downloads (default: ${WORK_DIR})
  --force                Re-download CLIs even if present
  --verbose              Enable shell tracing
  -h, --help             Show this help

Examples:
  $0
  $0 --version-13 1.3.2 --version-14 1.4.0 --bin-dir .
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_linux_x86_64() {
  local os arch
  os="$(uname -s)" || true
  arch="$(uname -m)" || true
  [[ "$os" == "Linux" ]] || die "This script supports Linux only (detected: $os)"
  [[ "$arch" == "x86_64" ]] || die "This script targets x86_64 (detected: $arch)"
}

strip_v_prefix() {
  # prints version without a leading 'v'
  local v="$1"
  if [[ "$v" == v* ]]; then
    printf "%s" "${v#v}"
  else
    printf "%s" "$v"
  fi
}

ensure_dirs() { mkdir -p "$INSTALL_BIN_DIR" "$WORK_DIR"; }

duckdb_bin_for() {
  # echoes absolute path to installed binary for a given version
  local ver
  ver="$(strip_v_prefix "$1")"
  printf "%s/duckdb-%s" "$INSTALL_BIN_DIR" "$ver"
}

download_cli() {
  local ver raw ver_tag url zip tmp_dir bin_path
  raw="$1"
  ver="$(strip_v_prefix "$raw")"
  ver_tag="v${ver}"
  url="https://github.com/duckdb/duckdb/releases/download/${ver_tag}/duckdb_cli-linux-amd64.zip"
  zip="${WORK_DIR}/duckdb_cli-${ver}.zip"
  tmp_dir="${WORK_DIR}/cli-${ver}"
  bin_path="$(duckdb_bin_for "$ver")"

  if [[ -x "$bin_path" && $FORCE -eq 0 ]]; then
    log "DuckDB ${ver} already installed at ${bin_path}"
    return 0
  fi

  log "Downloading DuckDB CLI ${ver} from ${url}"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  curl -fL --retry 3 -o "$zip" "$url"
  require_cmd unzip
  unzip -o "$zip" -d "$tmp_dir" >/dev/null
  if [[ ! -f "$tmp_dir/duckdb" ]]; then
    die "Unexpected archive layout: missing ${tmp_dir}/duckdb"
  fi
  install -m 0755 "$tmp_dir/duckdb" "$bin_path"
  log "Installed ${bin_path}"
}

sql_escape() {
  # escape single-quotes for SQL string literals
  printf "%s" "$1" | sed "s/'/''/g"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --version-13) VER13="$2"; shift 2 ;;
      --version-14) VER14="$2"; shift 2 ;;
      --bin-dir) INSTALL_BIN_DIR="$2"; shift 2 ;;
      --work-dir) WORK_DIR="$2"; shift 2 ;;
      --force) FORCE=1; shift ;;
      --verbose) set -x; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown argument: $1 (use --help)" ;;
    esac
  done
}

main() {
  parse_args "$@"
  ensure_linux_x86_64
  require_cmd curl
  ensure_dirs

  log "Target versions: 1.3.x=${VER13}, 1.4.x=${VER14}"
  download_cli "$VER13"
  download_cli "$VER14"

  log "Install complete. Binaries: $(duckdb_bin_for "$VER13"), $(duckdb_bin_for "$VER14")"
  log "Ensure ${INSTALL_BIN_DIR} is on your PATH. Example:"
  log "  export PATH=\"${INSTALL_BIN_DIR}:$PATH\""
}

main "$@"
