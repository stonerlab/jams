#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-integration-tests.sh [options]

Options:
  -p, --python <path>        Python interpreter to use (default: python3)
  -v, --venv <dir>           Venv directory (default: .venv-jams-tests)
  -e, --env-file <path>      Environment file for test deps (default: test/environment.yml)
  -w, --work-dir <dir>       Working directory for test outputs (default: temp dir)
  -b, --build-dir <dir>      CMake build directory (default: cmake-build-integration)
  -t, --build-type <type>    CMake build type (default: Release)
  -g, --generator <name>     CMake generator (default: unset)
  -o, --cmake-option <arg>   Extra CMake option (repeatable)
  -j, --jobs <n>             Parallel build jobs (default: cmake default)
  --tests <path>             Test module or file (repeatable)
  --binary-path <path>       Use existing JAMS binary instead of build output
  --reuse-binary             Reuse existing binary in the build directory
  --enable-gpu               Enable GPU solver tests
  --reuse-venv               Reuse existing venv if present
  --skip-build               Skip the CMake build step
  -h, --help                 Show this help
EOF
}

PYTHON_BIN="python3"
VENV_DIR=".venv-jams-tests"
ENV_FILE="test/environment.yml"
BUILD_DIR="cmake-build-integration"
BUILD_TYPE="Release"
GENERATOR=""
JOBS=""
ENABLE_GPU=""
REUSE_VENV=""
REUSE_BINARY=""
SKIP_BUILD=""
EXTERNAL_BINARY=""
WORK_DIR=""
declare -a CMAKE_OPTIONS=()
declare -a TESTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -v|--venv)
      VENV_DIR="$2"
      shift 2
      ;;
    -e|--env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    -w|--work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    -b|--build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    -t|--build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    -g|--generator)
      GENERATOR="$2"
      shift 2
      ;;
    -o|--cmake-option)
      CMAKE_OPTIONS+=("$2")
      shift 2
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    --tests)
      TESTS+=("$2")
      shift 2
      ;;
    --binary-path)
      EXTERNAL_BINARY="$2"
      shift 2
      ;;
    --reuse-binary)
      REUSE_BINARY="1"
      shift 1
      ;;
    --enable-gpu)
      ENABLE_GPU="1"
      shift 1
      ;;
    --reuse-venv)
      REUSE_VENV="1"
      shift 1
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=("test/test_integrator.py")
fi

if [[ -z "${REUSE_VENV}" && -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip

install_env_deps() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    echo "Environment file not found: ${env_file}"
    exit 1
  fi

  case "${env_file}" in
    *.yml|*.yaml)
      local in_pip=0
      local pip_deps=()
      while IFS= read -r line; do
        if [[ "${line}" =~ ^[[:space:]]*-[[:space:]]pip:[[:space:]]*$ ]]; then
          in_pip=1
          continue
        fi
        if [[ ${in_pip} -eq 1 ]]; then
          if [[ "${line}" =~ ^[[:space:]]*-[[:space:]]+([^[:space:]].*)$ ]]; then
            pip_deps+=("${BASH_REMATCH[1]}")
            continue
          fi
          if [[ "${line}" =~ ^[[:space:]]*- ]]; then
            in_pip=0
            continue
          fi
        fi
      done < "${env_file}"

      if [[ ${#pip_deps[@]} -gt 0 ]]; then
        "${VENV_DIR}/bin/python" -m pip install "${pip_deps[@]}"
      fi
      ;;
    *.txt)
      "${VENV_DIR}/bin/python" -m pip install -r "${env_file}"
      ;;
    *)
      echo "Unsupported environment file type: ${env_file}"
      exit 1
      ;;
  esac
}

install_env_deps "${ENV_FILE}"

if [[ -z "${SKIP_BUILD}" && -z "${EXTERNAL_BINARY}" ]]; then
  if [[ -n "${REUSE_BINARY}" && -x "${BUILD_DIR}/bin/jams" ]]; then
    SKIP_BUILD="1"
  fi
fi

if [[ -z "${SKIP_BUILD}" && -z "${EXTERNAL_BINARY}" ]]; then
  cmake_args=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
  if [[ ${#CMAKE_OPTIONS[@]} -gt 0 ]]; then
    for opt in "${CMAKE_OPTIONS[@]}"; do
      cmake_args+=("${opt}")
    done
  fi

  if [[ -n "${GENERATOR}" ]]; then
    cmake -S . -B "${BUILD_DIR}" -G "${GENERATOR}" "${cmake_args[@]}"
  else
    cmake -S . -B "${BUILD_DIR}" "${cmake_args[@]}"
  fi

  if [[ -n "${JOBS}" ]]; then
    cmake --build "${BUILD_DIR}" --target jams --parallel "${JOBS}"
  else
    cmake --build "${BUILD_DIR}" --target jams
  fi
fi

if [[ -n "${EXTERNAL_BINARY}" ]]; then
  BINARY_PATH="${EXTERNAL_BINARY}"
else
  BINARY_PATH="${BUILD_DIR}/bin/jams"
fi

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "JAMS binary not found or not executable: ${BINARY_PATH}"
  exit 1
fi

export JAMS_BINARY_PATH="${BINARY_PATH}"
if [[ -n "${ENABLE_GPU}" ]]; then
  export JAMS_TEST_ENABLE_GPU=1
fi
if [[ -n "${WORK_DIR}" ]]; then
  export JAMS_TEST_WORKDIR="${WORK_DIR}"
fi

"${VENV_DIR}/bin/python" -m unittest "${TESTS[@]}"
