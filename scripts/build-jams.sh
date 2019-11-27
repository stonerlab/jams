#!/usr/bin/env bash

# date:   2017-03-26
# author: Joseph Barker
# email:  joseph.barker@imr.tohoku.ac.jp

# trace error through pipes
set -o pipefail 
# trace error through time command and others
set -o errtrace
# disallow uninitialised variables
set -o nounset
# exit on first error
set -o errexit

trap 'clean_exit $?' EXIT
trap "error_exit 'SIGHUP'" SIGHUP
trap "error_exit 'SIGINT'" SIGINT
trap "error_exit 'SIGTERM'" SIGTERM

declare -r DIR="$(pwd)"
declare -r TMP_DIR=$(mktemp -d)
declare -r LOG="${TMP_DIR}/build.log"

declare -r NUMPROC="$(getconf _NPROCESSORS_ONLN)"
declare -r MAKEFLAGS="-j$NUMPROC"
declare -r URL=https://drjbarker@github.com/stonerlab/jams.git
declare -r EXE_PATH='bin/jams'
declare -r NVCC_PATH="$(which nvcc)"

function usage {
	echo "usage: $0 [-b <branch>] [-c <commit>] [-t <build_type>] [-g <generator>] [-d | Debug]"
}

function clean_exit {
  local exit_code=$1
  if [ "$exit_code" -ne "0" ]; then
    error_exit "EXIT $1"
  fi     
  rm -rf "$TMP_DIR"
}

function error_exit {      
  local signal=$1
  echo -e ""
  echo -e "\e[31m*** An error occured with signal ${signal} ***"
  echo -e ""
  cat "${LOG}"
 
  rm -rf "$TMP_DIR"
}

clean() {
  local directory="$1"
  if [ -d "$directory" ]; then
    rm -Rf "$directory"
  fi
}

message() {
  local text="$1"
  echo -e "${text}" | tee -a "${LOG}"
}

copy_binary() {
  local workdir=$1
  local exe_name=$2
  cp "${workdir}/build/${EXE_PATH}" "${DIR}/${exe_name}"
}

clone_git_branch_shallow() {
  local repository=$1
  local branch=$2
  local destination=$3

  mkdir -p "$destination"
  git clone -b "$branch" --single-branch --depth 1 "$repository" "$destination" >> "${LOG}" 2>&1 
}

clone_git_commit() {
  local repository=$1
  local commit=$2
  local destination=$3

  mkdir -p "$destination"
  git clone "$repository" "$destination" >> "${LOG}" 2>&1 
  (cd "${destination}" && git checkout -q "${commit}")
}

build_branch() {
  local branch="$1"
  local build_type="$2"
  local build_options="$3"
  local generator="$4"

  local sanitized_branch
  sanitized_branch=$(basename "${branch}")

  local build_name
  build_name=$(tr '[:upper:]' '[:lower:]' <<< "jams-${sanitized_branch}-${build_type}")
  local cmake_args="-DCMAKE_BUILD_TYPE=${build_type} ${build_options}"

  local workdir="${build_name}"

  message "\e[1m==> Building \e[34m${workdir}\e[39m from \e[32m${URL}\e[39m...\e[0m"

  clean "${workdir}"
  message "\e[1m==> Cloning git repository...\e[0m"
  clone_git_branch_shallow "${URL}" "${branch}" "${workdir}"
  message "\e[1m==> Running CMake...\e[0m"
  cmake_generate "${workdir}/build" "${cmake_args}" "${generator}"
  message "\e[1m==> Compiling source...\e[0m"
  build "${workdir}/build" "${generator}"
  message "\e[1m==> Creating binary...\e[0m"
  copy_binary "${workdir}" "${build_name}"
  message "\e[33m${DIR}/${build_name}\e[0m"
}

build_commit() {
  local commit="$1"
  local build_type="$2"
  local build_options="$3"
  local generator="$4"

  local build_name
  build_name=$(tr '[:upper:]' '[:lower:]' <<< "jams-${commit}-${build_type}")
  local cmake_args="-DCMAKE_BUILD_TYPE=${build_type} ${build_options}"

  local workdir="${build_name}"

  message "\e[1m==> Building \e[34m${workdir}\e[39m from \e[32m${URL}\e[39m...\e[0m"

  clean "${workdir}"
  message "\e[1m==> Cloning git repository...\e[0m"
  clone_git_commit "${URL}" "$commit" "${workdir}"
  message "\e[1m==> Running CMake...\e[0m"
  cmake_generate "${workdir}/build" "${cmake_args}" "${generator}"
  message "\e[1m==> Compiling source...\e[0m"
  build "${workdir}/build" "${generator}"
  message "\e[1m==> Creating binary...\e[0m"
  copy_binary "${workdir}" "${build_name}"
  message "\e[33m${DIR}/${build_name}\e[0m"
}

cmake_generate() {
  local cmake_build_dir="$1"
  local cmake_args="$2"
  local cmake_generator="$3"

  mkdir -p "${cmake_build_dir}"

  # echo "cmake .. -G \"${cmake_generator}\" ${cmake_args}"
  (cd -- "${cmake_build_dir}" && cmake .. -G "${cmake_generator}" ${cmake_args} >> "${LOG}" 2>&1)
}

build() {
  local build_dir="$1"
  local build_system="$2"

  case ${build_system} in
    "Ninja")
	(cd -- "${build_dir}" && ninja jams >> "${LOG}" 2>&1)
	;;
    "Unix Makefiles")
	(cd -- "${build_dir}" && make "${MAKEFLAGS}" >> "${LOG}" 2>&1)
	;;
    /?)
	echo "Unknown build system: ${build_system}"
	exit 1
	;;
  esac
}

main() {
  local branch="develop"
  local commit=""
  local build_type="Release"
  local build_options=""
  local generator="Unix Makefiles"

  if [[ -z "${NVCC_PATH}" ]]; then
    message "\e[1m==> Disabling CUDA in build...\e[0m"
    build_options="-DJAMS_BUILD_CUDA=OFF"
  fi

  while getopts ":b:c:dt:o:g:h" opt; do
    case $opt in
      b )
        branch=$OPTARG
        ;;
      c )
        commit=$OPTARG
        ;;
      d )
        build_type="Debug"
        ;;
      t )
        build_type=$OPTARG
        ;;
      o )
        build_options="${build_options} $OPTARG"
        ;;
      g )
        generator=$OPTARG
        ;;
      h )
		usage
		exit 0
		;;
      \?)
        echo "Invalid option: -$OPTARG"
        usage
        exit 1
        ;;
      esac
  done

  cd "$TMP_DIR"

  if [[ -n "${commit}" ]]; then
    build_commit "$commit" "$build_type" "$build_options" "$generator"
  else
    build_branch "$branch" "$build_type" "$build_options" "$generator"
  fi
}
 
main "$@"
