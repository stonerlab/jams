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

trap "clean_exit" EXIT
trap "error_exit 'SIGHUP'" SIGHUP
trap "error_exit 'SIGINT'" SIGINT
trap "error_exit 'SIGTERM'" SIGTERM

declare -r DIR="$(pwd)"
declare -r TMP_DIR=$(mktemp -d)
declare -r LOG="${TMP_DIR}/build.log"

declare -r MAKEFLAGS='-j 4'
declare -r URL=http://joebarker87@bitbucket.org/joebarker87/jams.git
declare -r EXE_PATH='build/src/jams/jams'

function clean_exit {      
  rm -rf "$TMP_DIR"
}

function error_exit {      
  local signal=$1
  echo "An error occured with signal ${signal}"
  echo ""
  echo "Logfile:"
  cat ${LOG}
 
  rm -rf "$TMP_DIR"
}

clean() {
  local directory="$1"
  if [ -d $directory ]; then
    rm -Rf $directory
  fi
}

message() {
  local text="$1"
  echo "${text}" | tee -a ${LOG}
}

copy_binary() {
  local workdir=$1
  local exe_name=$2
  cp ${workdir}/${EXE_PATH} ${DIR}/${exe_name}
}

clone_git_repo_shallow() {
  local repository=$1
  local branch=$2
  local destination=$3

  mkdir -p $destination
  git clone -b $branch --single-branch --depth 1 $repository $destination >> ${LOG} 2>&1 
}

clone_git_repo() {
  local repository=$1
  local destination=$2

  mkdir -p $destination
  git clone $repository $destination >> ${LOG} 2>&1 
}

checkout_git_commit() {
  local commit=$1
  local workdir=$2
  (cd ${workdir} && git checkout -q ${commit})
}

build_branch() {
  local branch="$1"
  local build_type=$2
  local generator=$3
  local cmake_extra_args="$4"

  local build_name=$(tr '[:upper:]' '[:lower:]' <<< "jams-${branch}-${build_type}")
  local cmake_args="-DCMAKE_BUILD_TYPE=${build_type} $cmake_extra_args"

  local workdir="${build_name}"

  message "Building ${workdir} from '${URL}'..."

  clean ${workdir}
  message "==> Cloning git repository..."
  clone_git_repo_shallow "${URL}" "${branch}" "${workdir}"
  message "==> Running CMake..."
  cmake_generate "${workdir}/build" "${generator}" "${cmake_args}" 
  message "==> Compiling source..."
  build "${workdir}/build" "${generator}"
  message "==> Creating binary..."
  copy_binary "${workdir}" "${build_name}"
}

build_commit() {
  local commit="$1"
  local build_type=$2
  local generator=$3
  local cmake_extra_args="$4"

  local build_name=$(tr '[:upper:]' '[:lower:]' <<< "jams-${commit}-${build_type}")
  local cmake_args="-DCMAKE_BUILD_TYPE=${build_type} $cmake_extra_args"

  local workdir="${build_name}"

  message "Building ${workdir} from '${URL}'..."

  clean "${workdir}"
  message "==> Cloning git repository..."
  clone_git_repo "${URL}" "${workdir}"
  checkout_git_commit "${commit}" "${workdir}"
  message "==> Running CMake..."
  cmake_generate "${workdir}/build" "${generator}" "${cmake_args}" 
  message "==> Compiling source..."
  build "${workdir}/build" "${generator}"
  message "==> Creating binary..."
  copy_binary "${workdir}" "${build_name}"
}

cmake_generate() {
  local cmake_build_dir="$1"
  local cmake_generator="$2"
  local cmake_args="$3"

  mkdir -p "${cmake_build_dir}"

  (cd -- "${cmake_build_dir}" && cmake -G "${cmake_generator}" ${cmake_args} .. >> ${LOG} 2>&1)
}

build() {
  local build_dir="$1"
  local build_system="$2"

  case "${build_system}" in
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
  local generator="Ninja"
  local extra_args=""

  while getopts ":b:c:dt:D:h:" opt; do
    case $opt in
      b )
        branch="$OPTARG"
        ;;
      c )
        commit="$OPTARG"
        ;;
      d )
        build_type="Debug"
        ;;
      t )
        build_type="$OPTARG"
        ;;
      g )
        generator="$OPTARG"
        ;;
      D )
        extra_args="-D$OPTARG $extra_args"
        ;;
      \?)
        echo "Invalid option: -$OPTARG"
        exit 1
        ;;
      esac
  done

  cd $TMP_DIR

  if [ ! -z "${commit}" ]; then
    build_commit "$commit" "$build_type" "$generator" "$extra_args"
  else
    build_branch "$branch" "$build_type" "$generator" "$extra_args"
  fi
}
 
main "$@"
