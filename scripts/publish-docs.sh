#!/usr/bin/env bash

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

declare -r JAMS_SRC_DIR=$DIR/../
declare -r JAMS_BUILD_DIR=$TMP_DIR/build
declare -r DOCS_SRC=$JAMS_BUILD_DIR/docs/html

declare -r GH_PAGES_REPO=https://drjbarker@github.com/drjbarker/jams.git
declare -r GH_PAGES_BRANCH=gh-pages
declare -r GH_PAGES_DIR=$TMP_DIR/gh-pages

function usage {
	echo "usage: $0 [-b <branch>] [-c <commit>] [-t <build_type>] [-g <generator>] [-d | Debug]"
}

function clean_exit {
  local exit_code=$1
  if [ "$exit_code" -ne "0" ]; then
    error_exit "EXIT $1"
  fi
#  rm -rf "$TMP_DIR"
}

function error_exit {
  local signal=$1
  echo -e ""
  echo -e "\e[31m*** An error occured with signal ${signal} ***"
  echo -e ""
  cat "${LOG}"

#  rm -rf "$TMP_DIR"
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

git_short_hash() {
  git -C "$JAMS_SRC_DIR" rev-parse --short HEAD
}

shallow_clone_git_branch() {
  local repo="$1"
  local branch="$2"
  local destination="$3"
  mkdir -p "$destination"
  git clone -b "$branch" --single-branch --depth 1 "$repo" "$destination" >> "${LOG}" 2>&1
}

build_sphinx_docs() {
  message "\e[1m==> Running CMake...\e[0m"
  cmake -DJAMS_BUILD_DOCS=ON -DJAMS_BUILD_OMP=OFF -DJAMS_BUILD_CUDA=OFF -DJAMS_BUILD_TESTS=OFF -B"$JAMS_BUILD_DIR" -H"$JAMS_SRC_DIR" >> "${LOG}" 2>&1
  message "\e[1m==> Running make...\e[0m"
  make --directory="$JAMS_BUILD_DIR" jams-docs >> "${LOG}" 2>&1
}

publish_github_pages() {
  local message="Adding gh-pages docs for $(git_short_hash)"
  shallow_clone_git_branch $GH_PAGES_REPO $GH_PAGES_BRANCH $GH_PAGES_DIR
  cp -r ${DOCS_SRC}/* "$GH_PAGES_DIR" >> "${LOG}" 2>&1
  git -C "$GH_PAGES_DIR" add -A >> "${LOG}" 2>&1
  git -C "$GH_PAGES_DIR" commit -m "${message}" && git -C "$GH_PAGES_DIR" push origin gh-pages >> "${LOG}" 2>&1
}

main() {
  local commit=$(git_short_hash)

  cd "$TMP_DIR"

  message "\e[1m==> Building \e[34mjams-docs\e[39m from \e[32m${JAMS_SRC_DIR}\e[39m...\e[0m"
  build_sphinx_docs

  message "\e[1m==> Publishing github pages to \e[34m${GH_PAGES_REPO}\e[39m...\e[0m"
  publish_github_pages
}

main "$@"
