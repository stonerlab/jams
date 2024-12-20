//
// Created by Joseph Barker on 2020-03-11.
//

#include <string>
#include <cstring>
#include <cerrno>
#include <cstdio>

// POSIX headers
#include <unistd.h>
#include <sys/stat.h>
#include <jams/helpers/utils.h>

#include "jams/interface/system.h"

bool jams::system::file_exists(const std::string &path) {
  struct stat buffer;
  return (stat (path.c_str(), &buffer) == 0);
}

void jams::system::make_directory(const std::string &path, mode_t mode) {
  // https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
  typedef struct stat Stat;

  Stat st;
  int status = 0;
  if (stat(path.c_str(), &st) != 0) {
    /* Directory does not exist. EEXIST for race condition */
    if (mkdir(path.c_str(), mode) != 0 && errno != EEXIST) {
      throw std::runtime_error("mkdir failed: " + std::string(strerror(errno)));
    }
  } else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    throw std::runtime_error("mkdir failed: " + std::string(strerror(errno)));
  }
}

void jams::system::make_path(const std::string &path, mode_t mode) {
  auto directories = split(path, "/");

  std::string total_path;
  for (const auto &d : directories) {
    total_path += d + "/";
    make_directory(total_path, mode);
  }
}

bool jams::system::stdout_is_tty() {
  return isatty(fileno(stdout));
}

