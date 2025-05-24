//
// Created by Joe Barker on 2018/08/28.
//

#include <iostream>
#include <fstream>

#include <fcntl.h>
#include <unistd.h>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/helpers/output.h"

namespace jams {
    namespace output {
        void desync_io() {
          std::cin.tie(nullptr);
          std::ios_base::sync_with_stdio(false);
        }

        void set_default_cout_flags() {
          std::cout << std::boolalpha;
        }

        void initialise() {
          desync_io();
          set_default_cout_flags();
        }

        std::string output_path() {
          if (jams::instance().output_path().empty()) {
            return std::string();
          }
          return jams::instance().output_path() + "/";
        }

        std::string full_path_filename(const std::string &ending) {
          auto sep = file_basename_no_extension(ending).empty() ? "" : "_";
          return output_path() + ::globals::simulation_name + sep + ending;
        }

        std::string full_path_filename_series(const std::string &ending, int num, int width) {
          auto base = file_basename_no_extension(ending);
          auto sep = base.empty() ? "" : "_";
          auto ext = file_extension(ending);
          return output_path() + ::globals::simulation_name + sep + base + "_" + zero_pad_number(num, width) + "." + ext;
        }


        void open_output_file_just_in_time(std::ofstream& os, const std::string &filename) {
          if (!os.is_open()) {
            os.open(jams::output::full_path_filename(filename));
          }
        }

        int lock_file(const std::string& lock_filename) {
          int fd = open(lock_filename.c_str(), O_CREAT | O_RDWR, 0666);
          if (fd < 0) {
            perror("open lock file");
            exit(1);
          }

          struct flock fl{};
          fl.l_type = F_WRLCK;
          fl.l_whence = SEEK_SET;
          fl.l_start = 0;
          fl.l_len = 0;  // entire file

          if (fcntl(fd, F_SETLKW, &fl) == -1) {
            perror("fcntl lock");
            close(fd);
            exit(1);
          }

          return fd;  // keep it open to maintain the lock
        }

        void unlock_file(int fd) {
          struct flock fl{};
          fl.l_type = F_UNLCK;
          fl.l_whence = SEEK_SET;
          fl.l_start = 0;
          fl.l_len = 0;

          fcntl(fd, F_SETLK, &fl);
          close(fd);
        }
    }
  }


