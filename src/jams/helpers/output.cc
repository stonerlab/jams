//
// Created by Joe Barker on 2018/08/28.
//

#include <cerrno>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>

#include <fcntl.h>
#include <unistd.h>

#include "jams/common.h"
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

        void redirect_standard_streams(const std::string& filename) {
          std::cout.flush();
          std::cerr.flush();

          const int log_fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
          if (log_fd < 0) {
            throw std::runtime_error(
                "Failed to open log file: " + filename + ": " + std::strerror(errno));
          }

          if (dup2(log_fd, STDOUT_FILENO) < 0) {
            const std::string error = std::strerror(errno);
            close(log_fd);
            throw std::runtime_error(
                "Failed to redirect stdout to log file: " + filename + ": " + error);
          }

          if (dup2(log_fd, STDERR_FILENO) < 0) {
            const std::string error = std::strerror(errno);
            close(log_fd);
            throw std::runtime_error(
                "Failed to redirect stderr to log file: " + filename + ": " + error);
          }

          close(log_fd);
        }

        std::string output_path() {
          if (jams::instance().output_path().empty()) {
            return std::string();
          }
          return jams::instance().output_path() + "/";
        }

        std::string full_path_filename(const std::string &ending) {
          return (std::filesystem::path(output_path()) / ending).string();
        }

        std::string full_path_filename_series(const std::string &ending, int num, int width) {
          auto base = file_basename_no_extension(ending);
          auto ext = file_extension(ending);
          auto filename = base.empty()
              ? zero_pad_number(num, width) + "." + ext
              : base + "_" + zero_pad_number(num, width) + "." + ext;
          return (std::filesystem::path(output_path()) / filename).string();
        }

        std::string safe_filename_token(std::string token) {
          if (token.empty()) {
            throw std::runtime_error("output filename token must not be empty");
          }

          for (auto& ch : token) {
            const auto uch = static_cast<unsigned char>(ch);
            if (!std::isalnum(uch) && ch != '.' && ch != '_' && ch != '-') {
              ch = '_';
            }
          }

          return token;
        }

        std::string instance_filename(
            const std::string& instance_group,
            const std::string& instance_name,
            const std::string& extension) {
          namespace fs = std::filesystem;

          auto ext = extension;
          while (!ext.empty() && ext.front() == '.') {
            ext.erase(ext.begin());
          }
          if (ext.empty()) {
            throw std::runtime_error("output extension must not be empty");
          }

          auto directory = fs::path(output_path()) / safe_filename_token(instance_group);
          fs::create_directories(directory);

          auto filename = safe_filename_token(instance_name) + "." + ext;
          return (directory / filename).string();
        }

        std::string monitor_filename(const std::string& monitor_name, const std::string& extension) {
          auto ext = extension;
          while (!ext.empty() && ext.front() == '.') {
            ext.erase(ext.begin());
          }
          if (ext.empty()) {
            throw std::runtime_error("output extension must not be empty");
          }

          const auto filename = "monitor_" + safe_filename_token(monitor_name) + "." + ext;
          return (std::filesystem::path(output_path()) / filename).string();
        }

        std::string monitor_filename_series(
            const std::string& monitor_name,
            const std::string& extension,
            int num,
            int width) {
          const auto base = safe_filename_token(monitor_name) + "_" + zero_pad_number(num, width);
          return monitor_filename(base, extension);
        }

        std::string hamiltonian_filename(const std::string& hamiltonian_name, const std::string& extension) {
          return instance_filename("hamiltonians", hamiltonian_name, extension);
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
