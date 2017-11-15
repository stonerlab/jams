// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_UTILS_H
#define JAMS_CORE_UTILS_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include "jams/core/types.h"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

inline std::string& left_trim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
    std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
inline std::string& right_trim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
    std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
inline std::string& trim(std::string &s) {
  return left_trim(right_trim(s));
}

// capitalize a string
inline std::string capitalize(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), toupper);
  return s;
}

// capitalize a string
inline std::string lowercase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), tolower);
  return s;
}

inline std::string file_basename(std::string filepath) {
  int dot = filepath.find_last_of(".");
  int slash = filepath.find_last_of("/\\");
  return filepath.substr(slash+1, dot-slash-1);
}

inline bool string_is_comment(const std::string& s) {
  std::stringstream ss(s);
  char two_chars[2];

  ss >> two_chars[0];
  ss >> two_chars[1];
  // accept '#' or '//' as a comment character
  if ((!ss) || (two_chars[0] == '#') || (two_chars[0] == '/' && two_chars[1] == '/') ) {
    return true;
  }
  return false;
}

// Lifted from http://www.cplusplus.com/forum/general/15952/
inline std::string zero_pad_number(const int num) {
    std::ostringstream ss;
    ss << std::setw(7) << std::setfill('0') << num;
    std::string result = ss.str();
    if (result.length() > 7) {
        result.erase(0, result.length() - 7);
    }
    return result;
}

inline int file_columns(std::string &line) {
  std::stringstream is(line);
  std::string tmp;
  int count = 0;
  while (is >> tmp) {
    count++;
  }
  return count;
}

inline int periodic_shift(const int x, const int dimx) {
  return (x+dimx)%dimx;
}

std::string word_wrap(const char *text, size_t line_length);

template <typename T>
bool vec_exists_in_container(const T& container, const Vec3& v1, const double tolerance = 1e-6) {
  auto it = std::find_if(container.begin(),container.end(), [&](const Vec3& v2) {
      return equal(v1, v2, tolerance);
  });

  if (it == container.end()) {
    return false;
  }

  return true;
}

inline uint64_t concatenate_32_bit(uint32_t msw, uint32_t lsw) {
  return (uint64_t(msw) << 32) | lsw;
}

#endif  // JAMS_CORE_UTILS_H
