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

inline std::string file_basename(std::string filepath) {
  int dot = filepath.find_last_of(".");
  int slash = filepath.find_last_of("/\\");
  return filepath.substr(slash+1, dot-slash-1);
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

#endif  // JAMS_CORE_UTILS_H
