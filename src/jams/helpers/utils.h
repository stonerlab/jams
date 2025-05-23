// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_UTILS_H
#define JAMS_CORE_UTILS_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"
#include "jams/helpers/maths.h"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

namespace jams {
    namespace util {
        // Forces containers to free any dynamical allocated memory (e.g. std::vector does not neccesarily free memory
        // even when emptied)
        template<class T>
        inline void force_deallocation(T &object) {
          T().swap(object);
        }

        template <class T>
        inline bool is_in_list(const T& value, std::initializer_list<T> list) {
          return std::find(list.begin(), list.end(), value) != list.end();
        }
    }
}

inline double division_or_zero(const double nominator, const double denominator) {
  if (denominator == 0.0) {
    return 0.0;
  } else {
    return nominator / denominator;
  }
}

inline std::string get_date_string(std::chrono::time_point<std::chrono::system_clock> t) {
  // https://stackoverflow.com/questions/34963738/c11-get-current-date-and-time-as-string
  auto as_time_t = std::chrono::system_clock::to_time_t(t);
  struct tm tm;
  if (::gmtime_r(&as_time_t, &tm)) {
    char timebuffer[80];
    if (std::strftime(timebuffer, sizeof(timebuffer), "%Y-%m-%d %H:%M:%S", &tm)) {
      return std::string{timebuffer};
    }
  }
  throw std::runtime_error("Failed to get current date as string");
}

inline std::string left_trim(std::string s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
      return !std::isspace(ch);}));
  return s;
}

// trim from end
inline std::string right_trim(std::string s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),[](unsigned char ch) {
      return !std::isspace(ch);
  }).base(), s.end());
  return s;
}

// trim from both ends
inline std::string trim(std::string s) {
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

inline std::string file_basename_no_extension(std::string filepath) {
  auto dot = filepath.find_last_of('.');
  auto slash = filepath.find_last_of("/\\");
  return filepath.substr(slash+1, dot-slash-1);
}

inline std::string file_basename(std::string filepath) {
  auto slash = filepath.find_last_of("/\\");
  return filepath.substr(slash+1);
}

inline std::string file_extension(std::string filepath) {
  auto dot = filepath.find_last_of('.');
  return filepath.substr(dot+1);
}



inline bool contains(const std::string& s1, const std::string& s2) {
  return s1.find(s2) != std::string::npos;
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

inline std::string string_wrap(std::string str, int width) {
  // https://stackoverflow.com/a/46836043
  int n = str.rfind(' ', width);
  if (n != std::string::npos) {
    str.at(n) = '\n';
  }
  return str;
}

// Lifted from http://www.cplusplus.com/forum/general/15952/
inline std::string zero_pad_number(const int num, const int width = 7) {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << num;
    std::string result = ss.str();
    if (result.length() > width) {
        result.erase(0, result.length() - width);
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

inline bool string_is_int(const std::string s){
  return s.find_first_not_of( "0123456789" ) == std::string::npos;
}

inline int periodic_shift(const int x, const int dimx) {
  return (x+dimx)%dimx;
}

std::string word_wrap(const char *text, size_t line_length);

// returns true if a Vec3 already exists in a container using safe floating point comparison
// to a given tolerance
template <typename T>
bool vec_exists_in_container(const T& container, const Vec3& v1, const double tolerance = FLT_EPSILON) {
  auto it = std::find_if(container.begin(),container.end(), [&](const Vec3& v2) {
      return approximately_equal(v1, v2, tolerance);
  });

  if (it == container.end()) {
    return false;
  }

  return true;
}

inline uint64_t concatenate_32_bit(uint32_t msw, uint32_t lsw) {
  return (uint64_t(msw) << 32) | lsw;
}

template <class T2, class A2 = std::allocator<T2>>
auto flatten_vector(const std::vector<T2, A2> &input) -> std::vector<typename T2::value_type> {
  std::vector<typename T2::value_type> result;
  for (const auto & v : input)
    result.insert(result.end(), v.begin(), v.end());
  return result;
}

// helper functions to make syntax shorter when apply lambda functions
template <class T, class F>
void apply_transform(std::vector<T> &x,  F func) {
  std::transform(x.begin(), x.end(), x.begin(), func);
}

template <class T, class UnaryPredicate>
void apply_predicate(std::vector<T> &x,  UnaryPredicate func) {
  x.erase(std::remove_if(x.begin(), x.end(), func), x.end());
}

// Tests if it is possible to dynamically cast from T2 to T1.
// This is mostly useful for checking if we can up or down
// cast between derived types.
template <typename T1, typename T2>
bool is_castable(const T2 x) {
  try {
    // https://en.cppreference.com/w/cpp/language/dynamic_cast
    auto y = dynamic_cast<T1>(x);
    if (std::is_pointer<T1>::value) {
      // for pointers dynamic_cast returns nullptr on failure
      if (y == nullptr) return false;
    }
  }
  catch (std::bad_cast &e) {
    // for references dynamic_cast throws an exception on failure
    return false;
  }
  return true;
};

inline std::string find_and_replace(std::string data, std::string find, std::string replace) {
  size_t pos = data.find(find);

  while(pos != std::string::npos) {
    data.replace(pos, find.size(), replace);
    pos = data.find(find, pos + replace.size());
  }
  return data;
}

inline std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> result;
  size_t last = 0;
  size_t next = 0;
  while ((next = s.find(delimiter, last)) != std::string::npos) {
    result.push_back(s.substr(last, next-last));
    last = next + 1;
  }
  result.push_back(s.substr(last));
  return result;
}

// Convert a string containing the output of git describe --tags --long into a
// semantic version string
inline std::string semantic_version(const std::string &git_description) {
  auto parts = split(git_description, "-");

  // this will be tag+commits_ahead.hash[.dirty], substr removes the 'g' at the front
  // of the hash which only stands for 'git' and is not part of the hash. If the
  // repo has uncommitted changes it is  marked with '.dirty' at the end
  if (parts.size() == 3) {
    return parts[0] + "+" + parts[1] + "." + parts[2].substr(1);
  } else if (parts.size() == 4) {
    return parts[0] + "+" + parts[1] + "." + parts[2].substr(1) + "." + parts[3];
  }

  // git description is not properly formatted to parse for semantic versioning
  // so just return as is.
  return git_description;
}

inline std::string memory_in_natural_units(std::size_t size) {
  std::map<int, std::string> byte_sizes;
  byte_sizes[0] = "B";
  byte_sizes[1] = "kB";
  byte_sizes[2] = "MB";
  byte_sizes[3] = "GB";
  byte_sizes[4] = "TB";
  byte_sizes[5] = "ZB";

  int factor = int(std::log2(size)/std::log2(1024));
  std::stringstream ss;
  ss << std::setprecision(3) <<  size / pow(1024,factor) << " " << byte_sizes[factor];
  return ss.str();
}

#endif  // JAMS_CORE_UTILS_H
