#ifndef JBLIB_UTILITIES_STRING_H
#define JBLIB_UTILITIES_STRING_H

#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iomanip>


#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

namespace jblib {

  inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
      std::not1(std::ptr_fun<int32, int32>(std::isspace))));
    return s;
  }

  // trim from end
  inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
      std::not1(std::ptr_fun<int32, int32>(std::isspace))).base(), s.end());
    return s;
  }

  // trim from both ends
  inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
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
  inline std::string zero_pad_number(int32 num) {
    std::ostringstream ss;
    ss << std::setw(7) << std::setfill('0') << num;
    std::string result = ss.str();
    if (result.length() > 7) {
      result.erase(0, result.length() - 7);
    }
    return result;
  }
}  // namespace jblib
#endif  // JBLIB_UTILITIES_STRING_H
