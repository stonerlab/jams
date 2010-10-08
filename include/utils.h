#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <sstream>
#include <locale>
#include <functional>
#include <algorithm>
#include <cmath>

inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
inline std::string &rtrim(std::string &s) {
s.erase(std::find_if(s.rbegin(), s.rend(),std::not1(std::ptr_fun<int,int>(std::isspace))).base(), s.end());
return s;
}

// trim from both ends
inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}

#endif // __UTILS_H__
