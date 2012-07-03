#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iomanip>

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

// Lifted from http://www.cplusplus.com/forum/general/15952/
std::string zero_pad_num(int num)
{
    std::ostringstream ss;
    ss << std::setw(7) << std::setfill('0') << num;
    std::string result = ss.str();
    if (result.length() > 7)
    {
        result.erase(0, result.length() - 7);
    }
    return result;
}

#endif // __UTILS_H__
