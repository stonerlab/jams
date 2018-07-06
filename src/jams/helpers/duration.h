//
// Created by Joe Barker on 2017/11/16.
//

#ifndef JAMS_DURATION_H
#define JAMS_DURATION_H
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>

//http://cppcodereviewers.com/time-duration-conversion-to-date-format/

template <typename Container, typename Fun>
void tuple_for_each(const Container& c, Fun fun)
{
  for (auto& e : c)
    fun(std::get<0>(e), std::get<1>(e), std::get<2>(e));
}

inline std::string duration_string(std::chrono::milliseconds time)
{
  using namespace std::chrono;

  using T = std::tuple<milliseconds, int, const char *>;

  const T formats[] = {
          T{hours(1), 2, ""},
          T{minutes(1), 2, ":"},
          T{seconds(1), 2, ":"},
          T{milliseconds(1), 3, "."}
  };

  std::ostringstream o;
  tuple_for_each(formats, [&time, &o](milliseconds denominator, int width, const char * separator) {
      o << separator << std::setw(width) << std::setfill('0') << (time / denominator);
      time = time % denominator;
  });
  return o.str();
}
#endif //JAMS_DURATION_H
