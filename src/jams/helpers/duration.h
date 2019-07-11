//
// Created by Joe Barker on 2017/11/16.
//

#ifndef JAMS_DURATION_H
#define JAMS_DURATION_H
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <array>
#include <string>

// using a struct instead of a tuple to avoid a bug in CUDA 9.0 with tuples
struct TimeOutputFormat {
    std::chrono::milliseconds denominator;
    int width;
    std::string separator;
};

template <typename Container, typename Fun>
void tuple_for_each(const Container& c, Fun fun)
{
  for (auto& e : c)
    fun(std::get<0>(e), std::get<1>(e), std::get<2>(e));
}

inline std::string duration_string(std::chrono::milliseconds time)
{
  using namespace std::chrono;

  const std::array<TimeOutputFormat, 4> formats = {
      TimeOutputFormat{hours(1), 2, ""},
      TimeOutputFormat{minutes(1), 2, ":"},
      TimeOutputFormat{seconds(1), 2, ":"},
      TimeOutputFormat{milliseconds(1), 3, "."}
  };

  std::ostringstream o;

  for (const auto &f : formats) {
    o << f.separator << std::setw(f.width) << std::setfill('0') << (time / f.denominator);
    time = time % f.denominator;
  }

  return o.str();
}

template <class T1, class T2>
inline std::string duration_string(T1 start_time, T2 end_time) {
  using namespace std::chrono;
  return duration_string(time_point_cast<milliseconds>(end_time) - time_point_cast<milliseconds>(start_time));
}


#endif //JAMS_DURATION_H
