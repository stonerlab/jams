//
// Created by Joe Barker on 2018/08/28.
//

#ifndef JAMS_HELPERS_TIMER_H
#define JAMS_HELPERS_TIMER_H

#include <chrono>

template <typename Clock = std::chrono::high_resolution_clock>
class Timer
{
    const typename Clock::time_point start_time;
public:
    Timer() :
            start_time(Clock::now())
    {}

    template <typename Rep = double, typename Units = std::chrono::duration<double>>
    Rep elapsed_time() const {
      auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_time).count();
      return static_cast<Rep>(counted_time);
    }
};
#endif //JAMS_HELPERS_TIMER_H
