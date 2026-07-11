#ifndef JAMS_TEST_HELPERS_TEST_UTILS_H
#define JAMS_TEST_HELPERS_TEST_UTILS_H

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "jams/helpers/utils.h"

namespace {

class TimezoneGuard {
public:
  explicit TimezoneGuard(const char* timezone) {
    const char* original_timezone = std::getenv("TZ");
    if (original_timezone != nullptr) {
      original_timezone_ = original_timezone;
    }

    set_timezone(timezone);
  }

  ~TimezoneGuard() {
    if (original_timezone_.has_value()) {
      ::setenv("TZ", original_timezone_->c_str(), 1);
    } else {
      ::unsetenv("TZ");
    }
    ::tzset();
  }

  TimezoneGuard(const TimezoneGuard&) = delete;
  TimezoneGuard& operator=(const TimezoneGuard&) = delete;

private:
  void set_timezone(const char* timezone) {
    ::setenv("TZ", timezone, 1);
    ::tzset();
  }

  std::optional<std::string> original_timezone_;
};

std::chrono::system_clock::time_point time_point_from_time_t(std::time_t time) {
  return std::chrono::system_clock::from_time_t(time);
}

TEST(UtilsDateStringTest, FormatsLocalTimeWithBritishSummerTimeOffset) {
  TimezoneGuard timezone("GMT0BST,M3.5.0/1,M10.5.0/2");

  EXPECT_EQ(
      get_date_string(time_point_from_time_t(std::time_t{1783776600})),
      "2026-07-11 14:30:00 BST +0100");
}

TEST(UtilsDateStringTest, FormatsLocalTimeWithGreenwichMeanTimeOffset) {
  TimezoneGuard timezone("GMT0BST,M3.5.0/1,M10.5.0/2");

  EXPECT_EQ(
      get_date_string(time_point_from_time_t(std::time_t{1768138200})),
      "2026-01-11 13:30:00 GMT +0000");
}

}  // namespace

#endif  // JAMS_TEST_HELPERS_TEST_UTILS_H
