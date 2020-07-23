//
// Created by Joseph Barker on 2018-12-04.
//

#ifndef JAMS_PROGRESS_BAR_H
#define JAMS_PROGRESS_BAR_H

#include <iomanip>

#include "jams/interface/system.h"

class ProgressBar {
public:
    ProgressBar() = default;
    explicit inline ProgressBar(unsigned width) : width_(width) {};

    inline void set(const float &x) {
      progress_ = x;
    }

    inline void add(const float &x) {
      progress_ += x;
    }

    inline void reset() {
      progress_ = 0.0;
    }

    inline float progress() const {
      return progress_;
    }

    inline float percent() const {
      return progress_ * 100.0f;
    }

    inline void width(const unsigned w) {
      width_ = w;
    }

    inline unsigned width() const {
      return width_;
    }

    inline bool do_next_static_output() {
      if (static_cast<unsigned>(percent()) == static_next_) {
        static_next_ += static_resolution_;
        return true;
      }
      return false;
    };

private:
    unsigned width_    = 72;
    float    progress_ = 0.0;
    unsigned static_resolution_ = 10; // percent
    unsigned static_next_ = 10; // percent
};

inline std::ostream& operator<<(std::ostream& os, ProgressBar &p) {
  if (jams::system::stdout_is_tty()) {
    auto pos = static_cast<unsigned>(p.width() * p.progress());
    os << "\r[";
    for (auto i = 0; i < p.width(); ++i) {
      os << ((i <= pos) ? "=" : " ");
    }
    os << "] ";
    os << std::setw(3) << static_cast<unsigned>(p.percent()) << " %" << std::flush;
  } else {
    if (p.do_next_static_output()) {
      os << "..." << static_cast<unsigned>(p.percent()) << "%" << std::flush;
    }
  }
  return os;
}

#endif //JAMS_PROGRESS_BAR_H
