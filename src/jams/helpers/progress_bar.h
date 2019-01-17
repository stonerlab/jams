//
// Created by Joseph Barker on 2018-12-04.
//

#ifndef JAMS_PROGRESS_BAR_H
#define JAMS_PROGRESS_BAR_H

#include <iomanip>

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

    inline void width(const unsigned w) {
      width_ = w;
    }

    inline unsigned width() const {
      return width_;
    }

private:
    unsigned width_    = 72;
    float    progress_ = 0.0;
};

inline std::ostream& operator<<(std::ostream& os, const ProgressBar &p) {

  auto pos = static_cast<unsigned>(p.width() * p.progress());
  os << "\r[";
  for (auto i = 0; i < p.width(); ++i) {
    if (i <= pos) {
      os << "=";
    } else {
      os << " ";
    }
  }
  os << "] ";
  os << std::setw(3) << static_cast<unsigned>(p.progress() * 100.0) << " %" << std::flush;
  return os;
}

#endif //JAMS_PROGRESS_BAR_H
