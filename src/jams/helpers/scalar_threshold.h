#ifndef JAMS_HELPERS_SCALAR_THRESHOLD_H
#define JAMS_HELPERS_SCALAR_THRESHOLD_H

#include <stdexcept>
#include "jams/helpers/finite.h"

namespace jams {

class ScalarThreshold {
 public:
  ScalarThreshold(double threshold, int consecutive_samples)
      : threshold_(threshold), required_(consecutive_samples) {
    if (!is_finite(threshold) || threshold <= 0.0) {
      throw std::runtime_error("convergence must be finite and positive");
    }
    if (required_ <= 0) {
      throw std::runtime_error("consecutive_samples must be positive");
    }
  }

  void update(double value) {
    if (!is_finite(value)) {
      reset();
      throw std::runtime_error("non-finite convergence measurement");
    }
    if (value < threshold_) {
      if (count_ < required_) ++count_;
    } else {
      reset();
    }
  }

  void reset() { count_ = 0; }
  bool satisfied() const { return count_ == required_; }

 private:
  double threshold_;
  int required_;
  int count_ = 0;
};

}  // namespace jams
#endif
