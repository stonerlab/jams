// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_MATH_SUMMATIONS_H
#define JBLIB_MATH_SUMMATIONS_H

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

namespace jblib {
  float64 kahan_sum(const float64 * restrict data, const uint32 size);

  class KahanSum {
   public:
    // default constructor
    KahanSum() : count_(0), value_(0.0), compensation_(0.0) {};
    // copy constructor
    KahanSum(const KahanSum& other) : count_(other.count_), value_(other.value_), compensation_(other.compensation_) {};

    // destructor
    ~KahanSum() {};

    inline float64 value() {
        return value_;
    }

    inline float64 count() {
        return count_;
    }

    inline void add(const float64 &x) {
        float64 y, t;
        y = x - compensation_;
        t = value_ + y;
        compensation_ = (t - value_) - y;
        value_ = t;
        count_++;
    }

   private:
    int count_;
    float64 value_;
    float64 compensation_;
  };
}

#endif  // JBLIB_MATH_SUMMATIONS_H
