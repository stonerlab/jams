// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_MATH_INTEGERS_H
#define JBLIB_MATH_INTEGERS_H

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

namespace jblib {
  inline int32 nint(const float64 x) {
    return static_cast<int32>(x+0.5);
  }

  template <typename Type1, typename Type2>
  inline Type1 sign(const Type1 x, const Type2 y) {
    if (y >= 0.0) {
      return std::abs(x);
    } else {
      return -std::abs(x);
    }
  }
}  // namespace jblib

#endif  // JBLIB_MATH_INTEGERS_H
