#ifndef JB_MATH_SUMMATIONS_H
#define JB_MATH_SUMMATIONS_H

#include "../sys/defines.h"
#include "../sys/types.h"

namespace jblib {
  float64 kahanSum(const float64 *data, const uint32 size);
}

#endif
