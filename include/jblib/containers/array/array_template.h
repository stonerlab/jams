// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_ARRAY_TEMPLATE_H
#define JBLIB_CONTAINERS_ARRAY_TEMPLATE_H

#include "jblib/sys/assert.h"
#include "jblib/sys/define.h"
#include "jblib/sys/types.h"
#include "jblib/sys/intrinsic.h"

#include <cstring>
#include <utility>
#include <algorithm>

namespace jblib {
  template <typename Tp_, int Dim_, typename Idx_ = int>
  class Array {
   public:
    Array();
    ~Array();
  };
}

#endif  // JBLIB_CONTAINERS_ARRAY_TEMPLATE_H
