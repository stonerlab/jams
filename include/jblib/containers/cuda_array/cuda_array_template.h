// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_CUDA_ARRAY_TEMPLATE_H
#define JBLIB_CONTAINERS_CUDA_ARRAY_TEMPLATE_H

#include "jblib/sys/types.h"
namespace jblib {
  template <typename Tp_, int Dim_, typename Idx_ = int>
  class CudaArray {
   public:
    CudaArray();
    ~CudaArray();
  };
}

#endif  // JBLIB_CONTAINERS_CUDA_ARRAY_TEMPLATE_H
