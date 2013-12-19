#ifndef JB_CUDAARRAY_H
#define JB_CUDAARRAY_H

#include "../sys/defines.h"
#include "../sys/types.h"

namespace jblib{
  template <typename Tp_, uint32 Dim_, typename Idx_=uint32>
    class CudaArray 
    {
      public:
        CudaArray(){}
        ~CudaArray(){};
    };
}

#include "cuda_array1d.h"

#endif
