#ifndef JB_SPARSEMATRIX_H
#define JB_SPARSEMATRIX_H

#include "../sys/sys_assert.h"
#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"
#include "../sys/sys_intrinsics.h"

#include <cstring>
#include <utility>
#include <algorithm>


namespace jbLib{

  template<class Iter, class T>
    Iter binary_find(Iter begin, Iter end, T val)
    {
      // Finds the lower bound in at most log(last - first) + 1 comparisons
      Iter i = std::lower_bound(begin, end, val);

      if (i != end && *i == val)
        return i; // found
      else
        return end; // not found
    }

  // taken from CUSP
  template< class type1, class type2>
    bool kv_pair_less(const std::pair<type1,type2>&x, const std::pair<type1,type2>&y){
      return x.first < y.first;
    }


  template <typename valueType_, uint32 dimension, typename indexType_=uint32>
    class Sparsematrix 
    {
      public:
        Sparsematrix(){}
        ~Sparsematrix(){};
    };

}

//#include "Sparsematrix3D.h"
#include "Sparsematrix4D.h"

#endif

