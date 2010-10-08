#ifndef __ARRAY2D_H__
#define __ARRAY2D_H__

#include "globals.h"

#include <valarray>
#include <cassert>

template <typename _Tp>
class Array2D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array2D() {
      for(int i=0; i<2; ++i) {
        dim[i] = 0;
      }
    }

    Array2D(size_type d0, size_type d1) {
      resize(d0,d1,d2);
    }

    inline void resize(size_type d0, size_type d1) {
      dim[0] = d0; dim[1] = d1;
      data.resize(d0*d1);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k) {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      return data[i*dim[1]+j];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k) const {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      return data[i*dim[1]+j];
    }

    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim[2];
    std::valarray<_Tp> data;
};

#endif // __ARRAY2D_H__
