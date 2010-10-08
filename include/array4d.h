#ifndef __ARRAY4D_H__
#define __ARRAY4D_H__

#include "globals.h"

#include <valarray>
#include <cassert>

template <typename _Tp>
class Array4D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array4D() {
      for(int i=0; i<4; ++i) {
        dim[i] = 0;
      }
    }

    Array4D(size_type d0, size_type d1, size_type d2, size_type d3) {
      resize(d0,d1,d2,d3);
    }

    inline void resize(const size_type d0, const size_type d1, const size_type d2, const size_type d3) {
      dim[0] = d0; dim[1] = d1; dim[2] = d2; dim[3] = d3;
      data.resize(d0*d1*d2*d3);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k, const size_type l) {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      assert( (k >= 0) && (k < dim[2]) );
      assert( (l >= 0) && (l < dim[3]) );
      return data[((i*dim[1]+j)*dim[2]+k)*dim[3]+l];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k, const size_type l) const {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      assert( (k >= 0) && (k < dim[2]) );
      assert( (l >= 0) && (l < dim[3]) );
      return data[((i*dim[1]+j)*dim[2]+k)*dim[3]+l];
    }

    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim[4];
    std::valarray<_Tp> data;
};

#endif // __ARRAY4D_H__
