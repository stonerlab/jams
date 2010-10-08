#ifndef __ARRAY3D_H__
#define __ARRAY3D_H__

#include "globals.h"

#include <valarray>
#include <cassert>

template <typename _Tp>
class Array3D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array3D() {
      for(int i=0; i<3; ++i) {
        dim[i] = 0;
      }
    }

    Array3D(size_type d0, size_type d1, size_type d2) {
      resize(d0,d1,d2);
    }

    inline void resize(size_type d0, size_type d1, size_type d2) {
      dim[0] = d0; dim[1] = d1; dim[2] = d2;
      data.resize(d0*d1*d2);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k) {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      assert( (k >= 0) && (k < dim[2]) );
      return data[(i*dim[1]+j)*dim[2]+k];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k) const {
      assert( (i >= 0) && (i < dim[0]) );
      assert( (j >= 0) && (j < dim[1]) );
      assert( (k >= 0) && (k < dim[2]) );
      return data[(i*dim[1]+j)*dim[2]+k];
    }

    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim[3];
    std::valarray<_Tp> data;
};

#endif // __ARRAY3D_H__
